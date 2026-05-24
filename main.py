import discord
import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

TOKEN = os.getenv("TOKEN")
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

EVENT_CHANNEL_ID = 1458936961044709539
EVENT_REPLY = f"Please check <#{EVENT_CHANNEL_ID}>, anything related to events or updates will be posted there."
SECRET_CODE_REPLY = f"Currently there are no secret codes. Keep an eye on <#{EVENT_CHANNEL_ID}> if we do drop any in the future!"

LEARNED_EXAMPLES_FILE = Path("learned_examples.json")

EVENT_EXAMPLES = [
    "when is the next event",
    "any upcoming events",
    "what events are coming up",
    "is there an event this weekend",
    "when's the next update",
    "event schedule",
    "are we having an event soon",
    "when is admin abuse",
    "admin abuse this weekend",
    "is there admin abuse this week",
    "when is the next admin abuse",
    "admin abuse schedule",
    "when does admin abuse start",
    "is admin abuse happening",
    "what time is the next event",
    "is there an event today",
    "any events planned",
]

SECRET_CODE_EXAMPLES = [
    "what is the secret code",
    "what's the secret code right now",
    "is there a secret code",
    "current secret code",
    "any active codes",
    "is there a code right now",
    "what is the pin code",
    "what's the pin code",
    "is there a pin code",
    "do you have a pin",
    "what is the pin",
    "current pin code",
    "any active pin",
    "is there a pin right now",
    "give me the pin",
    "what pin do i use",
]

NEGATIVE_EXAMPLES = [
    "that event was fun",
    "i went to an event yesterday",
    "admin abuse is so annoying",
    "this event was terrible",
    "lol admin abuse happened to me",
    "the last event was great",
    "i love events",
    "admin abuse is wild",
    "codes are hard to remember",
    "i used the code already",
]

QUESTION_SIGNALS = [
    "when", "what", "is there", "any", "do you", "does", "will", "are",
    "how", "where", "?", "gonna", "going to", "happening", "schedule",
    "date", "time", "soon", "next", "today", "tomorrow", "this week", "this weekend"
]

# How close a near-miss must be to the threshold to be auto-learned
AUTO_LEARN_WINDOW = 0.12
# The reply thresholds
EVENT_THRESHOLD = 0.72
SECRET_THRESHOLD = 0.74
NEGATIVE_PENALTY = 0.02


# --- PERSISTENCE ---

def load_learned_examples():
    if not LEARNED_EXAMPLES_FILE.exists():
        return [], []
    try:
        data = json.loads(LEARNED_EXAMPLES_FILE.read_text())
        return data.get("event", []), data.get("secret", [])
    except Exception as e:
        print(f"[warn] Could not load learned_examples.json: {e}")
        return [], []

def save_learned_example(text: str, label: str):
    data = {"event": [], "secret": []}
    if LEARNED_EXAMPLES_FILE.exists():
        try:
            data = json.loads(LEARNED_EXAMPLES_FILE.read_text())
        except Exception:
            pass
    if label in data and text not in data[label]:
        data[label].append(text)
        LEARNED_EXAMPLES_FILE.write_text(json.dumps(data, indent=2))
        print(f"[learn] Saved {label} example: {text!r}")


# --- MODEL + EMBEDDINGS ---

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

extra_event, extra_secret = load_learned_examples()
all_event_examples = EVENT_EXAMPLES + extra_event
all_secret_examples = SECRET_CODE_EXAMPLES + extra_secret

event_embeddings = model.encode(all_event_examples, convert_to_tensor=True)
secret_embeddings = model.encode(all_secret_examples, convert_to_tensor=True)
negative_embeddings = model.encode(NEGATIVE_EXAMPLES, convert_to_tensor=True)

print(f"Ready. {len(all_event_examples)} event examples, {len(all_secret_examples)} secret examples.")


def add_live_embedding(text: str, label: str):
    global event_embeddings, secret_embeddings
    new_emb = model.encode(text, convert_to_tensor=True).unsqueeze(0)
    if label == "event":
        event_embeddings = torch.cat([event_embeddings, new_emb], dim=0)
        all_event_examples.append(text)
    elif label == "secret":
        secret_embeddings = torch.cat([secret_embeddings, new_emb], dim=0)
        all_secret_examples.append(text)


# --- SCORING ---

def has_question_intent(text: str) -> bool:
    return any(signal in text.lower() for signal in QUESTION_SIGNALS)

def is_admin_abuse_query(text: str) -> bool:
    text_lower = text.lower()
    if "admin abuse" not in text_lower and "adminabuse" not in text_lower.replace(" ", ""):
        return False
    return any(w in text_lower for w in [
        "when", "next", "this weekend", "this week", "soon", "today",
        "tomorrow", "schedule", "date", "happening", "start", "event", "time", "?"
    ])

def get_scores(message: str):
    msg_emb = model.encode(message, convert_to_tensor=True)
    def top_mean(emb_bank, k=2):
        sims = util.cos_sim(msg_emb, emb_bank)[0]
        top_k = torch.topk(sims, min(k, len(sims))).values
        return top_k.mean().item()
    return top_mean(event_embeddings), top_mean(secret_embeddings), top_mean(negative_embeddings)


# --- BOT ---

@client.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content.strip()
    if not content or len(content) < 8:
        return

    content_lower = content.lower()

    if is_admin_abuse_query(content_lower):
        await message.reply(EVENT_REPLY)
        return

    if not has_question_intent(content_lower):
        return

    event_score, secret_score, negative_score = get_scores(content_lower)

    penalty = max(0, (negative_score - 0.55) * 2) * NEGATIVE_PENALTY
    adj_event = event_score - penalty
    adj_secret = secret_score - penalty

    if adj_event > EVENT_THRESHOLD:
        await message.reply(EVENT_REPLY)

    elif adj_secret > SECRET_THRESHOLD:
        await message.reply(SECRET_CODE_REPLY)

    else:
        # Auto-learn: if the message is close to a threshold and scores
        # higher on that category than the negative bank, treat it as a
        # genuine question and absorb it as a new example.
        negative_is_dominant = negative_score > max(event_score, secret_score)
        if negative_is_dominant:
            return  # Looks like a statement, not a question — ignore

        near_event = EVENT_THRESHOLD - AUTO_LEARN_WINDOW < adj_event < EVENT_THRESHOLD
        near_secret = SECRET_THRESHOLD - AUTO_LEARN_WINDOW < adj_secret < SECRET_THRESHOLD

        if near_event and adj_event > adj_secret:
            add_live_embedding(content_lower, "event")
            save_learned_example(content_lower, "event")
            await message.reply(EVENT_REPLY)
        elif near_secret and adj_secret > adj_event:
            add_live_embedding(content_lower, "secret")
            save_learned_example(content_lower, "secret")
            await message.reply(SECRET_CODE_REPLY)

client.run(TOKEN)
