import discord
import os
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

TOKEN = os.getenv("TOKEN")
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

EVENT_CHANNEL_ID = 1458936961044709539
FURNITURE_CHANNEL_ID = 1510456653085020290
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
    "is there a live event",
    "is there a live event in the game",
    "is there currently a live event",
    "guys is there an event",
    "does anyone know if there's an event",
    "has anyone heard about an upcoming event",
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
    "does anyone know the code",
    "has anyone figured out the code",
    "can someone give me hints on the code",
    "what is the code i need to use",
    "is there a code to use",
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
    "no live event at this time",
    "they have mini events all the time",
    "there are no events right now",
    "the event already ended",
    "we just had an event",
    "events will be posted when ready",
    "admin abuse will be announced",
    "the code has been used",
    "no secret codes currently",
]

# Filler phrases to strip before semantic scoring
FILLER_PATTERNS = [
    r"^(hey\s+)?(guys|everyone|all|y'all|folks)\b[,\s]*",
    r"^(hi|hey|hello|yo|sup)\b[,\s]*",
    r"\bhope\s+(everyone('s|s|is)\s+\w+\s*)+",
    r"\bjust\s+wanted\s+to\s+(see|ask|check|know)\s+(if\s+)?",
    r"\bthank\s+you(\s+in\s+advance)?\b",
    r"\bcan\s+(maybe|possibly|you)\s+",
    r"\band\s+can\s+(maybe|possibly)?\s*",
]

# Declarative sentence patterns — these are statements, not questions
DECLARATIVE_PATTERNS = [
    r"^(there\s+)(are\s+no|is\s+no|aren't\s+any|isn't\s+any)\b",
    r"^no\s+\w+\s+at\s+this\s+time",
    r"^they\s+(have|had|will\s+have)\b",
    r"^(it|this)\s+(was|is|will\s+be)\b",
    r"^(the|an?)\s+\w+\s+(was|is|has\s+been|will\s+be)\b",
    r"^(i\s+)(went|used|did|was|had|got)\b",
    r"^(we\s+)(just|already|recently)\b",
    r"^(admin\s+abuse|events?)\s+(will\s+be|are\s+going\s+to\s+be)\s+(announced|posted)\b",
    r"^lol\b",
]

# Strong question indicators (auxiliary inversion = almost certainly a question)
AUX_INVERSION = re.compile(
    r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\s",
    re.IGNORECASE
)

WH_QUESTION = re.compile(
    r"^(what|when|where|who|why|how|which|whose)\b",
    re.IGNORECASE
)

QUESTION_MARK = re.compile(r"\?")

# Softer signals used as a last resort
SOFT_SIGNALS = [
    "any upcoming", "any active", "any events", "any codes", "any pins",
    "give me the", "tell me the", "need the code", "need to know",
]

AUTO_LEARN_WINDOW = 0.12
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


# --- CLASSIFICATION ---

def strip_filler(text: str) -> str:
    """Remove common filler phrases to expose the core intent."""
    cleaned = text.strip()
    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    # Collapse extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 6 else text  # Safety: don't over-strip

def is_declarative(text: str) -> bool:
    """Return True if the text is almost certainly a statement, not a question."""
    t = text.strip().lower()
    return any(re.match(p, t) for p in DECLARATIVE_PATTERNS)

def is_question(text: str) -> bool:
    """
    Determine if a message is a genuine question using grammatical cues,
    not just keyword presence.
    """
    t = text.strip()
    t_lower = t.lower()

    # Hard no: declarative structure overrides everything
    if is_declarative(t_lower):
        return False

    # Strong yes: ends with ? (explicit question mark)
    if QUESTION_MARK.search(t):
        return True

    # Strong yes: starts with auxiliary inversion ("Is there", "Do you", "Can anyone")
    if AUX_INVERSION.match(t):
        return True

    # Strong yes: starts with a WH-word ("When is", "What is", "How do")
    if WH_QUESTION.match(t):
        return True

    # Medium yes: soft signals after filler stripping
    core = strip_filler(t_lower)
    if any(sig in core for sig in SOFT_SIGNALS):
        return True

    return False

def is_admin_abuse_query(text: str) -> bool:
    text_lower = text.lower()
    if "admin abuse" not in text_lower and "adminabuse" not in text_lower.replace(" ", ""):
        return False
    return any(w in text_lower for w in [
        "when", "next", "this weekend", "this week", "soon", "today",
        "tomorrow", "schedule", "date", "happening", "start", "event", "time", "?"
    ])

def get_scores(message: str):
    """Score the semantic core (filler-stripped) text against all example banks."""
    core = strip_filler(message)
    msg_emb = model.encode(core, convert_to_tensor=True)

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

    # Silently delete text-only messages in the furniture channel
    if message.channel.id == FURNITURE_CHANNEL_ID:
        has_attachment = len(message.attachments) > 0
        has_embed_image = any(e.image or e.thumbnail for e in message.embeds)
        if not has_attachment and not has_embed_image:
            await message.delete()
        return

    content = message.content.strip()
    if not content or len(content) < 8:
        return

    content_lower = content.lower()

    # Fast path: explicit admin abuse scheduling query
    if is_admin_abuse_query(content_lower):
        await message.reply(EVENT_REPLY)
        return

    # Gate: must be a genuine question (grammatical check, not keyword check)
    if not is_question(content):
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
        negative_is_dominant = negative_score > max(event_score, secret_score)
        if negative_is_dominant:
            return

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
