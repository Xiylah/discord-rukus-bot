import discord
import os
from sentence_transformers import SentenceTransformer, util
import torch

TOKEN = os.getenv("TOKEN")
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

EVENT_CHANNEL_ID = 1458936961044709539
EVENT_REPLY = f"Please check <#{EVENT_CHANNEL_ID}>, anything related to events or updates will be posted there."
SECRET_CODE_REPLY = f"Currently there are no secret codes. Keep an eye on <#{EVENT_CHANNEL_ID}> if we do drop any in the future!"

model = SentenceTransformer('all-MiniLM-L6-v2')

# --- QUESTION INTENT SIGNALS ---
# Message must contain at least one of these to even be considered
QUESTION_SIGNALS = [
    "when", "what", "is there", "any", "do you", "does", "will", "are",
    "how", "where", "?", "gonna", "going to", "happening", "schedule",
    "date", "time", "soon", "next", "today", "tomorrow", "this week", "this weekend"
]

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

# Negative examples — things that LOOK related but aren't questions
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

event_embeddings = model.encode(EVENT_EXAMPLES, convert_to_tensor=True)
secret_embeddings = model.encode(SECRET_CODE_EXAMPLES, convert_to_tensor=True)
negative_embeddings = model.encode(NEGATIVE_EXAMPLES, convert_to_tensor=True)


def has_question_intent(text: str) -> bool:
    """Only proceed if the message looks like a genuine inquiry."""
    text_lower = text.lower()
    return any(signal in text_lower for signal in QUESTION_SIGNALS)


def get_scores(message: str):
    """Returns (event_score, secret_score, negative_score)"""
    msg_emb = model.encode(message, convert_to_tensor=True)

    # Use mean of top-2 instead of max to reduce single-word false positives
    def top_mean(emb_bank, k=2):
        sims = util.cos_sim(msg_emb, emb_bank)[0]
        top_k = torch.topk(sims, min(k, len(sims))).values
        return top_k.mean().item()

    event_score = top_mean(event_embeddings)
    secret_score = top_mean(secret_embeddings)
    negative_score = top_mean(negative_embeddings)

    return event_score, secret_score, negative_score


def is_admin_abuse_query(text: str) -> bool:
    text_lower = text.lower()
    if "admin abuse" not in text_lower and "adminabuse" not in text_lower.replace(" ", ""):
        return False
    context_keywords = [
        "when", "next", "this weekend", "this week", "soon", "today",
        "tomorrow", "schedule", "date", "happening", "start", "event", "time", "?"
    ]
    return any(word in text_lower for word in context_keywords)


@client.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content.strip()
    if not content or len(content) < 8:
        return

    content_lower = content.lower()

    # Hard-coded admin abuse check (fast path)
    if is_admin_abuse_query(content_lower):
        await message.reply(EVENT_REPLY)
        return

    # Gate: must look like a question/inquiry
    if not has_question_intent(content_lower):
        return

    event_score, secret_score, negative_score = get_scores(content_lower)

    # Raise thresholds AND penalize if it scores high on negatives
    EVENT_THRESHOLD = 0.72
    SECRET_THRESHOLD = 0.74
    NEGATIVE_PENALTY = 0.02  # subtract this per 0.01 above 0.55 negative score

    # Dampen scores if message resembles a non-question statement
    penalty = max(0, (negative_score - 0.55) * 2) * NEGATIVE_PENALTY
    event_score -= penalty
    secret_score -= penalty

    if event_score > EVENT_THRESHOLD:
        await message.reply(EVENT_REPLY)
    elif secret_score > SECRET_THRESHOLD:
        await message.reply(SECRET_CODE_REPLY)


client.run(TOKEN)
