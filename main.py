import discord
import os
from sentence_transformers import SentenceTransformer, util

TOKEN = os.getenv("TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# -------------------------
# CONFIG
# -------------------------
EVENT_CHANNEL_ID = 1458936961044709539

EVENT_REPLY = f"Please check <#{EVENT_CHANNEL_ID}>, anything related to events or updates will be posted there."
SECRET_CODE_REPLY = f"Currently there are no secret codes. Keep an eye on <#{EVENT_CHANNEL_ID}> if we do drop any in the future!"

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# SEMANTIC EXAMPLES
# -------------------------
EVENT_EXAMPLES = [
    "when is the next event",
    "any upcoming events",
    "what events are coming up",
    "is there an event this weekend",
    "when's the next update",
    "any events soon",
    "event schedule",
    "are we having an event",
    # Admin Abuse related (now part of events)
    "is there any admin abuse this weekend",
    "any admin abuse",
    "when is admin abuse",
    "admin abuse this week",
    "when is the next admin abuse",
    "is admin abuse happening",
    "admin abuse schedule",
    "any admin abuse soon",
    "admin abuse event",
    "when does admin abuse start",
]

SECRET_CODE_EXAMPLES = [
    "what is the secret code",
    "what's the secret code right now",
    "is there a secret code",
    "do you have any codes",
    "give me the code",
    "current secret code",
]

# Pre-compute embeddings
event_embeddings = model.encode(EVENT_EXAMPLES, convert_to_tensor=True)
secret_embeddings = model.encode(SECRET_CODE_EXAMPLES, convert_to_tensor=True)

# -------------------------
# INTENT DETECTION
# -------------------------
def get_best_similarity(message: str, embeddings):
    message_emb = model.encode(message, convert_to_tensor=True)
    similarities = util.cos_sim(message_emb, embeddings)
    return similarities.max().item()

@client.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content.lower().strip()
   
    if len(content) < 8:
        return

    # Get similarity scores
    event_score = get_best_similarity(content, event_embeddings)
    secret_score = get_best_similarity(content, secret_embeddings)

    # Check intents
    if event_score > 0.65:
        await message.reply(EVENT_REPLY)
    elif secret_score > 0.68:
        await message.reply(SECRET_CODE_REPLY)

client.run(TOKEN)
