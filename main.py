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

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# IMPROVED EVENT EXAMPLES (More Specific)
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
    
    # More specific "Admin Abuse" examples
    "when is admin abuse",
    "admin abuse this weekend",
    "is there admin abuse this week",
    "when is the next admin abuse",
    "admin abuse event",
    "admin abuse schedule",
    "is admin abuse happening soon",
    "admin abuse date",
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
# HELPER FUNCTIONS
# -------------------------
def is_admin_abuse_query(text: str) -> bool:
    """More strict check for admin abuse questions"""
    text_lower = text.lower()
    
    # Must contain "admin abuse" AND some context word
    if "admin abuse" not in text_lower and "adminabuse" not in text_lower.replace(" ", ""):
        return False
    
    # Context keywords that suggest they're asking about timing/schedule
    context_keywords = ["when", "next", "this weekend", "this week", "soon", "today", "tomorrow", 
                       "schedule", "date", "happening", "start", "event", "time"]
    
    return any(word in text_lower for word in context_keywords)

def get_best_similarity(message: str, embeddings):
    message_emb = model.encode(message, convert_to_tensor=True)
    similarities = util.cos_sim(message_emb, embeddings)
    return similarities.max().item()

# -------------------------
# MAIN EVENT
# -------------------------
@client.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content.strip()
    if not content:
        return

    content_lower = content.lower()

    # === STRICT ADMIN ABUSE CHECK ===
    if is_admin_abuse_query(content_lower):
        await message.reply(EVENT_REPLY)
        return

    # Skip very short messages
    if len(content_lower) < 8:
        return

    # === SEMANTIC CHECKS ===
    event_score = get_best_similarity(content_lower, event_embeddings)
    secret_score = get_best_similarity(content_lower, secret_embeddings)

    if event_score > 0.65:
        await message.reply(EVENT_REPLY)
    elif secret_score > 0.68:
        await message.reply(SECRET_CODE_REPLY)

client.run(TOKEN)
