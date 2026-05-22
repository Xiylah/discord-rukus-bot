import discord
import os
from rapidfuzz import fuzz

TOKEN = os.getenv("TOKEN")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

# Replace with your actual channel ID (recommended)
CHANNEL_ID = 1458936961044709539  # ← CHANGE THIS

REPLY = f"Please check <#{CHANNEL_ID}>, anything related to events or updates will be posted there."

# These are "intent examples" (not strict keywords)
EVENT_PATTERNS = [
    "is there going to be an event",
    "any event this weekend",
    "are there events",
    "when is the next event",
    "events coming up",
    "event this weekend",
    "any upcoming events",
]

def is_event_question(message: str) -> bool:
    message = message.lower()

    for pattern in EVENT_PATTERNS:
        score = fuzz.partial_ratio(message, pattern)
        if score >= 75:   # sensitivity (you can tweak this)
            return True

    return False


@client.event
async def on_message(message):
    if message.author.bot:
        return

    if is_event_question(message.content):
        await message.reply(REPLY)


client.run(TOKEN)
