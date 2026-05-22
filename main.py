import discord
import os
from rapidfuzz import fuzz

TOKEN = os.getenv("TOKEN")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

REPLY = "Please check #🔔┇map-announcements, anything related to events or updates will be posted there."

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