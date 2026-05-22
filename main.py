import discord
import os
from rapidfuzz import fuzz

TOKEN = os.getenv("TOKEN")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

# Channel to direct users to for events
EVENT_CHANNEL_ID = 1458936961044709539

EVENT_REPLY = f"Please check <#{EVENT_CHANNEL_ID}>, anything related to events or updates will be posted there."

SECRET_CODE_REPLY = f"Currently there is no secret codes. Keep an eye on <#{EVENT_CHANNEL_ID}> if we do drop any codes in the future!"

# -------------------------
# PATTERNS
# -------------------------

EVENT_PATTERNS = [
    "is there going to be an event",
    "any event this weekend",
    "are there events",
    "when is the next event",
    "events coming up",
    "event this weekend",
    "any upcoming events",
]

SECRET_CODE_PATTERNS = [
    "what's the secret code",
    "secret hidden code",
    "what is the code",
    "do you know the code",
    "code for the server",
    "hidden code",
    "event code",
]

# -------------------------
# FUZZY MATCH HELPERS (FIXED)
# -------------------------

def fuzzy_match(message: str, patterns: list[str], threshold: int = 85) -> bool:
    message = message.lower().strip()

    # Ignore very short messages completely (fixes "no", "ya", etc.)
    if len(message) < 10:
        return False

    for pattern in patterns:
        score = fuzz.partial_ratio(message, pattern)
        if score >= threshold:
            return True

    return False


def contains_keyword(message: str, keywords: list[str]) -> bool:
    message = message.lower()
    return any(k in message for k in keywords)


# -------------------------
# STRONGER INTENT CHECKS
# -------------------------

EVENT_KEYWORDS = ["event", "events"]

SECRET_CODE_KEYWORDS = ["secret code", "hidden code", "code"]


def is_event_question(message: str) -> bool:
    return (
        contains_keyword(message, EVENT_KEYWORDS)
        or fuzzy_match(message, EVENT_PATTERNS)
    )


def is_secret_code_question(message: str) -> bool:
    return (
        contains_keyword(message, SECRET_CODE_KEYWORDS)
        or fuzzy_match(message, SECRET_CODE_PATTERNS)
    )


# -------------------------
# BOT LOGIC
# -------------------------

@client.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content

    # Event-related questions
    if is_event_question(content):
        await message.reply(EVENT_REPLY)
        return

    # Secret code questions
    if is_secret_code_question(content):
        await message.reply(SECRET_CODE_REPLY)
        return


client.run(TOKEN)
