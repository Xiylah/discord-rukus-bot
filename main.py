import discord
import os
from rapidfuzz import fuzz

TOKEN = os.getenv("TOKEN")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

# -------------------------
# CONFIG
# -------------------------

EVENT_CHANNEL_ID = 1458936961044709539

EVENT_REPLY = f"Please check <#{EVENT_CHANNEL_ID}>, anything related to events or updates will be posted there."

SECRET_CODE_REPLY = (
    f"Currently there are no secret codes. Keep an eye on <#{EVENT_CHANNEL_ID}> "
    "if we do drop any in the future!"
)

# -------------------------
# PATTERNS (ONLY REAL QUESTIONS)
# -------------------------

EVENT_PATTERNS = [
    "when is the next event",
    "any upcoming events",
    "what events are",
    "is there an event",
    "event this weekend",
]

SECRET_CODE_PATTERNS = [
    "what is the secret code",
    "what's the secret code",
    "is there a secret code",
    "do you have a code",
    "what is the code",
]

# -------------------------
# HELPERS
# -------------------------

def is_question(message: str) -> bool:
    message = message.lower().strip()

    return (
        "?" in message
        or message.startswith("what")
        or message.startswith("when")
        or message.startswith("is")
        or message.startswith("are")
        or message.startswith("do")
        or message.startswith("can")
    )


def fuzzy_match(message: str, patterns: list[str], threshold: int = 90) -> bool:
    message = message.lower().strip()

    # Ignore very short / low-context messages
    if len(message) < 12:
        return False

    for pattern in patterns:
        score = fuzz.partial_ratio(message, pattern)
        if score >= threshold:
            return True

    return False


# -------------------------
# INTENT DETECTION
# -------------------------

def is_event_question(message: str) -> bool:
    return is_question(message) and fuzzy_match(message, EVENT_PATTERNS)


def is_secret_code_question(message: str) -> bool:
    return is_question(message) and fuzzy_match(message, SECRET_CODE_PATTERNS)


# -------------------------
# BOT LOGIC
# -------------------------

@client.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content

    # Event questions
    if is_event_question(content):
        await message.reply(EVENT_REPLY)
        return

    # Secret code questions
    if is_secret_code_question(content):
        await message.reply(SECRET_CODE_REPLY)
        return


client.run(TOKEN)
