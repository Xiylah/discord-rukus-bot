import discord
import os
import re

TOKEN = os.getenv("TOKEN")
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

EVENT_CHANNEL_ID = 1458936961044709539
FURNITURE_CHANNEL_ID = 1510456653085020290
SUPPORT_CHANNEL_ID = 1478734423972384799
EVENT_REPLY = f"Please check <#{EVENT_CHANNEL_ID}>, anything related to events or updates will be posted there."
SUPPORT_REPLY = f"Please file a support ticket here: <#{SUPPORT_CHANNEL_ID}> and we can help you out!"


# --- PATTERNS ---

# Filler phrases to strip before matching
FILLER_PATTERNS = [
    r"^(hey\s+)?(guys|everyone|all|y'all|folks)\b[,\s]*",
    r"^(hi|hey|hello|yo|sup)\b[,\s]*",
    r"\bhope\s+(everyone('s|s|is)\s+\w+\s*)+",
    r"\bjust\s+wanted\s+to\s+(see|ask|check|know)\s+(if\s+)?",
    r"\bthank\s+you(\s+in\s+advance)?\b",
    r"\bcan\s+(maybe|possibly|you)\s+",
    r"\band\s+can\s+(maybe|possibly)?\s*",
]

# Declarative patterns — statements, not questions, never trigger replies
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

AUX_INVERSION = re.compile(
    r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\s",
    re.IGNORECASE
)
WH_QUESTION = re.compile(r"^(what|when|where|who|why|how|which|whose)\b", re.IGNORECASE)
QUESTION_MARK = re.compile(r"\?")

# Event keywords — message must contain at least one of these
EVENT_KEYWORDS = re.compile(
    r"\b(event|events|admin\s*abuse|adminabuse|update|updates|live\s+event|schedule)\b",
    re.IGNORECASE
)

# Event question signals — indicates the message is asking about timing/existence
EVENT_QUESTION_SIGNALS = re.compile(
    r"\b(when|next|upcoming|soon|today|tomorrow|this\s+week|this\s+weekend|"
    r"any\s+events?|any\s+upcoming|happening|scheduled?|planned|going\s+on|"
    r"is\s+there|are\s+there|is\s+it|will\s+there)\b",
    re.IGNORECASE
)

# Lost items — any of these phrases trigger the support reply
LOST_ITEMS_PATTERNS = re.compile(
    r"\b(lost|missing|gone|disappeared|vanished|wiped|reset|empty|can'?t\s+find)\b.{0,30}"
    r"\b(item|items|inventory|stuff|gear|weapons?|tools?|everything)\b"
    r"|"
    r"\b(item|items|inventory|stuff|gear|weapons?|tools?|everything)\b.{0,30}"
    r"\b(lost|missing|gone|disappeared|vanished|wiped|reset|empty)\b",
    re.IGNORECASE
)

SOFT_EVENT_SIGNALS = [
    "any upcoming", "any events", "any active",
    "need to know", "tell me when",
]


# --- HELPERS ---

def strip_filler(text: str) -> str:
    cleaned = text.strip()
    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 6 else text

def is_declarative(text: str) -> bool:
    t = text.strip().lower()
    return any(re.match(p, t) for p in DECLARATIVE_PATTERNS)

def is_question(text: str) -> bool:
    t = text.strip()
    t_lower = t.lower()

    if is_declarative(t_lower):
        return False
    if QUESTION_MARK.search(t):
        return True
    if AUX_INVERSION.match(t):
        return True
    if WH_QUESTION.match(t):
        return True

    core = strip_filler(t_lower)
    if any(sig in core for sig in SOFT_EVENT_SIGNALS):
        return True

    return False

def is_event_query(text: str) -> bool:
    """True if the message is a question asking about an event or admin abuse."""
    if not EVENT_KEYWORDS.search(text):
        return False
    if not is_question(text):
        return False
    return bool(EVENT_QUESTION_SIGNALS.search(text))

def is_lost_items_report(text: str) -> bool:
    """True if the message describes losing items/inventory."""
    return bool(LOST_ITEMS_PATTERNS.search(text))


# --- BOT ---

@client.event
async def on_ready():
    print(f"Logged in as {client.user}. Ready.")

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

    if is_lost_items_report(content_lower):
        await message.reply(SUPPORT_REPLY)
        return

    if is_event_query(content_lower):
        await message.reply(EVENT_REPLY)
        return

client.run(TOKEN)
