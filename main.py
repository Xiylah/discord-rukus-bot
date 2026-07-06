import discord
import os
import json
import re
import random
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

TOKEN = os.getenv("TOKEN")
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

EVENT_CHANNEL_ID = 1458936961044709539
FURNITURE_CHANNEL_ID = 1510456653085020290
SUPPORT_CHANNEL_ID = 1478734423972384799

EVENT_EMBEDS = [
    ("📅 Check the Events Channel", f"Anything related to events or updates gets posted in <#{EVENT_CHANNEL_ID}>."),
    ("👀 Events are posted here!", f"Head over to <#{EVENT_CHANNEL_ID}> for the latest event info."),
    ("🗓️ Event info this way →", f"Keep an eye on <#{EVENT_CHANNEL_ID}> for upcoming events and schedules."),
    ("🔔 Stay in the loop", f"All event announcements and updates are in <#{EVENT_CHANNEL_ID}>!"),
]

SUPPORT_EMBEDS = [
    ("🎫 Need help?", f"Open a support ticket in <#{SUPPORT_CHANNEL_ID}> and the team will sort you out!"),
    ("🛠️ Lost something?", f"File a ticket in <#{SUPPORT_CHANNEL_ID}> and we'll look into it for you."),
    ("📬 We've got you", f"Head to <#{SUPPORT_CHANNEL_ID}> and open a ticket — we'll get it resolved."),
    ("🆘 Let's get this fixed", f"Submit a support ticket in <#{SUPPORT_CHANNEL_ID}> and we'll help you out!"),
]

def make_event_embed() -> discord.Embed:
    title, desc = random.choice(EVENT_EMBEDS)
    embed = discord.Embed(title=title, description=desc, color=0x5865F2)
    embed.set_footer(text="Automated response")
    return embed

def make_support_embed() -> discord.Embed:
    title, desc = random.choice(SUPPORT_EMBEDS)
    embed = discord.Embed(title=title, description=desc, color=0xED4245)
    embed.set_footer(text="Automated response")
    return embed

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

LOST_ITEMS_EXAMPLES = [
    "i lost my items",
    "i lost my inventory",
    "my items are gone",
    "my inventory disappeared",
    "i lost everything in my inventory",
    "all my items are missing",
    "my stuff disappeared",
    "i lost all my stuff",
    "my items got wiped",
    "my inventory got reset",
    "i lost my gear",
    "all my gear is gone",
    "my items vanished",
    "i can't find my items",
    "where did my items go",
    "my inventory is empty",
    "i lost my weapons",
    "my weapons disappeared",
    "i lost my tools",
    "everything i had is gone",
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
    "no live event at this time",
    "they have mini events all the time",
    "there are no events right now",
    "the event already ended",
    "we just had an event",
    "events will be posted when ready",
    "admin abuse will be announced",
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
    "any upcoming", "any active", "any events",
    "give me the", "tell me the", "need to know",
]

AUTO_LEARN_WINDOW = 0.12
EVENT_THRESHOLD = 0.72
LOST_ITEMS_THRESHOLD = 0.74
NEGATIVE_PENALTY = 0.02


# --- DRUG / SUBSTANCE FILTER ---

DRUG_TERMS = [
    "weed", "marijuana", "cannabis", "blunt", "bong", "dank",
    "kush", "reefer", "ganja", "420", "thc", "cbd", "edibles",
    "xan", "xanax", "xannies", "percs", "percocet", "oxycontin",
    "vicodin", "adderall", "addy", "molly", "mdma", "ecstasy",
    "cocaine", "meth", "heroin", "fentanyl", "fent", "lsd",
    "shrooms", "ketamine", "vape", "vaping", "juul",
    "dab", "dabs",
]

# Match whole words only, so "potato" or "escape" won't trigger
DRUG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in DRUG_TERMS) + r")\b",
    re.IGNORECASE,
)

DRUG_WARNINGS = [
    "Please keep all conversations appropriate for all ages. 🙏",
    "Let's keep the chat family-friendly for everyone here!",
    "Reminder: please keep all discussion appropriate for all ages.",
]


# --- PERSISTENCE ---

def load_learned_examples():
    if not LEARNED_EXAMPLES_FILE.exists():
        return []
    try:
        data = json.loads(LEARNED_EXAMPLES_FILE.read_text())
        return data.get("event", [])
    except Exception as e:
        print(f"[warn] Could not load learned_examples.json: {e}")
        return []

def save_learned_example(text: str):
    data = {"event": []}
    if LEARNED_EXAMPLES_FILE.exists():
        try:
            data = json.loads(LEARNED_EXAMPLES_FILE.read_text())
        except Exception:
            pass
    if text not in data["event"]:
        data["event"].append(text)
        LEARNED_EXAMPLES_FILE.write_text(json.dumps(data, indent=2))
        print(f"[learn] Saved event example: {text!r}")


# --- MODEL + EMBEDDINGS ---

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

extra_event = load_learned_examples()
all_event_examples = EVENT_EXAMPLES + extra_event

event_embeddings = model.encode(all_event_examples, convert_to_tensor=True)
negative_embeddings = model.encode(NEGATIVE_EXAMPLES, convert_to_tensor=True)
lost_items_embeddings = model.encode(LOST_ITEMS_EXAMPLES, convert_to_tensor=True)

print(f"Ready. {len(all_event_examples)} event examples.")


def add_live_embedding(text: str):
    global event_embeddings
    new_emb = model.encode(text, convert_to_tensor=True).unsqueeze(0)
    event_embeddings = torch.cat([event_embeddings, new_emb], dim=0)
    all_event_examples.append(text)


# --- CLASSIFICATION ---

def strip_filler(text: str) -> str:
    """Remove common filler phrases to expose the core intent."""
    cleaned = text.strip()
    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 6 else text

def is_declarative(text: str) -> bool:
    """Return True if the text is almost certainly a statement, not a question."""
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

    return top_mean(event_embeddings), top_mean(negative_embeddings), top_mean(lost_items_embeddings)


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

    # Drug/substance filter — delete and warn (runs on all channels except furniture)
    if DRUG_PATTERN.search(content):
        try:
            await message.delete()
        except discord.errors.NotFound:
            pass
        await message.channel.send(
            f"{message.author.mention} {random.choice(DRUG_WARNINGS)}",
            delete_after=10,
        )
        return

    if not content or len(content) < 8:
        return

    content_lower = content.lower()

    # Fast path: explicit admin abuse scheduling query
    if is_admin_abuse_query(content_lower):
        await message.reply(embed=make_event_embed())
        return

    # Score first — lost items are statements and would be blocked by the question gate
    event_score, negative_score, lost_items_score = get_scores(content_lower)

    # Lost items check runs regardless of whether the message is a question
    if lost_items_score > LOST_ITEMS_THRESHOLD:
        await message.reply(embed=make_support_embed())
        return

    # Gate: event replies should only trigger on genuine questions
    if not is_question(content):
        return

    penalty = max(0, (negative_score - 0.55) * 2) * NEGATIVE_PENALTY
    adj_event = event_score - penalty

    if adj_event > EVENT_THRESHOLD:
        await message.reply(embed=make_event_embed())
        return

    # Don't auto-learn or reply if a negative example is dominant
    if negative_score > event_score:
        return

    # Near-miss: auto-learn and reply
    if EVENT_THRESHOLD - AUTO_LEARN_WINDOW < adj_event < EVENT_THRESHOLD:
        add_live_embedding(content_lower)
        save_learned_example(content_lower)
        await message.reply(embed=make_event_embed())


client.run(TOKEN)
