import discord
from discord import app_commands
from discord.ext import commands
import os
import json
import re
import random
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
from deep_translator import GoogleTranslator
from deep_translator.exceptions import (
    NotValidPayload,
    TranslationNotFound,
    RequestError,
    TooManyRequests,
)

TOKEN = os.getenv("TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

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


# --- TRANSLATION CORE ---

# Don't translate messages shorter than this (emotes, "gg", "lol", etc.)
TRANSLATE_MIN_LEN = 12

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"<[@#!&:][^>]+>")      # mentions + custom emoji
_EMOJI_UNICODE_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

def _strippable(text: str) -> str:
    """Remove URLs, mentions, and emoji to see if any real language remains."""
    t = _URL_RE.sub(" ", text)
    t = _MENTION_RE.sub(" ", t)
    t = _EMOJI_UNICODE_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


# Common chat slang that shouldn't trigger translation on its own.
SLANG_TERMS = {
    "ty", "tysm", "thx", "thanks", "np", "yw", "wyd", "wym", "hru",
    "wbu", "wb", "gg", "glhf", "brb", "afk", "gtg", "g2g", "ttyl",
    "lol", "lmao", "lmfao", "rofl", "lmaooo", "omg", "omfg", "wtf",
    "idk", "idc", "ikr", "imo", "imho", "tbh", "ngl", "fr", "frfr",
    "smh", "istg", "irl", "dm", "pm", "afaik", "asap", "aka", "btw",
    "nvm", "rn", "atm", "ez", "op", "pog", "poggers", "w", "l",
    "ratio", "based", "cap", "nocap", "bet", "fam", "bruh", "bro",
    "yo", "sup", "wsg", "wassup", "ong", "icl", "lowkey", "highkey",
    "sus", "goat", "mid", "gyat", "rizz", "fyp", "sheesh", "yeet",
    "oof", "yikes", "welp", "meh", "eh", "hmm", "ok", "okay", "kk",
    "k", "yeah", "yea", "yep", "nah", "nope", "ye", "ya", "u", "ur",
    "pls", "plz", "plss", "ffs", "wdym",
}

# Keep only letters/digits/spaces so "ty!" or "wyd??" still match
_SLANG_CLEAN_RE = re.compile(r"[^\w\s]")

def strip_slang(text: str) -> str:
    """Remove standalone slang tokens; return whatever real text is left."""
    cleaned = _SLANG_CLEAN_RE.sub(" ", text.lower())
    kept = [w for w in cleaned.split() if w not in SLANG_TERMS]
    return " ".join(kept).strip()

def translate_text(text: str, target: str = "en"):
    """
    Translate text into `target` language code.
    Returns (translated_text, detected_source) or (None, reason_string).
    """
    core = _strippable(text)
    if len(core) < TRANSLATE_MIN_LEN:
        return None, "too_short"

    # If removing standalone slang leaves nothing meaningful, don't translate
    if len(strip_slang(core)) < TRANSLATE_MIN_LEN:
        return None, "slang_only"

    try:
        translator = GoogleTranslator(source="auto", target=target)
        translated = translator.translate(core)
    except (NotValidPayload, TranslationNotFound):
        return None, "no_translation"
    except (RequestError, TooManyRequests) as e:
        print(f"[translate] backend error: {e}")
        return None, "backend_error"
    except Exception as e:
        print(f"[translate] unexpected error: {e}")
        return None, "error"

    if not translated:
        return None, "empty"

    src = getattr(translator, "source", "auto")

    # If result equals input AND we were targeting English, it was already English
    if target == "en" and translated.strip().lower() == core.strip().lower():
        return None, "already_target"

    return translated, src

def make_translation_embed(translated: str, src: str, target: str, requester: str = None) -> discord.Embed:
    src_label = src.upper() if src and src != "auto" else "Auto-detected"
    embed = discord.Embed(
        title="🌐 Translation",
        description=translated[:4000],
        color=0x57F287,
    )
    footer = f"{src_label} → {target.upper()}"
    if requester:
        footer += f" • requested by {requester}"
    embed.set_footer(text=footer)
    return embed


# --- FLAG EMOJI -> LANGUAGE ---

def flag_to_country_code(emoji: str):
    """Convert a flag emoji (two regional-indicator chars) to its ISO country code."""
    if len(emoji) != 2:
        return None
    cps = [ord(c) for c in emoji]
    if not all(0x1F1E6 <= cp <= 0x1F1FF for cp in cps):
        return None
    return "".join(chr(cp - 0x1F1E6 + ord("A")) for cp in cps)

# Country code -> deep_translator language code
COUNTRY_TO_LANG = {
    "US": "en", "GB": "en", "AU": "en", "CA": "en", "IE": "en", "NZ": "en",
    "FR": "fr", "ES": "es", "MX": "es", "AR": "es", "CO": "es", "CL": "es",
    "BR": "pt", "PT": "pt", "DE": "de", "AT": "de", "IT": "it",
    "JP": "ja", "KR": "ko", "CN": "zh-CN", "TW": "zh-TW", "HK": "zh-TW",
    "RU": "ru", "SA": "ar", "AE": "ar", "EG": "ar", "IN": "hi",
    "NL": "nl", "BE": "nl", "PL": "pl", "TR": "tr", "VN": "vi",
    "TH": "th", "ID": "id", "PH": "tl", "SE": "sv", "NO": "no",
    "DK": "da", "FI": "fi", "GR": "el", "UA": "uk", "RO": "ro",
    "HU": "hu", "CZ": "cs", "SK": "sk", "IL": "iw", "IR": "fa",
    "PK": "ur", "BD": "bn", "MY": "ms", "BG": "bg", "HR": "hr",
}

# Common language names for the dropdown / slash-command choices
LANGUAGE_CHOICES = {
    "English": "en", "Spanish": "es", "French": "fr", "Portuguese": "pt",
    "German": "de", "Italian": "it", "Dutch": "nl", "Russian": "ru",
    "Japanese": "ja", "Korean": "ko", "Chinese (Simplified)": "zh-CN",
    "Arabic": "ar", "Hindi": "hi", "Turkish": "tr", "Polish": "pl",
    "Vietnamese": "vi", "Thai": "th", "Indonesian": "id", "Filipino": "tl",
    "Ukrainian": "uk", "Greek": "el", "Swedish": "sv",
}

# Broader code -> readable name map for language detection results
# (detection can return languages not offered in the translate menu)
LANG_CODE_TO_NAME = {
    "en": "English", "es": "Spanish", "fr": "French", "pt": "Portuguese",
    "de": "German", "it": "Italian", "nl": "Dutch", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)", "ar": "Arabic", "hi": "Hindi",
    "tr": "Turkish", "pl": "Polish", "vi": "Vietnamese", "th": "Thai",
    "id": "Indonesian", "tl": "Filipino", "uk": "Ukrainian", "el": "Greek",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish",
    "ro": "Romanian", "hu": "Hungarian", "cs": "Czech", "sk": "Slovak",
    "iw": "Hebrew", "he": "Hebrew", "fa": "Persian", "ur": "Urdu",
    "bn": "Bengali", "ms": "Malay", "bg": "Bulgarian", "hr": "Croatian",
    "sr": "Serbian", "sl": "Slovenian", "lt": "Lithuanian", "lv": "Latvian",
    "et": "Estonian", "ca": "Catalan", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "kn": "Kannada", "gu": "Gujarati", "mr": "Marathi",
    "pa": "Punjabi", "af": "Afrikaans", "sw": "Swahili", "tl-ph": "Filipino",
}

def detect_language(text: str):
    """
    Detect the language of `text` using the free Google endpoint.
    Returns (code, readable_name) or (None, None).
    Works by asking the translator to auto-detect while translating to English.
    """
    core = _strippable(text)
    if len(core) < TRANSLATE_MIN_LEN:
        return None, None
    try:
        translator = GoogleTranslator(source="auto", target="en")
        translator.translate(core)  # populates detected source
        code = getattr(translator, "source", None)
    except (RequestError, TooManyRequests) as e:
        print(f"[detect] backend error: {e}")
        return None, None
    except Exception as e:
        print(f"[detect] unexpected error: {e}")
        return None, None

    if not code or code == "auto":
        return None, None
    name = LANG_CODE_TO_NAME.get(code, code.upper())
    return code, name


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


# --- MESSAGE HANDLER ---

@bot.event
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

    # Drug/substance filter — delete and warn
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

    # Allow prefix command processing (future-proof; harmless alongside slash cmds)
    await bot.process_commands(message)


# --- FLAG REACTION -> TRANSLATE ---

@bot.event
async def on_raw_reaction_add(payload):
    # Ignore the bot's own reactions
    if bot.user and payload.user_id == bot.user.id:
        return

    emoji = str(payload.emoji)
    country = flag_to_country_code(emoji)
    if not country:
        return  # not a flag emoji

    target_lang = COUNTRY_TO_LANG.get(country)
    if not target_lang:
        return  # flag we don't have a language mapping for

    channel = bot.get_channel(payload.channel_id)
    if channel is None:
        try:
            channel = await bot.fetch_channel(payload.channel_id)
        except (discord.NotFound, discord.Forbidden):
            return

    try:
        message = await channel.fetch_message(payload.message_id)
    except (discord.NotFound, discord.Forbidden):
        return

    if not message.content or not message.content.strip():
        return

    translated, src = translate_text(message.content, target=target_lang)
    if not translated:
        return  # too short, already that language, or backend hiccup — stay silent

    reactor = payload.member.mention if payload.member else "someone"
    await message.reply(
        content=f"{reactor} requested a translation:",
        embed=make_translation_embed(translated, src, target_lang),
        mention_author=False,
    )


# --- SLASH COMMAND: /translate ---

@bot.tree.command(name="translate", description="Translate text into a chosen language")
@app_commands.describe(text="The text to translate", to="Target language")
@app_commands.choices(
    to=[app_commands.Choice(name=name, value=code) for name, code in LANGUAGE_CHOICES.items()]
)
async def translate_command(
    interaction: discord.Interaction,
    text: str,
    to: app_commands.Choice[str] = None,
):
    target = to.value if to else "en"
    translated, src = translate_text(text, target=target)
    if not translated:
        await interaction.response.send_message(
            "Couldn't translate that — it may be too short, already in that language, "
            "or the translation service is busy. Try again in a moment.",
            ephemeral=True,
        )
        return
    await interaction.response.send_message(
        embed=make_translation_embed(translated, src, target, requester=interaction.user.display_name)
    )


# --- CONTEXT MENUS: right-click a message -> Apps -> Translate to <language> ---

async def _context_translate(interaction: discord.Interaction, message: discord.Message, target: str):
    """Shared handler: translate the right-clicked message into `target`."""
    if not message.content or not message.content.strip():
        await interaction.response.send_message(
            "That message has no text to translate.", ephemeral=True
        )
        return
    translated, src = translate_text(message.content, target=target)
    if not translated:
        await interaction.response.send_message(
            "Couldn't translate that — it may be too short, slang, already in that "
            "language, or the service is busy.",
            ephemeral=True,
        )
        return
    await interaction.response.send_message(
        embed=make_translation_embed(translated, src, target, requester=interaction.user.display_name)
    )


@bot.tree.context_menu(name="Translate to English")
async def translate_to_english(interaction: discord.Interaction, message: discord.Message):
    await _context_translate(interaction, message, "en")


@bot.tree.context_menu(name="Translate to French")
async def translate_to_french(interaction: discord.Interaction, message: discord.Message):
    await _context_translate(interaction, message, "fr")


@bot.tree.context_menu(name="Translate to Spanish")
async def translate_to_spanish(interaction: discord.Interaction, message: discord.Message):
    await _context_translate(interaction, message, "es")


@bot.tree.context_menu(name="Detect Language")
async def detect_language_menu(interaction: discord.Interaction, message: discord.Message):
    if not message.content or not message.content.strip():
        await interaction.response.send_message(
            "That message has no text to analyze.", ephemeral=True
        )
        return
    code, name = detect_language(message.content)
    if not code:
        await interaction.response.send_message(
            "Couldn't detect the language — the message may be too short, slang, "
            "or the service is busy.",
            ephemeral=True,
        )
        return
    embed = discord.Embed(
        title="🔎 Language Detected",
        description=f"This message appears to be **{name}** (`{code}`).",
        color=0x5865F2,
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)



# --- READY / SYNC ---

# Your server ID — guild-scoped sync makes commands appear instantly.
GUILD_ID = 700382994946588814

@bot.event
async def on_ready():
    try:
        guild = discord.Object(id=GUILD_ID)

        # 1. Copy code-defined commands onto the guild scope FIRST, while they're
        #    still present in the in-memory tree.
        bot.tree.copy_global_to(guild=guild)

        # 2. Remove the GLOBAL registrations on Discord's side (stale duplicates
        #    from earlier deploys). Clearing only the global scope does NOT affect
        #    the guild copies we just made above.
        bot.tree.clear_commands(guild=None)
        await bot.tree.sync()  # pushes empty global set -> deletes global dupes

        # 3. Sync the guild scope (instant). These are the commands users see.
        synced = await bot.tree.sync(guild=guild)
        print(f"Synced {len(synced)} command(s) to guild {GUILD_ID}.")
    except Exception as e:
        print(f"[warn] Command sync failed: {e}")
    print(f"Logged in as {bot.user}.")


bot.run(TOKEN)
