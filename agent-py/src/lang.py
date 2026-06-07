"""Language helpers for ANSIO — offline-importable, no credentials needed.

Provides four capabilities used by agent.py at session startup:

1. normalize_language()   — canonicalize any user-supplied language string
2. language_directive()   — system-instruction suffix that pins the reply language
3. greeting_suffix()      — appended to the opening-greeting instruction
4. tts_language_boost()   — maps our canonical lang code to MiniMax TTS
                            ``language_boost`` parameter values.

All functions are pure helpers: no I/O, no async, no livekit/Moss imports.
They can be freely imported in unit tests or scripts without credentials.

language_boost legal values sourced from:
  .venv/lib/python3.11/site-packages/livekit/plugins/minimax/tts.py lines 94-98
  TTSLanguageBoost = Literal["auto", "Chinese", "Chinese,Yue", "English", ...]
"""

from __future__ import annotations

import logging

logger = logging.getLogger("lang")

# ----------------------------------------------------------------------------
# Canonical language codes accepted everywhere in this module.
# ----------------------------------------------------------------------------
LANGS = ("en", "zh", "auto")

# ----------------------------------------------------------------------------
# Alias tables for normalize_language
# ----------------------------------------------------------------------------
_EN_ALIASES = {"en", "english", "英文", "英语"}
_ZH_ALIASES = {"zh", "chinese", "中文", "zh-cn", "zh_cn", "普通话", "汉语"}


# ===========================================================================
# Public helpers
# ===========================================================================


def normalize_language(value: object) -> str:
    """Return a canonical language code: "en", "zh", or "auto".

    Case-insensitive. Unknown / empty / None values fall back to "auto".

    >>> normalize_language("English")
    'en'
    >>> normalize_language("中文")
    'zh'
    >>> normalize_language(None)
    'auto'
    """
    if not value:
        return "auto"
    lowered = str(value).strip().lower()
    if lowered in _EN_ALIASES:
        return "en"
    if lowered in _ZH_ALIASES:
        return "zh"
    logger.debug("normalize_language: unrecognized value %r -> auto", value)
    return "auto"


def language_directive(lang: str) -> str:
    """Return the system-instruction block that enforces the reply language.

    The returned string starts with ``\\n\\n`` so it can be appended directly
    to an existing system prompt without extra whitespace handling.

    Args:
        lang: One of "en", "zh", "auto" (anything else is treated as "auto").
    """
    if lang == "en":
        return (
            "\n\n# Language rule\n"
            "Speak ONLY in English. Never use Chinese words. "
            "Every reply must be 100% English."
        )
    if lang == "zh":
        return (
            "\n\n# Language rule\n"
            "只用中文回答。绝对不要在回复中夹杂英文句子（产品名/人名等专有名词除外）。"  # noqa: RUF001
            "每条回复必须是纯中文。"
        )
    # "auto" or anything unrecognized
    return (
        "\n\n# Language rule\n"
        "Reply in the SAME language the user last spoke. "
        "NEVER mix English and Chinese in a single reply — "
        "pick one language per reply and stay in it (proper nouns excepted)."
    )


def greeting_suffix(lang: str) -> str:
    """Return a short phrase appended to the greeting-generation instruction.

    Tells the LLM which language to use for the opening sentence.

    Args:
        lang: One of "en", "zh", "auto".
    """
    if lang == "en":
        return " Speak the greeting entirely in English."
    if lang == "zh":
        return " 开场白整句用中文说。"
    return " Default the greeting to English unless told otherwise."


def tts_language_boost(lang: str) -> str:
    """Return the MiniMax TTS ``language_boost`` parameter value.

    Evidence: .venv/lib/python3.11/site-packages/livekit/plugins/minimax/tts.py
    line 94 — TTSLanguageBoost = Literal["auto", "Chinese", "Chinese,Yue",
    "English", "Arabic", "Russian", "Spanish", "French", "Portuguese",
    "German", ...]

    Mapping:
      "en"   -> "English"  (exact Literal member, tts.py line 98)
      "zh"   -> "Chinese"  (exact Literal member, tts.py line 96)
      "auto" -> "auto"     (exact Literal member, tts.py line 95)

    Args:
        lang: One of "en", "zh", "auto".
    """
    _map = {
        "en": "English",
        "zh": "Chinese",
        "auto": "auto",
    }
    return _map.get(lang, "auto")


def stt_language(lang: str) -> str:
    """Return the Deepgram nova-3 ``language`` argument for our canonical code.

    Deepgram nova-3 "multi" (the Multilingual model) does NOT include Chinese —
    it covers English/Spanish/French/German/Hindi/Italian/Japanese/Dutch/
    Russian/Portuguese only. Mandarin needs the dedicated language model
    (``language="zh"``). So we switch STT per the user's language toggle:

      "zh"   -> "zh"     (nova-3 Mandarin Simplified, dedicated lang model)
      "en"   -> "en"     (nova-3 English)
      "auto" -> "multi"  (Deepgram multilingual; Chinese excluded by design)

    Note: nova-3 cannot do zh+en code-switching within one turn, so Chinese is
    a per-session choice (select 中文 to be understood in Mandarin).

    Args:
        lang: One of "en", "zh", "auto".
    """
    _map = {
        "zh": "zh",
        "en": "en",
        "auto": "multi",
    }
    return _map.get(lang, "multi")
