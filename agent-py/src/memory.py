"""Per-user agentic memory for ANSIO (consented profile recall).

This module gives ANSIO a *long-term memory* of returning users: a single Moss
document per user holding a short, structured profile (product, audience,
platform, budget, goals, style). On the next call the agent loads that profile
and greets the user as someone it already knows — no re-interrogation.

Privacy is the hard rule. When memory is *disabled* (user did not consent),
every method here is a strict no-op that performs ZERO Moss calls. The unit
tests in ``tests/test_memory.py`` prove this.

Index strategy (matches the free-tier 3-index plan in agent.py):
  - Primary: a dedicated 4th index ``ansio_memory`` (env ANSIO_MEMORY_INDEX).
  - Fallback: if that index can't be loaded (free-tier cap), the profile rides
    in ``ansio_content`` (env ANSIO_CONTENT_INDEX) and is isolated by a
    ``doc_type="user_profile"`` + ``user_id`` double filter.

Doc convention: stable id ``profile-{user_id}`` (so add_docs upserts), text is
the profile prose, metadata = {"user_id", "doc_type": "user_profile"}. An empty
profile is written as the EMPTY_PROFILE sentinel (a degraded "clear" when
delete_docs is unavailable); on load the sentinel reads back as "no profile".

Pure helpers (history_to_items / profile_instructions / memory_greeting /
extract_profile_text) import without Moss or livekit credentials so they stay
unit-testable offline.
"""

from __future__ import annotations

import contextlib
import logging
import os

from moss import DocumentInfo, MossClient, QueryOptions

logger = logging.getLogger("memory")

# ----------------------------------------------------------------------------
# Index + sentinel constants (env-overridable; defaults match agent.py).
# ----------------------------------------------------------------------------
PROFILE_INDEX = os.getenv("ANSIO_MEMORY_INDEX", "ansio_memory")
FALLBACK_INDEX = os.getenv("ANSIO_CONTENT_INDEX", "ansio_content")
EMPTY_PROFILE = "__EMPTY_PROFILE__"

# Internal-injection markers stripped from history before profile extraction so
# the LLM never re-summarizes our own retrieval/memory notes back into a profile.
_INTERNAL_MARKERS = ("[ANSIO retrieval", "[ANSIO memory")

# Transcript shaping for one-shot extraction (keep the prompt cheap + bounded).
_MAX_ITEM_CHARS = 300
_MAX_TRANSCRIPT_CHARS = 6000

# Default first-turn greeting instruction (kept identical to agent.py's so a
# cold/no-profile session greets exactly as before).
_DEFAULT_GREETING = (
    "Greet the user warmly in one sentence as ANSIO, the growth "
    "engineer, and ask what product they want to find creators for."
)
_RETURNING_GREETING = (
    "Greet the user warmly in one sentence as ANSIO. You REMEMBER her from "
    "before — naturally mention her product and what you discussed, and offer "
    "to pick up where you left off. Do not ask for information you already know."
)

# One-shot extraction prompt. Asks for a compact, spoken-friendly profile or the
# literal NONE when the conversation carries nothing worth remembering.
EXTRACTION_PROMPT = (
    "You distill a brand operator's conversation into a compact memory profile "
    "for next time. Read the transcript and write at most 200 words covering, "
    "only when present: product / category, target audience, platform "
    "preference, budget, goals, communication-style preference, and any other "
    "key points. Use short labelled lines, no markdown. If the transcript has "
    "too little to remember, reply with exactly NONE.\n\nTranscript:\n"
)


# ===========================================================================
# UserMemory — the per-user, consent-gated profile store
# ===========================================================================


class UserMemory:
    """Loads/saves one consented profile document for a single user.

    All Moss access is funneled through this object and wrapped in try/except so
    a memory failure can never break the live call. When ``enabled`` is False,
    every coroutine returns immediately without touching Moss.
    """

    def __init__(
        self, user_id: str, enabled: bool, moss: MossClient | None = None
    ) -> None:
        self._user_id = user_id
        self._enabled = bool(enabled)
        self._moss = moss or MossClient(
            os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")
        )
        # Active index; flips to FALLBACK_INDEX if the primary won't load.
        self._index = PROFILE_INDEX

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def _doc_id(self) -> str:
        return f"profile-{self._user_id}"

    def _filter(self):
        """User-scoped filter; double-filter (doc_type + user_id) on fallback."""
        user_cond = {"field": "user_id", "condition": {"$eq": self._user_id}}
        if self._index == FALLBACK_INDEX:
            return {
                "$and": [
                    user_cond,
                    {"field": "doc_type", "condition": {"$eq": "user_profile"}},
                ]
            }
        return user_cond

    # -- lifecycle ---------------------------------------------------------
    async def startup(self, refresh: bool = False) -> str | None:
        """Ensure the index is loaded, optionally clear, then load the profile.

        Privacy rule: disabled -> return None with ZERO Moss calls. On a primary
        load failure we retry the fallback index; if that also fails we disable
        ourselves and return None so the rest of the call proceeds memory-free.
        """
        if not self._enabled:
            return None
        try:
            await self._moss.load_index(self._index)
        except Exception:
            logger.warning("load_index(%s) failed; trying fallback", self._index)
            self._index = FALLBACK_INDEX
            try:
                await self._moss.load_index(self._index)
            except Exception:
                logger.exception("memory index load failed; disabling memory")
                self._enabled = False
                return None
        if refresh:
            await self.clear_profile()
        return await self.load_profile()

    async def load_profile(self) -> str | None:
        """Return the stored profile text, or None (disabled / empty / sentinel)."""
        if not self._enabled:
            return None
        try:
            result = await self._moss.query(
                self._index,
                "user profile",
                QueryOptions(top_k=1, filter=self._filter()),
            )
            docs = getattr(result, "docs", None) or []
            if not docs:
                return None
            text = (getattr(docs[0], "text", "") or "").strip()
            if not text or text == EMPTY_PROFILE:
                return None
            return text
        except Exception:
            logger.exception("load_profile failed")
            return None

    async def save_profile(self, profile_text: str) -> bool:
        """Upsert the profile doc under the stable id, then reload the index."""
        if not self._enabled:
            return False
        text = (profile_text or "").strip()
        if not text:
            return False
        try:
            doc = DocumentInfo(
                id=self._doc_id,
                text=text,
                metadata={"user_id": self._user_id, "doc_type": "user_profile"},
            )
            await self._moss.add_docs(self._index, [doc])
            # Writes are only visible after a reload (verified in template).
            await self._moss.load_index(self._index)
            return True
        except Exception:
            logger.exception("save_profile failed")
            return False

    async def clear_profile(self) -> bool:
        """Delete the profile doc; degrade to an EMPTY_PROFILE write if delete fails."""
        if not self._enabled:
            return False
        try:
            await self._moss.delete_docs(self._index, [self._doc_id])
            await self._moss.load_index(self._index)
            return True
        except Exception:
            logger.warning("delete_docs failed; degrading clear to sentinel write")
            # _write_sentinel mirrors save_profile but bypasses its empty guard.
            return await self._write_sentinel()

    async def _write_sentinel(self) -> bool:
        """Internal: persist the EMPTY_PROFILE marker (the clear-degrade path)."""
        try:
            doc = DocumentInfo(
                id=self._doc_id,
                text=EMPTY_PROFILE,
                metadata={"user_id": self._user_id, "doc_type": "user_profile"},
            )
            await self._moss.add_docs(self._index, [doc])
            await self._moss.load_index(self._index)
            return True
        except Exception:
            logger.exception("sentinel write failed")
            return False

    async def extract_and_save(self, history_items: list[tuple[str, str]], llm) -> bool:
        """Distill the conversation into a profile via the LLM and persist it."""
        if not self._enabled or not history_items:
            return False
        profile = await extract_profile_text(history_items, llm)
        if not profile:
            return False
        return await self.save_profile(profile)


# ===========================================================================
# Pure helpers (offline-importable; no Moss/livekit credentials needed)
# ===========================================================================


def history_to_items(chat_ctx) -> list[tuple[str, str]]:
    """Flatten a livekit ChatContext into [(role, text)] for extraction.

    Skips system/developer turns, our own internal injection notes, and empty
    text so the profile is built only from real user/assistant dialogue.
    """
    items = getattr(chat_ctx, "items", None) or []
    out: list[tuple[str, str]] = []
    for item in items:
        role = getattr(item, "role", None)
        if role in ("system", "developer"):
            continue
        text = (getattr(item, "text_content", None) or "").strip()
        if not text:
            continue
        if any(marker in text for marker in _INTERNAL_MARKERS):
            continue
        out.append((role or "user", text))
    return out


def profile_instructions(profile: str | None) -> str:
    """Build the system-instruction suffix that primes the agent on a known user."""
    if not profile:
        return ""
    return (
        "\n\n# Known user profile (from memory, user consented)\n"
        "你已经认识这位用户。画像：" + profile + "\n"  # noqa: RUF001
        "开场和建议要自然体现你记得她——直接引用她的产品/平台/预算，"  # noqa: RUF001
        "不要重新盘问已知信息。"
    )


def memory_greeting(profile: str | None) -> str:
    """Pick the opening instruction: cold default vs. warm returning-user."""
    return _RETURNING_GREETING if profile else _DEFAULT_GREETING


async def extract_profile_text(history_items, llm) -> str | None:
    """One-shot LLM call turning a transcript into a <=200-word profile or None.

    Uses the livekit LLM interface: ``llm.chat(chat_ctx=ChatContext)`` returns
    an async stream of ChatChunk; we concatenate ``chunk.delta.content``. A bare
    NONE (case-insensitive) means "nothing to remember" -> None. Any failure is
    swallowed and returns None so extraction never breaks a session.
    """
    if not history_items:
        return None
    lines: list[str] = []
    total = 0
    for role, text in history_items:
        snippet = (text or "")[:_MAX_ITEM_CHARS]
        line = f"{role}: {snippet}"
        if total + len(line) > _MAX_TRANSCRIPT_CHARS:
            break
        lines.append(line)
        total += len(line)
    transcript = "\n".join(lines)
    if not transcript:
        return None
    try:
        from livekit.agents import ChatContext

        chat_ctx = ChatContext.empty()
        chat_ctx.add_message(role="user", content=EXTRACTION_PROMPT + transcript)
        parts: list[str] = []
        stream = llm.chat(chat_ctx=chat_ctx)
        try:
            async for chunk in stream:
                delta = getattr(chunk, "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    parts.append(content)
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
        profile = "".join(parts).strip()
        if not profile or profile.upper() == "NONE":
            return None
        return profile
    except Exception:
        logger.exception("extract_profile_text failed")
        return None
