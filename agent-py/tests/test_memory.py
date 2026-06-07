"""Offline unit tests for the per-user agentic memory module (memory.py).

These stub ``MossClient`` and the LLM so they run with NO Moss credentials and
NO network. They prove the privacy rule (disabled => zero Moss calls), the
upsert/reload write path, sentinel handling, the clear-degrade fallback, the
refresh-then-load order, history filtering, and one-shot profile extraction.

Live, credentialed behavior is covered by src/build_memory_index.py.
"""

from __future__ import annotations

import pytest

from memory import (
    EMPTY_PROFILE,
    PROFILE_INDEX,
    UserMemory,
    extract_profile_text,
    history_to_items,
    memory_greeting,
    profile_instructions,
)

USER_ID = "user_1"


# ---------------------------------------------------------------------------
# Fakes (mirror the Moss SDK surface UserMemory actually touches)
# ---------------------------------------------------------------------------


class _FakeDoc:
    def __init__(self, text="", score=None, metadata=None) -> None:
        self.text = text
        self.score = score
        self.metadata = metadata or {}


class _FakeResult:
    def __init__(self, docs) -> None:
        self.docs = docs


class _FakeMossClient:
    """Records every call; returns a configurable query result."""

    def __init__(self) -> None:
        self.load_index_calls: list[str] = []
        self.query_calls: list[tuple] = []
        self.add_docs_calls: list[tuple] = []
        self.delete_docs_calls: list[tuple] = []
        self.query_result = _FakeResult([])
        self.delete_raises = False

    async def load_index(self, name, *args, **kwargs):
        self.load_index_calls.append(name)
        return name

    async def query(self, name, query, options=None):
        self.query_calls.append((name, query, options))
        return self.query_result

    async def add_docs(self, name, docs, options=None):
        self.add_docs_calls.append((name, docs))
        return None

    async def delete_docs(self, name, doc_ids):
        self.delete_docs_calls.append((name, doc_ids))
        if self.delete_raises:
            raise RuntimeError("delete unsupported")
        return None

    def total_calls(self) -> int:
        return (
            len(self.load_index_calls)
            + len(self.query_calls)
            + len(self.add_docs_calls)
            + len(self.delete_docs_calls)
        )


class _FakeChunk:
    def __init__(self, content) -> None:
        self.delta = type("Delta", (), {"content": content})()


class _FakeStream:
    """Async-iterable LLM stream yielding ChatChunk-shaped objects."""

    def __init__(self, contents) -> None:
        self._contents = list(contents)

    def __aiter__(self):
        self._it = iter(self._contents)
        return self

    async def __anext__(self):
        try:
            return _FakeChunk(next(self._it))
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        return None


class _FakeLLM:
    def __init__(self, contents) -> None:
        self._contents = contents
        self.chat_calls = 0

    def chat(self, *, chat_ctx, **kwargs):
        self.chat_calls += 1
        return _FakeStream(self._contents)


class _FakeMessage:
    """Minimal ChatMessage stand-in (role + text_content like the real one)."""

    def __init__(self, role, text) -> None:
        self.role = role
        self.text_content = text


class _FakeChatCtx:
    def __init__(self, items) -> None:
        self.items = items


def _mem(enabled: bool, moss: _FakeMossClient | None = None) -> UserMemory:
    return UserMemory(USER_ID, enabled, moss=moss or _FakeMossClient())


# ---------------------------------------------------------------------------
# 1. Privacy rule: disabled => total no-op, ZERO Moss calls (most important)
# ---------------------------------------------------------------------------


async def test_disabled_memory_is_total_noop() -> None:
    fake = _FakeMossClient()
    mem = _mem(False, fake)

    assert mem.enabled is False
    assert await mem.startup() is None
    assert await mem.startup(refresh=True) is None
    assert await mem.load_profile() is None
    assert await mem.save_profile("anything") is False
    assert await mem.clear_profile() is False
    assert await mem.extract_and_save([("user", "hi")], _FakeLLM(["x"])) is False

    # The privacy guarantee: not a single Moss call was made.
    assert fake.total_calls() == 0


# ---------------------------------------------------------------------------
# 2. save_profile: stable id, user_id metadata, reload after write
# ---------------------------------------------------------------------------


async def test_save_profile_uses_stable_id_and_reloads() -> None:
    fake = _FakeMossClient()
    mem = _mem(True, fake)

    ok = await mem.save_profile("Product: ANSIO. Platform: Xiaohongshu.")
    assert ok is True

    assert len(fake.add_docs_calls) == 1
    index, docs = fake.add_docs_calls[0]
    assert index == PROFILE_INDEX
    assert len(docs) == 1
    doc = docs[0]
    assert doc.id == "profile-user_1"
    assert doc.metadata["user_id"] == USER_ID
    assert doc.metadata["doc_type"] == "user_profile"

    # Write must be followed by a reload so it becomes queryable.
    assert PROFILE_INDEX in fake.load_index_calls


async def test_save_profile_rejects_empty_text() -> None:
    fake = _FakeMossClient()
    mem = _mem(True, fake)
    assert await mem.save_profile("   ") is False
    assert fake.add_docs_calls == []


# ---------------------------------------------------------------------------
# 3. load_profile: hit / sentinel / empty
# ---------------------------------------------------------------------------


async def test_load_profile_returns_text_on_hit() -> None:
    fake = _FakeMossClient()
    fake.query_result = _FakeResult([_FakeDoc("Loves budget travel gear.")])
    mem = _mem(True, fake)

    assert await mem.load_profile() == "Loves budget travel gear."
    # Query scoped to this user via the metadata filter.
    _name, _q, options = fake.query_calls[0]
    assert options.filter == {"field": "user_id", "condition": {"$eq": USER_ID}}


async def test_load_profile_treats_sentinel_as_no_profile() -> None:
    fake = _FakeMossClient()
    fake.query_result = _FakeResult([_FakeDoc(EMPTY_PROFILE)])
    mem = _mem(True, fake)
    assert await mem.load_profile() is None


async def test_load_profile_empty_result_is_none() -> None:
    fake = _FakeMossClient()
    fake.query_result = _FakeResult([])
    mem = _mem(True, fake)
    assert await mem.load_profile() is None


# ---------------------------------------------------------------------------
# 4. clear_profile: delete path + degrade-to-sentinel path
# ---------------------------------------------------------------------------


async def test_clear_profile_deletes_correct_id() -> None:
    fake = _FakeMossClient()
    mem = _mem(True, fake)

    assert await mem.clear_profile() is True
    assert len(fake.delete_docs_calls) == 1
    index, ids = fake.delete_docs_calls[0]
    assert index == PROFILE_INDEX
    assert ids == ["profile-user_1"]
    # delete then reload
    assert PROFILE_INDEX in fake.load_index_calls
    # Happy delete path does not write a sentinel.
    assert fake.add_docs_calls == []


async def test_clear_profile_degrades_to_sentinel_when_delete_fails() -> None:
    fake = _FakeMossClient()
    fake.delete_raises = True
    mem = _mem(True, fake)

    assert await mem.clear_profile() is True
    # delete was attempted, then degraded to a sentinel add_docs write.
    assert len(fake.delete_docs_calls) == 1
    assert len(fake.add_docs_calls) == 1
    _index, docs = fake.add_docs_calls[0]
    assert docs[0].id == "profile-user_1"
    assert docs[0].text == EMPTY_PROFILE


# ---------------------------------------------------------------------------
# 5. startup(refresh=True): clear before load
# ---------------------------------------------------------------------------


async def test_startup_refresh_clears_before_load() -> None:
    fake = _FakeMossClient()
    fake.query_result = _FakeResult([_FakeDoc("old profile")])
    mem = _mem(True, fake)

    await mem.startup(refresh=True)

    # A delete (clear) must have happened, and a query (load) afterwards.
    assert len(fake.delete_docs_calls) == 1
    assert len(fake.query_calls) == 1


async def test_startup_without_refresh_does_not_clear() -> None:
    fake = _FakeMossClient()
    fake.query_result = _FakeResult([_FakeDoc("kept profile")])
    mem = _mem(True, fake)

    profile = await mem.startup(refresh=False)
    assert profile == "kept profile"
    assert fake.delete_docs_calls == []


# ---------------------------------------------------------------------------
# 6. history_to_items: filter system / internal-injection / empty
# ---------------------------------------------------------------------------


def test_history_to_items_filters_noise() -> None:
    ctx = _FakeChatCtx(
        [
            _FakeMessage("system", "You are ANSIO."),
            _FakeMessage("user", "I sell skincare for Gen Z."),
            _FakeMessage(
                "assistant",
                "[ANSIO retrieval — internal context] candidates: foo, bar",
            ),
            _FakeMessage("assistant", "[ANSIO memory] known profile note"),
            _FakeMessage("assistant", "   "),
            _FakeMessage("assistant", "Great, let me find creators."),
        ]
    )

    items = history_to_items(ctx)
    assert items == [
        ("user", "I sell skincare for Gen Z."),
        ("assistant", "Great, let me find creators."),
    ]


def test_history_to_items_empty_ctx() -> None:
    assert history_to_items(_FakeChatCtx([])) == []
    assert history_to_items(object()) == []


# ---------------------------------------------------------------------------
# 7. extract_profile_text + extract_and_save: profile saved vs NONE skipped
# ---------------------------------------------------------------------------


async def test_extract_profile_text_returns_text() -> None:
    llm = _FakeLLM(["Product: skincare. ", "Platform: TikTok."])
    profile = await extract_profile_text([("user", "I sell skincare")], llm)
    assert profile == "Product: skincare. Platform: TikTok."
    assert llm.chat_calls == 1


async def test_extract_profile_text_none_sentinel() -> None:
    llm = _FakeLLM(["NONE"])
    assert await extract_profile_text([("user", "hello")], llm) is None


async def test_extract_and_save_saves_when_profile_present() -> None:
    fake = _FakeMossClient()
    mem = _mem(True, fake)
    llm = _FakeLLM(["Product: skincare for Gen Z."])

    ok = await mem.extract_and_save([("user", "I sell skincare")], llm)
    assert ok is True
    assert len(fake.add_docs_calls) == 1
    assert fake.add_docs_calls[0][1][0].text == "Product: skincare for Gen Z."


async def test_extract_and_save_skips_on_none() -> None:
    fake = _FakeMossClient()
    mem = _mem(True, fake)
    llm = _FakeLLM(["NONE"])

    ok = await mem.extract_and_save([("user", "hi")], llm)
    assert ok is False
    assert fake.add_docs_calls == []


async def test_extract_and_save_empty_history_no_llm_call() -> None:
    fake = _FakeMossClient()
    mem = _mem(True, fake)
    llm = _FakeLLM(["should not run"])

    assert await mem.extract_and_save([], llm) is False
    assert llm.chat_calls == 0


# ---------------------------------------------------------------------------
# Pure greeting / instruction helpers
# ---------------------------------------------------------------------------


def test_profile_instructions_none_is_empty() -> None:
    assert profile_instructions(None) == ""


def test_profile_instructions_embeds_profile() -> None:
    out = profile_instructions("Product: skincare")
    assert "Product: skincare" in out
    assert out.startswith("\n\n# Known user profile")


def test_memory_greeting_switches_on_profile() -> None:
    cold = memory_greeting(None)
    warm = memory_greeting("Product: skincare")
    assert "ask what product" in cold
    assert "REMEMBER" in warm
    assert cold != warm


if __name__ == "__main__":  # pragma: no cover - convenience runner
    raise SystemExit(pytest.main([__file__, "-q"]))
