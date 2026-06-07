"""Offline unit tests for the rewritten ANSIO agent tools (PRD T10).

These stub ``MossClient`` so they run with NO Moss credentials and NO network.
They exercise white-list normalization, @-stripping, brand/handle split-param
routing, the wide-pool cache, the meta-tool recall chain, and the
on_user_turn_completed dedup guard (#3414). Live behavior is covered separately
by the credentialed harnesses.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import agent as agent_module
from agent import Assistant

USER_ID = "user_42"


class _FakeDoc:
    def __init__(self, text="", score=None, metadata=None) -> None:
        self.text = text
        self.score = score
        self.metadata = metadata or {}


class _FakeResult:
    def __init__(self, docs, time_taken_ms=5.0) -> None:
        self.docs = docs
        self.time_taken_ms = time_taken_ms


class _FakeMoss:
    """Records query calls; returns canned results keyed by index name."""

    def __init__(self, *args, **kwargs) -> None:
        self.query_calls: list[tuple] = []
        self.loaded: list = []
        self.results_by_index: dict[str, _FakeResult] = {}
        self.default_result = _FakeResult([])

    async def load_index(self, name, *a, **k):
        self.loaded.append(name)

    async def load_indexes(self, names, *a, **k):
        self.loaded.extend(names)

    async def query(self, index, query, options=None):
        self.query_calls.append((index, query, options))
        return self.results_by_index.get(index, self.default_result)


class _Pub:
    def __init__(self) -> None:
        self.published: list[tuple] = []

    async def publish_data(self, payload, reliable=None):
        self.published.append((payload, reliable))


class _Room:
    def __init__(self) -> None:
        self.local_participant = _Pub()


class _Turn:
    """Stand-in for ChatContext: records add_message calls."""

    def __init__(self) -> None:
        self.added: list[dict] = []

    def add_message(self, role, content):
        self.added.append({"role": role, "content": content})


class _Msg:
    def __init__(self, text) -> None:
        self.text_content = text


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(agent_module, "MossClient", _FakeMoss)
    # Avoid constructing a real LLM (no creds): patch build_llm.
    monkeypatch.setattr(agent_module, "build_llm", lambda *a, **k: object())


def _kol_doc(handle, name="Name", **md):
    base = {"handle": handle, "name": name, "niche": "tech",
            "platform": "YouTube", "tier": "mid", "followers": "100000",
            "engagement_pct": "4.0", "price_usd": "2000", "region": "US"}
    base.update(md)
    return _FakeDoc(text=f"{name} profile", score=0.9, metadata=base)


# --- white-list normalization (T2) -----------------------------------------


def test_find_similar_kols_drops_invalid_filters(stub):
    a = Assistant(room=_Room(), user_id=USER_ID)
    a._moss.results_by_index[agent_module.IDX_KOLS] = _FakeResult(
        [_kol_doc("travismedia")]
    )
    asyncio.run(a.find_similar_kols(None, "ai coding creator",
                                    niche="NotANiche", platform="MySpace"))
    _, _, opts = a._moss.query_calls[0]
    assert opts.filter is None
    assert opts.top_k == agent_module.WIDE_POOL_TOP_K
    assert a._state.has_pool()


def test_find_similar_kols_valid_filter_builds_eq(stub):
    a = Assistant(user_id=USER_ID)
    a._moss.results_by_index[agent_module.IDX_KOLS] = _FakeResult([_kol_doc("x")])
    asyncio.run(a.find_similar_kols(None, "brief", niche="Tech", platform="YouTube"))
    _, _, opts = a._moss.query_calls[0]
    assert opts.filter == {"$and": [
        {"field": "platform", "condition": {"$eq": "YouTube"}},
        {"field": "niche", "condition": {"$eq": "tech"}},
    ]}


# --- handle @-strip (T3) ----------------------------------------------------


def test_get_kol_profile_strips_at(stub):
    a = Assistant(user_id=USER_ID)
    a._moss.results_by_index[agent_module.IDX_KOLS] = _FakeResult(
        [_kol_doc("theobennett1", name="Theo")]
    )
    asyncio.run(a.get_kol_profile(None, "@theobennett1"))
    _, _, opts = a._moss.query_calls[0]
    assert opts.filter == {"field": "handle", "condition": {"$eq": "theobennett1"}}


# --- brand/handle split-param (T4) -----------------------------------------


def test_promoted_forward_uses_brand_field(stub):
    a = Assistant(user_id=USER_ID)
    a._moss.results_by_index[agent_module.IDX_CONTENT] = _FakeResult([
        _FakeDoc(metadata={"kol_handle": "k1", "views": "1000", "title": "t"}),
    ])
    asyncio.run(a.find_kols_who_promoted(None, brand="Cursor"))
    _, _, opts = a._moss.query_calls[0]
    assert opts.filter == {"field": "brand", "condition": {"$eq": "cursor"}}


def test_promoted_reverse_uses_handle_field(stub):
    a = Assistant(user_id=USER_ID)
    a._moss.results_by_index[agent_module.IDX_CONTENT] = _FakeResult([
        _FakeDoc(metadata={"kol_handle": "buildwithsam", "views": "5", "title": "t"}),
    ])
    asyncio.run(a.find_kols_who_promoted(None, kol_handle="@buildwithsam"))
    _, _, opts = a._moss.query_calls[0]
    assert opts.filter == {"field": "kol_handle",
                           "condition": {"$eq": "buildwithsam"}}


# --- empty fallback (no crash) ---------------------------------------------


def test_empty_results_return_graceful_text(stub):
    a = Assistant(user_id=USER_ID)
    out = asyncio.run(a.find_competitors(None, "obscure product"))
    assert isinstance(out, str) and "No clear competitors" in out


# --- meta-tool recall chain -------------------------------------------------


def test_recommend_kols_caches_pool_and_ranks(stub):
    a = Assistant(room=_Room(), user_id=USER_ID)
    docs = [_kol_doc(f"k{i}", name=f"K{i}", engagement_pct=str(5 - i),
                     price_usd=str(1000 + i * 500)) for i in range(4)]
    a._moss.results_by_index[agent_module.IDX_KOLS] = _FakeResult(docs)
    out = asyncio.run(a.recommend_kols(None, "ai coding tool", niche="tech"))
    assert "Top picks" in out
    assert a._state.recommended is True
    assert len(a._state.candidate_pool) == 4
    payloads = [json.loads(p.decode()) for p, _ in a._room.local_participant.published]
    assert any(p["type"] == "alpha_ranking" for p in payloads)


def test_recommend_kols_budget_filters_in_python_no_requery(stub):
    a = Assistant(user_id=USER_ID)
    docs = [_kol_doc("cheap", price_usd="300"),
            _kol_doc("pricey", price_usd="9000")]
    a._moss.results_by_index[agent_module.IDX_KOLS] = _FakeResult(docs)
    asyncio.run(a.recommend_kols(None, "brief", budget=500))
    kol_queries = [c for c in a._moss.query_calls if c[0] == agent_module.IDX_KOLS]
    assert len(kol_queries) == 1  # budget applied in Python, no re-query (T1)
    ranked = a._rerank(budget=500)
    assert all(float(c["metadata"]["price_usd"]) <= 500 for c in ranked)


# --- on_user_turn_completed dedup guard (#3414) ----------------------------


def test_turn_injection_dedup_guard(stub):
    a = Assistant(room=_Room(), user_id=USER_ID)
    a._moss.results_by_index[agent_module.IDX_KOLS] = _FakeResult([_kol_doc("k")])
    turn = _Turn()
    msg = _Msg("find me ai coding creators")
    asyncio.run(a.on_user_turn_completed(turn, msg))
    asyncio.run(a.on_user_turn_completed(turn, msg))  # duplicate same-turn fire
    assert len(turn.added) == 1  # injected exactly once despite two fires
    assert len(a._moss.query_calls) == 1
