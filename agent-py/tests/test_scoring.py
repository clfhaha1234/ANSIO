"""Offline unit tests for scoring (Plan B) + the gating state machine.

Pure Python, no Moss, no livekit, no credentials (PRD T10). Validates the
budget FLIP (no re-query), the weight FLIP (alpha re-rank), graceful degradation
when the aggregate benchmark is absent, and the recommendation gate conditions.
"""

from __future__ import annotations

from scoring import audience_overlap, score_and_rank
from state import AnsioState


def _cand(handle, followers, engagement, price, **md):
    base = {"handle": handle, "name": handle, "niche": "tech",
            "platform": "YouTube", "tier": "mid", "region": "US",
            "followers": str(followers), "engagement_pct": str(engagement),
            "price_usd": str(price)}
    base.update(md)
    return {"metadata": base, "sim": 0.8, "text": ""}


POOL = [
    _cand("cheap_hot", 50000, 8.0, 400),    # high engagement, low price -> high alpha
    _cand("mid", 200000, 4.0, 3000),
    _cand("pricey", 800000, 3.0, 12000),
]


# --- budget FLIP (no re-query) ---------------------------------------------


def test_budget_filter_removes_over_budget():
    ranked = score_and_rank(POOL, slots={"budget": 500})
    handles = [c["metadata"]["handle"] for c in ranked]
    assert handles == ["cheap_hot"]  # only the sub-$500 creator survives


def test_no_budget_keeps_all():
    ranked = score_and_rank(POOL, slots={})
    assert len(ranked) == 3


# --- weight FLIP (alpha emphasis re-ranks to value pick) -------------------


def test_alpha_weight_promotes_undervalued():
    balanced = score_and_rank(POOL, weights={"match": 0.4, "perf": 0.3, "alpha": 0.3})
    alpha_heavy = score_and_rank(POOL, weights={"match": 0.1, "perf": 0.1, "alpha": 0.8})
    assert alpha_heavy[0]["metadata"]["handle"] == "cheap_hot"
    assert "alpha_score" in balanced[0]


# --- graceful degradation (no benchmark) -----------------------------------


def test_scoring_degrades_without_benchmark():
    ranked = score_and_rank(POOL, slots={}, benchmark={})
    assert len(ranked) == 3
    assert all(c.get("estimated_market_cost") is None for c in ranked)


def test_empty_pool_returns_empty():
    assert score_and_rank([], slots={"budget": 100}) == []


# --- audience overlap proxy -------------------------------------------------


def test_audience_overlap_same_niche_region_platform():
    a = _cand("a", 1, 1, 1)
    b = _cand("b", 2, 2, 2)
    assert audience_overlap(a, b) == 1.0


def test_audience_overlap_partial():
    a = _cand("a", 1, 1, 1, region="US")
    b = _cand("b", 2, 2, 2, region="UK")
    assert 0.0 < audience_overlap(a, b) < 1.0


# --- gating state machine ---------------------------------------------------


def test_slots_complete_gate():
    s = AnsioState()
    assert not s.slots_complete()
    s.update_slots(category="tech", audience="devs", platform="YouTube",
                   budget=500, goal="conversion")
    assert s.slots_complete()


def test_update_slots_never_clobbers_with_none():
    s = AnsioState()
    s.update_slots(category="tech")
    s.update_slots(category=None, platform="")
    assert s.slots["category"] == "tech"
    assert "platform" not in s.slots


def test_recommend_on_user_urge():
    s = AnsioState()
    assert s.should_recommend(user_urged=True, top5_overlap=0)


def test_recommend_on_stable_top5():
    s = AnsioState()
    assert s.should_recommend(user_urged=False, top5_overlap=4)


def test_recommend_forced_on_turn_limit():
    s = AnsioState()
    s.turn_count = 5
    assert s.should_recommend(user_urged=False, top5_overlap=0)


def test_note_top5_overlap():
    s = AnsioState()
    r1 = [{"metadata": {"handle": h}} for h in ("a", "b", "c", "d", "e")]
    assert s.note_top5(r1) == 0
    r2 = [{"metadata": {"handle": h}} for h in ("a", "b", "c", "d", "z")]
    assert s.note_top5(r2) == 4
