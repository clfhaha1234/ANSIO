"""Offline unit tests for the events payload builders.

These import ``events`` directly (no Moss, no LiveKit, no network) and assert the
FROZEN ``chain_step`` schema the frontend streams as multi-step reasoning before
each evidence card. Mirrors the stub-only style of ``tests/test_tools.py``.
"""

from __future__ import annotations

from events import EVIDENCE_TYPES, build_chain_step

# --- shape (FROZEN schema) --------------------------------------------------


def test_build_chain_step_minimal_shape():
    step = build_chain_step("alpha_ranking", "Scanning for underpriced creators")
    assert step["type"] == "chain_step"
    assert step["card_type"] == "alpha_ranking"
    assert step["t"] == "Scanning for underpriced creators"
    assert step["src"] == ""
    assert step["res"] is False
    assert step["latency_ms"] is None
    assert isinstance(step["timestamp"], float)
    # Exactly the frozen keys, nothing extra.
    assert set(step) == {"type", "card_type", "t", "src", "res", "latency_ms",
                         "timestamp"}


def test_build_chain_step_result_row_carries_latency():
    step = build_chain_step(
        "similar_creators", "80 candidates recalled",
        src="ansio_kols", res=True, latency_ms=8.3456,
    )
    assert step["res"] is True
    assert step["src"] == "ansio_kols"
    assert step["latency_ms"] == 8.35  # rounded to 2 dp like build_evidence


# --- unknown card_type coercion (parity with build_evidence) ----------------


def test_build_chain_step_unknown_type_coerces_to_content_hits():
    step = build_chain_step("not_a_real_card", "querying")
    assert step["card_type"] == "content_hits"


def test_build_chain_step_all_known_types_pass_through():
    for ct in EVIDENCE_TYPES:
        assert build_chain_step(ct, "step")["card_type"] == ct


# --- res is always coerced to a real bool -----------------------------------


def test_build_chain_step_res_is_coerced_to_bool():
    truthy = build_chain_step("kol_profile", "loaded", res=1)
    falsy = build_chain_step("kol_profile", "loading", res=0)
    assert truthy["res"] is True
    assert falsy["res"] is False
