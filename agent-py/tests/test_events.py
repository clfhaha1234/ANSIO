"""Offline unit tests for the events payload builders (no Moss/LiveKit/network).

The old chain_step schema was retired by the 8-step orchestration refactor
(cards now carry their own ``step``); these tests pin the surviving contract,
most importantly the CONFIDENTIALITY red line: ``kol_items`` must NEVER emit a
raw ``price_usd`` — only the aggregate-derived ``estimated_market_cost``.
"""

from __future__ import annotations

from events import EVIDENCE_TYPES, build_evidence, kol_items

# --- build_evidence (frozen card contract) -----------------------------------


def test_build_evidence_minimal_shape():
    card = build_evidence("alpha_ranking", step=5, latency_ms=8.3456)
    assert card["type"] == "alpha_ranking"
    assert card["card_type"] == "alpha_ranking"
    assert card["step"] == 5
    assert card["latency_ms"] == 8.35  # rounded to 2 dp
    assert card["items"] == []
    assert isinstance(card["timestamp"], float)


def test_build_evidence_unknown_type_coerces_to_content_hits():
    assert build_evidence("not_a_real_card")["type"] == "content_hits"


def test_build_evidence_all_known_types_pass_through():
    for ct in EVIDENCE_TYPES:
        assert build_evidence(ct)["type"] == ct


# --- kol_items confidentiality red line ---------------------------------------


def _doc(**md):
    return {"metadata": md}


def test_kol_items_never_emits_raw_price_usd():
    items = kol_items([
        _doc(name="Example KOL", handle="examplekol", followers="10000",
             price_usd="1234", tier="micro"),
    ])
    assert len(items) == 1
    item = items[0]
    assert "price_usd" not in item, "raw per-KOL price must never leave the agent"
    # The sanctioned public form is the derived estimate.
    assert item.get("estimated_market_cost") == 1234


def test_kol_items_survives_missing_price_and_followers():
    items = kol_items([_doc(name="No Price", handle="noprice")])
    assert items[0]["name"] == "No Price"
    assert "price_usd" not in items[0]


def test_kol_items_passes_public_fields_and_avatar():
    items = kol_items([
        _doc(name="Avatar KOL", handle="avatarkol", platform="YouTube",
             followers="5000", niche="tech", engagement_pct="6.1",
             avatar_url="https://example.com/a.png"),
    ])
    item = items[0]
    assert item["platform"] == "YouTube"
    assert item["niche"] == "tech"
    assert item.get("avatar") == "https://example.com/a.png"
