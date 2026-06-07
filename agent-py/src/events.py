"""Evidence-stream events for the ANSIO right panel (PRD §4.3 + bridge 04/F4).

Each tool call pushes one evidence card over the LiveKit DataChannel. The
frontend (``bridge.js`` ``renderEvidence``) dual-recognizes two payload shapes:

1. ``moss_context`` — the already-shipped contract (agent.py legacy), kept as a
   safety net (F4 §0.4 / N2).
2. The 9 typed evidence cards below (target state) — ``{type, step, index,
   latency_ms, items[], insight, source, title}``. ``card_type`` == ``type``.

This module ONLY builds the JSON-serializable payload dicts and (optionally)
publishes them. It imports nothing from livekit; the publisher is passed in as
``room`` (duck-typed: ``room.local_participant.publish_data(payload, reliable)``)
so it stays unit-testable offline.

CONFIDENTIALITY: payloads carry only the *estimated market cost* (aggregate-
derived) — never an individual real quote. Callers must pass safe fields only.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("agent.events")

# The 9 evidence card types, aligned to the standalone card "kind" renderers
# (F4 §4 KIND map). card_type is the discriminator the frontend switches on.
EVIDENCE_TYPES = (
    "competitor_landscape",
    "content_hits",
    "playbook_hit",
    "kol_profile",
    "similar_creators",
    "alpha_ranking",
    "bundle",
    "content_strategy",
    "roi_forecast",
)


def build_evidence(
    card_type: str,
    *,
    step: int | str = "··",
    index: str = "moss",
    latency_ms: float | None = None,
    items: list[dict] | None = None,
    insight: str = "",
    source: str = "",
    title: str = "",
) -> dict:
    """Build a JSON-serializable evidence payload (the 9-type contract).

    ``latency_ms`` is the REAL measured Moss wall-clock for this hop (never
    hard-coded) — the HUD millisecond number. Unknown types are coerced to a
    generic ``content_hits`` card so the frontend never drops a payload.
    """
    if card_type not in EVIDENCE_TYPES:
        card_type = "content_hits"
    return {
        "type": card_type,
        "card_type": card_type,
        "step": step,
        "index": index,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "items": items or [],
        "insight": insight,
        "source": source,
        "title": title or card_type.replace("_", " ").title(),
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }


def build_chain_step(
    card_type: str,
    t: str,
    *,
    src: str = "",
    res: bool = False,
    latency_ms: float | None = None,
) -> dict:
    """Build a JSON-serializable ``chain_step`` payload (FROZEN schema).

    Emitted BEFORE each evidence card so the frontend can stream multi-step
    retrieval reasoning ("chain-first"). ``card_type`` is coerced exactly like
    ``build_evidence`` — an unknown type degrades to ``content_hits`` so the
    frontend never drops a payload. ``latency_ms`` carries the REAL measured
    Moss wall-clock and rides only on the final ``res=True`` result row.
    """
    if card_type not in EVIDENCE_TYPES:
        card_type = "content_hits"
    return {
        "type": "chain_step",
        "card_type": card_type,
        "t": t,
        "src": src,
        "res": bool(res),
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }


async def publish_evidence(room, payload: dict) -> None:
    """Publish one evidence payload to the room. No-op if room is None.

    Never raises (degradation discipline): a failed publish must not break the
    voice turn. ``reliable=True`` so the right-panel card is not dropped.
    """
    if room is None:
        return
    try:
        encoded = json.dumps(payload, default=str).encode("utf-8")
        await room.local_participant.publish_data(payload=encoded, reliable=True)
    except Exception:
        logger.exception("Failed to publish evidence card %s", payload.get("type"))


def kol_items(docs: list[dict], limit: int = 6) -> list[dict]:
    """Render KOL/candidate dicts into safe card items (no real quote).

    Aligned to BOTH frontends' score chains: each item carries a top-level
    ``avatar`` (public profile image URL) plus the precomputed 0-100 alpha signal
    under TWO field names so neither renderer drops it:

    * ``alpha``       — a-evidence / BUILD-1 standalone card field name.
    * ``alpha_score`` — the field the conversational app's ``renderItem`` score
      chain actually reads (``score -> total_score -> alpha_score -> ...``) and
      labels "alpha N". Without this the precomputed headline alpha never shows
      in the conversational right panel (it fell through to ``sim``).

    The runtime scoring breakdown (match/perf/alpha/total) still overrides both
    when present (alpha_ranking card). Legacy field names (name/handle/sim/...)
    are kept intact so the 9-type contract is unchanged. CONFIDENTIALITY: never
    emits ``price_usd`` — only the aggregate-derived estimated market cost.
    """
    out: list[dict] = []
    for d in docs[:limit]:
        md = d.get("metadata", d) if isinstance(d, dict) else {}
        item = {
            "name": md.get("name", "?"),
            "handle": md.get("handle", ""),
            "platform": md.get("platform", ""),
            "followers": md.get("followers", ""),
            "niche": md.get("niche", ""),
            "engagement_pct": md.get("engagement_pct", ""),
            "region": md.get("region", ""),
        }
        # Public avatar URL (real face for real KOLs; initials otherwise).
        avatar = md.get("avatar_url") or md.get("avatar")
        if avatar:
            item["avatar"] = avatar
        # Precomputed standalone alpha (baked into metadata at index build).
        # Emitted under BOTH names: ``alpha`` (a-evidence headline) and
        # ``alpha_score`` (the conversational app's renderItem score chain), so
        # the precomputed headline alpha actually renders in either frontend.
        with contextlib.suppress(TypeError, ValueError):
            md_alpha = md.get("alpha_score")
            if md_alpha is not None:
                alpha_val = round(float(md_alpha), 1)
                item["alpha"] = alpha_val
                item["alpha_score"] = alpha_val
        # Precomputed sub-dimensions (optional card detail, all public-derived).
        for k in ("influence_score", "momentum_score", "brand_fit_percentile"):
            if md.get(k) is not None:
                with contextlib.suppress(TypeError, ValueError):
                    item[k] = round(float(md[k]), 1)
        # Runtime scoring breakdown columns when present (alpha_ranking card).
        # These live on the candidate dict ``d`` (added by scoring.py), and the
        # runtime alpha_score (pool-relative) takes precedence for ``alpha``.
        for k in ("match_score", "perf_score", "alpha_score", "total_score"):
            if k in d:
                item[k] = d[k]
        if d.get("alpha_score") is not None:
            with contextlib.suppress(TypeError, ValueError):
                item["alpha"] = (
                    round(float(d["alpha_score"]) * 100, 1)
                    if float(d["alpha_score"]) <= 1
                    else round(float(d["alpha_score"]), 1)
                )
        # SAFE cost only — estimated market cost, never the real quote.
        emc = d.get("estimated_market_cost")
        if emc is not None:
            item["estimated_market_cost"] = emc
        with contextlib.suppress(TypeError, ValueError):
            sim = d.get("sim")
            if sim is not None:
                item["sim"] = round(float(sim), 3)
        out.append(item)
    return out
