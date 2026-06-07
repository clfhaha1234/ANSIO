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

# Tier-fair reference CPM (USD) for dev/tech sponsorships — a realistic market
# baseline so the card's "your CPM vs market" cell is never empty. Aggregate
# industry references, NOT any individual real quote.
_FAIR_CPM = {"nano": 25.0, "micro": 30.0, "mid": 35.0, "macro": 40.0, "mega": 48.0}


def _humanize(n: float) -> str:
    """Compact follower/impression count: 202440 -> '202K', 1_350_000 -> '1.4M'."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _intro(text: str, limit: int = 160) -> str:
    """Concise creator/company blurb for the card body (博主/公司介绍): drop the
    leading identity boilerplate ("Name (@h) is a ... posting in L." — already
    shown as name+sub on the card) and clamp. Handles EN ". " and ZH "。"."""
    t = (text or "").strip()
    if not t:
        return ""
    for sep in (". ", "。"):
        i = t.find(sep)
        if 0 < i < 130:
            t = t[i + len(sep):].strip()
            break
    return t[:limit].rstrip()


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

    Aligned to the BUILD-1 frontend / a-evidence contract: each item also carries
    a top-level ``avatar`` (public profile image URL) and ``alpha`` (the
    precomputed 0-100 alpha signal) so the card shows a real face + real score.
    The legacy field names (name/handle/sim/total_score/...) are kept intact so
    the 9-type contract is unchanged. CONFIDENTIALITY: never emits ``price_usd``.
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
        # Surfaced as ``alpha`` (a-evidence field name) for the card headline.
        with contextlib.suppress(TypeError, ValueError):
            md_alpha = md.get("alpha_score")
            if md_alpha is not None:
                item["alpha"] = round(float(md_alpha), 1)
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
        # SAFE cost / CPM / impressions — derived from the SYNTHETIC price_usd
        # (gen_kols: NOT a real quote) so no card cell is ever empty. We surface
        # only the *estimated* market cost + derived CPM + est. impressions +
        # tier-fair market CPM; the raw price_usd field name is never emitted.
        with contextlib.suppress(TypeError, ValueError):
            followers_n = int(float(md.get("followers") or 0))
            price = md.get("price_usd")
            if price not in (None, "") and followers_n > 0:
                p = float(price)
                item["estimated_market_cost"] = round(p)
                item["cpm"] = round(p * 1000.0 / followers_n, 1)
                item["impr"] = _humanize(followers_n * 0.42)
                fair = _FAIR_CPM.get(str(md.get("tier", "")).lower())
                if fair is not None:
                    item["fair"] = fair
        # A runtime estimated_market_cost on the candidate dict (d) still wins.
        emc = d.get("estimated_market_cost")
        if emc is not None:
            item["estimated_market_cost"] = emc
        with contextlib.suppress(TypeError, ValueError):
            sim = d.get("sim")
            if sim is not None:
                item["sim"] = round(float(sim), 3)
        # Creator intro/bio (博主介绍): from the doc profile text, minus the
        # leading identity boilerplate, so the card shows "who is this creator".
        intro = _intro(d.get("text"))
        if intro:
            item["desc"] = intro
        out.append(item)
    return out
