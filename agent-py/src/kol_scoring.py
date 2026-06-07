"""Offline, deterministic per-KOL alpha + sub-dimension scoring (BUILD-2).

This is the *precompute* path described in
``.omc/research/ansio-v5/c-ansio-data.md`` §3-4: every KOL's alpha and the
sub-dimensions that feed it are computed ONCE, offline, from PUBLIC fields only,
and baked into ``ansio_kols`` metadata. At runtime Moss only *retrieves* these
numbers — it never recomputes them — so the evidence card shows a real,
explainable score with zero query-time latency.

Distinct from ``scoring.py``: that module does a *pool-relative* min-max rerank
of a live candidate set at runtime (Plan B). This module produces a *standalone
absolute* score for a single creator from their own public metrics, suitable for
storing in the index. The two are complementary (retrieval shows the absolute
precomputed score; the agent's final top-5 rerank still uses scoring.py).

Formula (ported from ANSIO's alpha engine, c-ansio-data.md Appendix):
  influence = 0.6*log10(followers)/log10(10M) + 0.3*engagement_ratio + 0.1*tier_presence
  momentum  = engagement-velocity proxy (smaller, hotter creators score higher)
  brand_fit = niche/tier coding-relevance proxy -> percentile within tier cohort
  alpha     = 0.40*momentum + 0.30*brand_fit + 0.20*influence + 0.10*engagement

All inputs are PUBLIC (follower count, engagement %, tier). No pricing, no real
quote, no private contact ever enters a score. Deterministic: same inputs ->
same outputs across runs (no RNG), so index rebuilds are stable.
"""

from __future__ import annotations

import math

# Tier -> a small cross-platform "presence/credibility" bonus and a momentum
# bias (nano/micro creators convert engagement into momentum more efficiently).
_TIER_PRESENCE = {"nano": 0.55, "micro": 0.7, "mid": 0.82, "macro": 0.9, "mega": 1.0}
_TIER_MOMENTUM_BIAS = {
    "nano": 1.0,
    "micro": 0.92,
    "mid": 0.8,
    "macro": 0.68,
    "mega": 0.55,
}

_LOG_CAP = math.log10(10_000_000)  # 10M followers == saturation


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def influence_score(followers: float, engagement_pct: float, tier: str) -> float:
    """0-100. Log-normalized reach + engagement + cross-platform presence."""
    f = max(1.0, float(followers))
    follower_score = min(math.log10(f) / _LOG_CAP, 1.0)  # 0..1
    engagement_score = min(float(engagement_pct) / 10.0, 1.0)  # 10% eng == saturated
    presence = _TIER_PRESENCE.get(str(tier), 0.7)
    composite = follower_score * 0.6 + engagement_score * 0.3 + presence * 0.1
    return round(_clip(composite * 100.0), 1)


def momentum_score(engagement_pct: float, tier: str) -> float:
    """0-100. Engagement velocity proxy — hotter/smaller creators rank higher.

    Without a follower-growth time series in the public seed, we proxy momentum
    by engagement scaled by a tier bias (nano/micro convert attention into
    growth faster). Deterministic, monotonic, explainable.
    """
    bias = _TIER_MOMENTUM_BIAS.get(str(tier), 0.8)
    eng_norm = min(float(engagement_pct) / 8.0, 1.0)  # 8%+ engagement == hot
    return round(_clip(eng_norm * bias * 100.0), 1)


def brand_fit_percentile(followers: float, engagement_pct: float, tier: str) -> float:
    """0-100 percentile-style brand-fit signal for AI-coding/dev brands.

    A proxy combining mid-tier sweet-spot (advertisers prefer reachable mid/micro
    creators with strong engagement) and engagement strength. Expressed on a
    0-100 scale so the card can read it as a percentile rank.
    """
    # Mid/micro is the advertiser sweet spot for dev-tool sponsorships.
    tier_fit = {"nano": 0.5, "micro": 0.85, "mid": 1.0, "macro": 0.8, "mega": 0.65}.get(
        str(tier), 0.7
    )
    eng_norm = min(float(engagement_pct) / 9.0, 1.0)
    score = (tier_fit * 0.6 + eng_norm * 0.4) * 100.0
    return round(_clip(score), 1)


def alpha_score(followers: float, engagement_pct: float, tier: str) -> float:
    """0-100 composite alpha (the headline signal shown on the leaderboard card).

    alpha = 0.40*momentum + 0.30*brand_fit + 0.20*influence + 0.10*engagement
    (ANSIO alpha weights, c-ansio-data.md Appendix). Deterministic & absolute.
    """
    inf = influence_score(followers, engagement_pct, tier)
    mom = momentum_score(engagement_pct, tier)
    fit = brand_fit_percentile(followers, engagement_pct, tier)
    eng_norm = min(float(engagement_pct) / 8.0, 1.0) * 100.0
    alpha = 0.40 * mom + 0.30 * fit + 0.20 * inf + 0.10 * eng_norm
    return round(_clip(alpha), 1)


def precompute_scores(followers: float, engagement_pct: float, tier: str) -> dict:
    """All four precomputed signals for one KOL, ready to merge into metadata.

    Keys (all PUBLIC-derived, 0-100 floats): ``alpha_score``,
    ``influence_score``, ``momentum_score``, ``brand_fit_percentile``.
    """
    return {
        "alpha_score": alpha_score(followers, engagement_pct, tier),
        "influence_score": influence_score(followers, engagement_pct, tier),
        "momentum_score": momentum_score(engagement_pct, tier),
        "brand_fit_percentile": brand_fit_percentile(followers, engagement_pct, tier),
    }
