"""Python-side scoring & ranking for ANSIO (Plan B, task F1).

This module consumes the wide candidate pool returned by ``find_similar_kols``
(``top_k=80``, cached in session) and produces the final ranked top-5. It does
all budget filtering and Alpha scoring in Python so changing the budget or the
weight mix re-ranks instantly **without re-querying Moss** (PRD T1).

Plan B (PRD T4b — recommended, zero index rebuild):
* Alpha = engagement / price_norm  (under-priced relative to engagement = high
  Alpha = "undervalued" creator).
* Audience overlap uses niche/region/platform as a proxy (no growth_3m_pct etc.).

Benchmark calibration:
* The "fair price" reference comes from the **aggregated** benchmark
  ``benchmark_agg.json`` (tier x category median CPM), NOT from any individual
  quote. If the benchmark file is absent (gitignored, may not exist on a clean
  clone), scoring degrades gracefully to a pool-relative price_norm so the demo
  still runs.

CONFIDENTIALITY: this module reads only the AGGREGATE benchmark. It never reads
the xlsx and never has access to an individual creator's real quote. The
``price_usd`` it divides by is the synthetic per-post rate stored on the KOL doc,
never the real spreadsheet price.
"""

from __future__ import annotations

import json
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = AGENT_DIR / "benchmark_agg.json"  # gitignored aggregate

DEFAULT_WEIGHTS = {"match": 0.4, "perf": 0.3, "alpha": 0.3}


# ---------------------------------------------------------------------------
# Benchmark loading (aggregate only)
# ---------------------------------------------------------------------------


def load_benchmark(path: Path | None = None) -> dict:
    """Load the aggregated (tier x category) benchmark.

    Returns ``{}`` if the file is absent — callers degrade to pool-relative
    scoring. Never raises on a missing file (it is intentionally gitignored).
    """
    p = path or BENCHMARK_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def benchmark_cpm(benchmark: dict, tier: str, category: str = "tech") -> float | None:
    """Median CPM for a (tier, category) cell; falls back to the global median.

    Pure aggregate lookup — no individual value involved.
    """
    if not benchmark:
        return None
    cells = benchmark.get("cells", {})
    cell = cells.get(f"{tier}|{category}")
    if cell and "cpm_median" in cell:
        return float(cell["cpm_median"])
    g = benchmark.get("global", {})
    if g.get("cpm_median") is not None:
        return float(g["cpm_median"])
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num(md: dict, key: str, default: float = 0.0) -> float:
    """Read a metadata field that may be stored as a numeric string."""
    v = md.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _minmax(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if hi == lo:
        return lo, lo + 1.0  # avoid div-by-zero; flat field -> all 0.5 after norm
    return lo, hi


def _norm(x: float, lo: float, hi: float) -> float:
    return (x - lo) / (hi - lo) if hi > lo else 0.5


def estimated_market_cost(md: dict, benchmark: dict) -> int | None:
    """The number that is SAFE to show on the demo screen.

    = benchmark median CPM (tier x category) * (followers / 1000), i.e. an
    *estimated market cost* derived purely from the aggregate benchmark and the
    creator's public follower count. It is explicitly NOT the creator's real
    quote (which never leaves the aggregator).
    """
    tier = str(md.get("tier", "")) or "mid"
    category = "tech"
    cpm = benchmark_cpm(benchmark, tier, category)
    if cpm is None:
        return None
    followers = _num(md, "followers")
    if followers <= 0:
        return None
    return int(cpm * followers / 1000.0)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def score_and_rank(
    candidates: list[dict],
    slots: dict | None = None,
    weights: dict | None = None,
    benchmark: dict | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Budget-filter + Alpha-score the wide candidate pool -> top_n.

    candidates: list of dicts each with a ``metadata`` (or flat) dict carrying
        ``followers``, ``engagement_pct``, ``price_usd``, ``tier``, ``niche``,
        ``platform``, ``region``, and a ``sim`` similarity score.
    slots: optional {budget, platform, niche, region, ...} hard constraints.
    weights: {match, perf, alpha}; defaults to DEFAULT_WEIGHTS.
    benchmark: aggregate benchmark dict (loaded once via load_benchmark()).

    Returns top_n dicts, each annotated with score breakdown columns:
        match_score, perf_score, alpha_score, total_score, estimated_market_cost.
    Pure Python, no Moss call — re-runnable on budget/weight change (T1).
    """
    slots = slots or {}
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if benchmark is None:
        benchmark = load_benchmark()

    def md_of(c: dict) -> dict:
        return c.get("metadata", c)

    # --- 1. Hard budget filter (Python, on the wide pool) ----------------
    budget = slots.get("budget")
    pool = []
    for c in candidates:
        md = md_of(c)
        if budget is not None:
            price = _num(md, "price_usd")
            if price > float(budget):
                continue
        if slots.get("platform") and md.get("platform") != slots["platform"]:
            continue
        if slots.get("niche") and md.get("niche") != slots["niche"]:
            continue
        pool.append(c)

    if not pool:
        return []

    # --- 2. Raw signals --------------------------------------------------
    sims, perfs, alphas = [], [], []
    for c in pool:
        md = md_of(c)
        sim = _num(c, "sim", default=_num(md, "sim", 0.0))
        engagement = _num(md, "engagement_pct")
        price = max(1.0, _num(md, "price_usd", 1.0))

        # price_norm: price relative to the aggregate benchmark "fair price".
        # Plan B Alpha = engagement / price_norm. Higher engagement per dollar
        # (vs the tier x category market) => more undervalued.
        bench_cpm = benchmark_cpm(benchmark, str(md.get("tier", "mid")), "tech")
        followers = _num(md, "followers")
        if bench_cpm and followers > 0:
            fair_price = max(1.0, bench_cpm * followers / 1000.0)
            price_norm = price / fair_price
        else:
            # Degrade: pool-relative (raw price; min-max'd across pool below).
            price_norm = price
        alpha = engagement / max(0.05, price_norm)

        c["_sim"] = sim
        c["_perf"] = engagement
        c["_alpha"] = alpha
        sims.append(sim)
        perfs.append(engagement)
        alphas.append(alpha)

    # --- 3. min-max normalize each signal --------------------------------
    s_lo, s_hi = _minmax(sims)
    p_lo, p_hi = _minmax(perfs)
    a_lo, a_hi = _minmax(alphas)

    for c in pool:
        match_score = _norm(c["_sim"], s_lo, s_hi)
        perf_score = _norm(c["_perf"], p_lo, p_hi)
        alpha_score = _norm(c["_alpha"], a_lo, a_hi)
        total = (
            weights["match"] * match_score
            + weights["perf"] * perf_score
            + weights["alpha"] * alpha_score
        )
        md = md_of(c)
        c["match_score"] = round(match_score, 3)
        c["perf_score"] = round(perf_score, 3)
        c["alpha_score"] = round(alpha_score, 3)
        c["total_score"] = round(total, 4)
        c["estimated_market_cost"] = estimated_market_cost(md, benchmark)

    ranked = sorted(pool, key=lambda c: c["total_score"], reverse=True)
    # Clean the private scratch fields before returning.
    for c in ranked:
        for k in ("_sim", "_perf", "_alpha"):
            c.pop(k, None)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Audience overlap (Plan B proxy) — used by the bundle card
# ---------------------------------------------------------------------------


def audience_overlap(a: dict, b: dict) -> float:
    """Proxy overlap in [0,1] from niche/region/platform agreement (Plan B).

    No growth/audience-composition fields exist on the live index, so this is a
    deliberate proxy: shared niche dominates, region and platform add weight.
    """
    ma, mb = a.get("metadata", a), b.get("metadata", b)
    score = 0.0
    if ma.get("niche") and ma.get("niche") == mb.get("niche"):
        score += 0.6
    if ma.get("region") and ma.get("region") == mb.get("region"):
        score += 0.25
    if ma.get("platform") and ma.get("platform") == mb.get("platform"):
        score += 0.15
    return round(min(1.0, score), 3)
