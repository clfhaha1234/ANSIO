"""Generate a synthetic but realistic sponsored-content dataset for index B.

Writes ``agent-py/content.jsonl`` — one JSON document per line, shaped for Moss:

    {"id": "content-NNNN", "text": "<title + caption + caption-summary>",
     "metadata": {kol_id, kol_handle, brand, platform, title,
                  views, likes, comments, engagement_pct, date}}

This is the **most important index** in the ANSIO demo (PRD §2.1/§2.2): it powers
"which KOLs promoted brand X" (forward, filter on ``brand``) and "what has @handle
promoted" (reverse, filter on ``kol_handle``).

Design rules (from PRD §2.2 / §2.3 and 02-prd-digest.md §④):
- ``brand`` is normalized to lowercase (cursor / copilot / replit / codeium / ...).
- ``kol_id`` / ``kol_handle`` reference **real** entries in ``kols.json`` (no @ prefix).
- Every metadata value is a string (numbers stored as numeric strings) — Moss
  metadata is ``Dict[str, str]`` and does ``$gte/$lte`` comparison on numeric strings.
- ``platform`` is taken from the linked KOL (so it is always a valid enum value and
  consistent with that creator's actual channel).
- A piece of content mentioning multiple brands is split into **one document per
  brand** (Moss metadata does not support arrays) — same ``text``/``kol``, distinct
  ``id`` + ``brand``.
- ``text`` is natural language = post title + caption + a caption/subtitle summary.

Deterministic: seeded RNG so re-runs produce a byte-identical file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KOLS_PATH = ROOT / "kols.json"
OUT_PATH = ROOT / "content.jsonl"

SEED = 42
# Number of base posts to generate (before multi-brand fan-out). Multi-brand
# fan-out pushes the final document count to ~1600.
N_BASE_POSTS = 1450

# Brand catalog. The dict key is the canonical lowercase brand stored in metadata;
# the values are surface forms ("aliases") woven into the natural-language text so
# the "LLM-extracted + normalized to lowercase" requirement is demonstrated and
# semantic search still hits.
AI_CODING_BRANDS = {
    "cursor": ["Cursor", "Cursor AI", "cursor.sh", "the Cursor editor"],
    "copilot": ["Copilot", "GitHub Copilot", "Copilot X"],
    "replit": ["Replit", "Replit Agent", "Replit Ghostwriter"],
    "codeium": ["Codeium", "Codeium AI", "Windsurf by Codeium"],
    "windsurf": ["Windsurf", "the Windsurf editor", "Windsurf IDE"],
    "tabnine": ["Tabnine", "Tabnine AI"],
    "claude-code": ["Claude Code", "Anthropic Claude Code"],
    "codex": ["Codex", "OpenAI Codex", "the Codex CLI"],
    "v0": ["v0", "v0.dev", "Vercel v0"],
    "bolt": ["Bolt", "Bolt.new", "StackBlitz Bolt"],
    "lovable": ["Lovable", "Lovable.dev"],
    "aider": ["Aider", "aider-chat"],
    "continue": ["Continue", "Continue.dev"],
    "supermaven": ["Supermaven"],
    "amazon-q": ["Amazon Q", "Amazon Q Developer", "CodeWhisperer"],
}

# Adjacent dev-tool / SaaS brands so non-AI-coding KOLs also have plausible deals,
# and so the "其实做 X" off-script pivots have content to land on.
OTHER_BRANDS = {
    "vercel": ["Vercel"],
    "supabase": ["Supabase"],
    "linear": ["Linear"],
    "notion": ["Notion", "Notion AI"],
    "figma": ["Figma"],
    "raycast": ["Raycast"],
    "warp": ["Warp", "the Warp terminal"],
    "postman": ["Postman"],
    "mongodb": ["MongoDB", "MongoDB Atlas"],
    "datadog": ["Datadog"],
}

ALL_BRANDS = {**AI_CODING_BRANDS, **OTHER_BRANDS}

# Per-brand sampling weights for the *primary* brand of each base post. Heavily
# skewed toward the demo brands so the hard floors (Cursor >=60, the others >=35)
# are met with margin even before multi-brand fan-out.
PRIMARY_BRAND_WEIGHTS = {
    "cursor": 95,
    "copilot": 60,
    "replit": 55,
    "codeium": 55,
    "windsurf": 45,
    "claude-code": 40,
    "codex": 35,
    "v0": 32,
    "bolt": 30,
    "lovable": 28,
    "tabnine": 25,
    "aider": 22,
    "continue": 20,
    "supermaven": 16,
    "amazon-q": 16,
    "vercel": 22,
    "supabase": 20,
    "linear": 16,
    "notion": 18,
    "figma": 16,
    "raycast": 14,
    "warp": 12,
    "postman": 12,
    "mongodb": 12,
    "datadog": 10,
}

# Title / caption templates. ``{b}`` = a surface alias of the brand.
TITLE_TEMPLATES = [
    "How I ship 3x faster with {b}",
    "I replaced my whole workflow with {b} — here's what happened",
    "{b} vs my old setup: an honest 30-day review",
    "Building a SaaS in a weekend using {b}",
    "Why every indie hacker should try {b}",
    "{b} changed how I refactor legacy code",
    "My honest take on {b} after 6 months",
    "Stop writing boilerplate — let {b} do it",
    "The {b} feature nobody talks about",
    "I gave {b} my hardest bug. It crushed it.",
    "{b} for beginners: from zero to first app",
    "5 {b} tricks that doubled my output",
    "Is {b} worth it in 2026? Real numbers inside",
    "Pair programming with {b} all day",
    "How our team rolled out {b} across 12 engineers",
    "{b} saved me 10 hours this week",
    "From idea to deployed app with {b}",
    "The truth about coding with {b}",
]

CAPTION_TEMPLATES = [
    "In this one I walk through my full agentic workflow with {b}, from scaffolding to shipping.",
    "Sponsored by {b}. I show the exact prompts and the diffs they produced on a real codebase.",
    "I put {b} head to head against my old tooling on a messy production repo.",
    "Real project, real deadline — here's how {b} held up under pressure.",
    "Tried {b} on a legacy Python service and the refactor speed surprised me.",
    "Quick tutorial: setting up {b} and getting your first useful completion in minutes.",
    "Honest review. {b} is great at some things and rough at others — full breakdown inside.",
    "Used {b} to build and deploy a side project live on stream.",
    "My team has been using {b} for a quarter. Here are the numbers and the gotchas.",
    "Thanks to {b} for sponsoring. Everything you see is unedited, real output.",
]

SUMMARY_TEMPLATES = [
    "Captions cover setup, prompting strategy, and a before/after on build time.",
    "Subtitle highlights: pricing, the agent mode demo, and where it still struggles.",
    "Key moments: live refactor, test generation, and a candid cost breakdown.",
    "Covers onboarding a new repo, multi-file edits, and reviewer feedback.",
    "Walks through scaffolding, debugging a tricky bug, and deploying to production.",
    "Discusses developer audience reactions and the most-asked questions in comments.",
]

# Audience-flavor phrases mixed into text so the content reads like it comes from a
# real creator and so semantic search on developer themes lands well.
AUDIENCE_FLAVORS = [
    "Aimed at indie developers and solo founders.",
    "For backend engineers tired of boilerplate.",
    "Great for students learning to ship real apps.",
    "Targeted at startup teams evaluating AI coding tools.",
    "For full-stack devs who live in the terminal.",
    "Made for product engineers shipping fast.",
]


def pick_engagement(rng: random.Random, kol_engagement: float) -> float:
    """Per-post engagement, anchored on the creator's baseline with noise."""
    val = rng.gauss(kol_engagement, 1.2)
    return round(max(0.4, min(val, 18.0)), 1)


def pick_views(rng: random.Random, followers: int) -> int:
    """Views correlate with follower count, with a wide log-normal spread."""
    ratio = 10 ** rng.uniform(-0.7, 0.5)  # 0.2x .. ~3.2x of followers
    views = int(followers * ratio)
    return max(500, views)


def build_text(title: str, caption: str, summary: str, flavor: str) -> str:
    return f"{title}. {caption} {summary} {flavor}"


def weighted_brand_keys() -> tuple[list[str], list[int]]:
    keys = list(PRIMARY_BRAND_WEIGHTS.keys())
    weights = [PRIMARY_BRAND_WEIGHTS[k] for k in keys]
    return keys, weights


def select_kol_pool(kols: list[dict]) -> list[dict]:
    """Pool of KOLs eligible to post sponsored dev-tool content.

    Weighted toward tech / business / finance / education creators (the people who
    realistically promote AI coding tools), but every niche gets a small share so
    cross-niche / off-script pivots have data. Returns a flat list where eligible
    KOLs appear multiple times according to weight (so rng.choice favors them).
    """
    niche_weight = {
        "tech": 8,
        "business": 5,
        "finance": 3,
        "education": 3,
        "gaming": 2,
        "crypto": 2,
        "lifestyle": 1,
        "comedy": 1,
        "art": 1,
    }
    pool: list[dict] = []
    for k in kols:
        niche = k["metadata"]["niche"]
        w = niche_weight.get(niche, 1)
        pool.extend([k] * w)
    return pool


def _real_content_docs(rng: random.Random, kols: list[dict]) -> list[dict]:
    """Sponsored-content posts for the REAL KOLs, on the brands they actually
    promoted (from the spreadsheet Note column).

    This makes "who promoted Cursor / Claude / Copilot" surface the genuine
    creators (Corbin, Matthew Berman, Siraj Raval, ...). Brands are already
    lowercase-canon and handles already @-stripped (ingest_real normalizes both);
    kol_id links back to the kol-real-* docs in kols.json — relations stay
    consistent. NO pricing of any kind enters these docs.
    """
    try:
        from ingest_real import real_kols
    except ImportError:  # pragma: no cover
        from src.ingest_real import real_kols  # type: ignore

    # Map handle -> the kol-real-* doc so kol_id/platform/engagement are real.
    by_handle = {
        k["metadata"]["handle"]: k
        for k in kols
        if k["metadata"].get("source") == "real"
    }

    docs: list[dict] = []
    idx = 0
    for rk in real_kols():
        kdoc = by_handle.get(rk["handle"])
        if not kdoc:
            continue
        meta = kdoc["metadata"]
        followers = int(meta["followers"])
        kol_eng = float(meta["engagement_pct"])
        # Brands the creator genuinely promoted; if none parsed, give a generic
        # AI-coding deal so the creator still has content in the index.
        brands = [b for b in rk["brands"] if b in ALL_BRANDS] or ["cursor"]
        for brand in brands:
            # 1-2 posts per real (creator, brand) pair for retrieval density.
            for _ in range(rng.choice([1, 2])):
                idx += 1
                alias = rng.choice(ALL_BRANDS[brand])
                title = rng.choice(TITLE_TEMPLATES).format(b=alias)
                caption = rng.choice(CAPTION_TEMPLATES).format(b=alias)
                summary = rng.choice(SUMMARY_TEMPLATES)
                flavor = rng.choice(AUDIENCE_FLAVORS)
                text = build_text(title, caption, summary, flavor)
                views = pick_views(rng, followers)
                engagement = pick_engagement(rng, kol_eng)
                likes = int(views * engagement / 100 * rng.uniform(0.6, 0.95))
                comments = int(likes * rng.uniform(0.01, 0.06))
                year = 2026 if rng.random() < 0.6 else 2025
                month = rng.randint(1, 5) if year == 2026 else rng.randint(1, 12)
                day = rng.randint(1, 28)
                date = f"{year}-{month:02d}-{day:02d}"
                docs.append(
                    {
                        "id": f"content-real-{idx:04d}",
                        "text": text,
                        "metadata": {
                            "kol_id": kdoc["id"],
                            "kol_handle": rk["handle"],  # already no @
                            "brand": brand,  # already lowercase-canon
                            "platform": meta["platform"],
                            "title": title,
                            "views": str(views),
                            "likes": str(likes),
                            "comments": str(comments),
                            "engagement_pct": str(engagement),
                            "date": date,
                            "source": "real",
                        },
                    }
                )
    return docs


def main() -> None:
    rng = random.Random(SEED)
    kols = json.loads(KOLS_PATH.read_text(encoding="utf-8"))

    pool = select_kol_pool(kols)
    brand_keys, brand_weights = weighted_brand_keys()

    docs: list[dict] = []
    # Real KOL sponsored-content posts first (genuine brand collaborations).
    real_docs = _real_content_docs(rng, kols)
    docs.extend(real_docs)
    content_idx = 0

    for _ in range(N_BASE_POSTS):
        kol = rng.choice(pool)
        meta = kol["metadata"]
        kol_id = kol["id"]
        kol_handle = meta["handle"]  # already no @
        platform = meta["platform"]
        followers = int(meta["followers"])
        kol_eng = float(meta["engagement_pct"])

        # Primary brand for this post.
        primary = rng.choices(brand_keys, weights=brand_weights, k=1)[0]

        # Decide whether this post mentions multiple brands (e.g. a comparison
        # video). ~18% of posts are multi-brand and fan out to one doc per brand.
        brands = [primary]
        if rng.random() < 0.18:
            # Mix only within a sensible group: AI-coding posts add another
            # AI-coding brand; other-brand posts stay single.
            if primary in AI_CODING_BRANDS:
                extra_pool = [b for b in AI_CODING_BRANDS if b != primary]
                n_extra = rng.choice([1, 1, 2])
                extras = rng.sample(extra_pool, k=min(n_extra, len(extra_pool)))
                brands.extend(extras)

        # Shared post-level attributes (same underlying video/post).
        title_tpl = rng.choice(TITLE_TEMPLATES)
        caption_tpl = rng.choice(CAPTION_TEMPLATES)
        summary = rng.choice(SUMMARY_TEMPLATES)
        flavor = rng.choice(AUDIENCE_FLAVORS)
        views = pick_views(rng, followers)
        engagement = pick_engagement(rng, kol_eng)
        likes = int(views * engagement / 100 * rng.uniform(0.6, 0.95))
        comments = int(likes * rng.uniform(0.01, 0.06))
        # Dates spread across the past ~14 months.
        year = 2026 if rng.random() < 0.6 else 2025
        month = rng.randint(1, 5) if year == 2026 else rng.randint(1, 12)
        day = rng.randint(1, 28)
        date = f"{year}-{month:02d}-{day:02d}"

        for brand in brands:
            content_idx += 1
            alias = rng.choice(ALL_BRANDS[brand])
            title = title_tpl.format(b=alias)
            caption = caption_tpl.format(b=alias)
            text = build_text(title, caption, summary, flavor)

            docs.append(
                {
                    "id": f"content-{content_idx:04d}",
                    "text": text,
                    "metadata": {
                        "kol_id": kol_id,
                        "kol_handle": kol_handle,
                        "brand": brand,
                        "platform": platform,
                        "title": title,
                        "views": str(views),
                        "likes": str(likes),
                        "comments": str(comments),
                        "engagement_pct": str(engagement),
                        "date": date,
                    },
                }
            )

    OUT_PATH.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs),
        encoding="utf-8",
    )

    # Distribution summary.
    from collections import Counter

    brand_counts = Counter(d["metadata"]["brand"] for d in docs)
    plat_counts = Counter(d["metadata"]["platform"] for d in docs)
    kol_ids = {d["metadata"]["kol_id"] for d in docs}
    n_real = sum(1 for d in docs if d["metadata"].get("source") == "real")
    real_kol_ids = {
        d["metadata"]["kol_id"] for d in docs if d["metadata"].get("source") == "real"
    }
    print(
        f"Wrote {len(docs)} content docs to {OUT_PATH} "
        f"({n_real} real posts across {len(real_kol_ids)} real KOLs)"
    )
    print("Distinct KOLs referenced:", len(kol_ids))
    print("Platforms:", dict(plat_counts))
    print("Brand counts (desc):")
    for b, c in brand_counts.most_common():
        print(f"  {b}: {c}")
    print("\nSample:\n", json.dumps(docs[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
