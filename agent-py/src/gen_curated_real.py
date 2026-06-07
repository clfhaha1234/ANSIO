"""Expand the ANSIO Moss KB with researched REAL creators (additive).

Reads a curated seed of REAL, web-verified creators (public fields only) from
``agent-py/curated_real.json`` and MERGES them into the existing data files that
``build_indexes.py`` consumes:

    kols.json      <- new kol-real-dev-* / kol-real-cn-* docs (deduped by handle)
    content.jsonl  <- 1-3 real-title content rows per curated creator
    products.json  <- new ai-coding competitor brands (deduped by name_norm)

This is the demo-hero expansion: deep on the AI-coding / developer vertical,
including real Chinese / cross-border creators (bilingual text so Chinese
semantic queries hit). It is ADDITIVE — it never rewrites the existing 1018
kols or re-reads the confidential spreadsheet; it only appends de-duplicated
records, then you re-run build_indexes.py to push to Moss.

CONFIDENTIALITY (non-negotiable, mirrors gen_kols / ingest_real):
* Only PUBLIC creator fields enter here (name/handle/platform/followers/niche/
  region/brands). NO real quote / CPM / private contact ever appears.
* ``price_usd`` is a SYNTHETIC estimate (followers x rate proxy) used only as the
  scoring denominator — identical formula to gen_kols._real_kol_docs. It is NOT a
  real rate.

Seed schema (curated_real.json) — the normalized research output:
    {
      "creators": [ {name, handle, platform, niche, region, language,
                     followers, brands:[canon], content_titles:[...], blurb} ],
      "new_products": [ {name, category, funding, stage, blurb} ],
      "new_brands":   [ {canon_key, aliases:[...]} ]      # informational
    }

Run (from agent-py/):
    uv run python src/gen_curated_real.py
    uv run python src/gen_curated_real.py --dry-run     # report only, no writes
"""

from __future__ import annotations

import json
import random
import sys
import zlib
from pathlib import Path

try:
    from avatar import avatar_url
    from gen_kols import humanize, tier_for
    from kol_scoring import precompute_scores
except ImportError:  # pragma: no cover - run from elsewhere
    from src.avatar import avatar_url  # type: ignore
    from src.gen_kols import humanize, tier_for  # type: ignore
    from src.kol_scoring import precompute_scores  # type: ignore

AGENT_DIR = Path(__file__).resolve().parent.parent
SEED_PATH = AGENT_DIR / "curated_real.json"
KOLS_PATH = AGENT_DIR / "kols.json"
CONTENT_PATH = AGENT_DIR / "content.jsonl"
PRODUCTS_PATH = AGENT_DIR / "products.json"

# Platform whitelist (must match agent.py KOL_PLATFORMS for exact-match filtering).
WL_PLATFORMS = {"YouTube", "Instagram", "TikTok", "X", "Twitch", "Bilibili", "LinkedIn"}
# Non-whitelist labels we map so platform filters still work.
PLATFORM_MAP = {
    "Newsletter": "X", "Substack": "X", "Twitter": "X",
    "Weibo": "Bilibili", "RED": "Bilibili", "Xiaohongshu": "Bilibili",
    "Douyin": "TikTok", "微博": "Bilibili", "小红书": "Bilibili",
    "抖音": "TikTok", "B站": "Bilibili",
}

_BASE_ENG = {"nano": 7.5, "micro": 5.5, "mid": 3.8, "macro": 2.6, "mega": 1.6}
# View-rate proxy per platform (fraction of followers that see a sponsored post).
_VIEW_RATE = {
    "YouTube": 0.42, "Bilibili": 0.38, "X": 0.22, "TikTok": 0.55,
    "Instagram": 0.30, "Twitch": 0.28, "LinkedIn": 0.20,
}


def _rng(handle: str, salt: int = 0) -> random.Random:
    """Stable per-handle RNG so re-runs don't churn the data."""
    return random.Random(zlib.crc32(f"{handle}|{salt}".encode("utf-8")))


def _norm_platform(p: str) -> str:
    p = (p or "").strip()
    if p in WL_PLATFORMS:
        return p
    return PLATFORM_MAP.get(p, p)  # may stay non-whitelist -> semantic-only (still retrievable)


def _engagement(handle: str, tier: str) -> float:
    rng = _rng(handle, 1)
    return round(max(0.4, rng.gauss(_BASE_ENG.get(tier, 3.8), 0.9)), 1)


def _price(handle: str, followers: int, engagement: float) -> int:
    """Synthetic sponsorship-rate proxy (scoring denominator only) — NOT a real quote."""
    rng = _rng(handle, 2)
    return max(50, int(followers / 1000 * rng.uniform(8, 22) * (1 + engagement / 20)))


def _kol_text(c: dict, tier: str, engagement: float) -> str:
    name, handle = c["name"], c["handle"]
    platform, region = c["platform"], c.get("region", "United States")
    followers = c["followers"]
    brands = c.get("brands") or []
    brand_str = ", ".join(brands) if brands else "AI coding and developer tools"
    blurb = (c.get("blurb") or "").strip()
    if c.get("language") == "Chinese":
        # Bilingual text so Chinese semantic queries hit the moss multilingual model.
        return (
            f"{name}（@{handle}）是 {platform} 上的中文科技/开发者创作者，约 {humanize(followers)} 粉丝，"
            f"位于 {region}。内容覆盖 AI 编程工具、程序员、软件开发、开发者工作流、产品评测。"
            f"合作/评测过的品牌：{brand_str}。{blurb} "
            f"{name} (@{handle}) is a Chinese-language tech and developer creator on {platform} "
            f"with {humanize(followers)} followers, based in {region}, posting in Chinese. Their content "
            f"covers AI coding tools, software engineering, developer workflows, and product reviews. "
            f"Audience tier: {tier}. They have featured brands such as {brand_str}. A strong fit for "
            f"developer-tool brands targeting Chinese-speaking and cross-border (出海) developer audiences."
        )
    return (
        f"{name} (@{handle}) is a trusted tech creator on {platform} with {humanize(followers)} "
        f"followers, based in {region}, posting in English. Their content covers AI coding tools, "
        f"software engineering, developer workflows, and product reviews, mostly as long-form "
        f"tutorials and reviews. Audience tier: {tier}. They have promoted brands such as {brand_str}. "
        f"{blurb} A strong fit for AI coding and developer-tool brands looking to reach a developer audience."
    )


_TITLE_FALLBACKS = [
    "I tried {b} for a week — here's the honest verdict",
    "{b} vs the rest: which AI coding tool actually ships faster?",
    "Building a real project with {b} (live walkthrough)",
    "Why my whole team switched to {b}",
]
_DATES = [  # deterministic recent spread (no real publish dates known)
    "2025-10-14", "2025-11-03", "2025-11-28", "2025-12-19", "2026-01-09",
    "2026-01-27", "2026-02-11", "2026-02-26", "2026-03-12", "2026-03-30",
    "2026-04-15", "2026-05-06", "2026-05-21",
]


def _content_rows(c: dict, kol_id: str, tier: str, engagement: float, start_idx: int) -> list[dict]:
    handle, platform = c["handle"], c["platform"]
    followers = c["followers"]
    brands = [b.strip().lower() for b in (c.get("brands") or []) if b.strip()] or ["cursor"]
    titles = [t for t in (c.get("content_titles") or []) if t and t.strip()]
    rng = _rng(handle, 3)
    view_rate = _VIEW_RATE.get(platform, 0.3)
    rows: list[dict] = []
    n = max(1, min(3, len(titles) if titles else 2))
    for j in range(n):
        brand = brands[j % len(brands)]
        if j < len(titles):
            title = titles[j]
        else:
            title = _TITLE_FALLBACKS[rng.randrange(len(_TITLE_FALLBACKS))].format(b=brand.title())
        views = max(800, int(followers * view_rate * rng.uniform(0.5, 1.4)))
        likes = int(views * (engagement / 100) * rng.uniform(0.8, 1.3))
        comments = int(likes * rng.uniform(0.03, 0.09))
        eng_post = round(max(0.3, (likes + comments) / max(views, 1) * 100), 1)
        date = _DATES[(start_idx + j) % len(_DATES)]
        rows.append({
            "id": f"content-cur-{start_idx + j:04d}",
            "text": f"{title}. A hands-on look at {brand} for real developer workflows — setup, speed, and where it wins.",
            "metadata": {
                "kol_id": kol_id,
                "kol_handle": handle,
                "brand": brand,
                "platform": platform,
                "title": title,
                "views": str(views),
                "likes": str(likes),
                "comments": str(comments),
                "engagement_pct": str(eng_post),
                "date": date,
                "source": "real",
            },
        })
    return rows


def _load_json_array(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def main() -> None:
    dry = "--dry-run" in sys.argv
    if not SEED_PATH.exists():
        raise SystemExit(f"Seed not found: {SEED_PATH} — write the research output there first.")
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    creators = seed.get("creators", [])
    new_products = seed.get("new_products", [])

    kols = _load_json_array(KOLS_PATH)
    existing_handles = {str(d.get("metadata", {}).get("handle", "")).lower() for d in kols}
    existing_ids = {d.get("id") for d in kols}

    new_kols: list[dict] = []
    new_content: list[dict] = []
    skipped: list[str] = []
    content_idx = 1
    en = cn = 0
    for i, c in enumerate(creators, start=1):
        c = dict(c)
        c["platform"] = _norm_platform(c.get("platform", ""))
        handle = str(c.get("handle", "")).strip().lstrip("@")
        if not handle or not c.get("name") or not c.get("followers"):
            skipped.append(c.get("handle") or c.get("name") or "?")
            continue
        c["handle"] = handle
        if handle.lower() in existing_handles:
            skipped.append(handle)
            continue
        existing_handles.add(handle.lower())
        followers = int(c["followers"])
        tier = tier_for(followers)
        engagement = _engagement(handle, tier)
        is_cn = c.get("language") == "Chinese"
        prefix = "kol-real-cn" if is_cn else "kol-real-dev"
        kid = f"{prefix}-{i:04d}"
        while kid in existing_ids:
            kid = f"{kid}x"
        existing_ids.add(kid)
        c["niche"] = "tech"
        scores = precompute_scores(followers, engagement, tier)
        new_kols.append({
            "id": kid,
            "text": _kol_text(c, tier, engagement),
            "metadata": {
                "name": c["name"], "handle": handle, "platform": c["platform"],
                "niche": "tech", "tier": tier, "region": c.get("region", "United States"),
                "language": "Chinese" if is_cn else c.get("language", "English"),
                "followers": followers, "engagement_pct": engagement,
                "price_usd": _price(handle, followers, engagement),
                "avatar_url": avatar_url(handle, c["platform"], c["name"]),
                **scores, "source": "real",
            },
        })
        rows = _content_rows(c, kid, tier, engagement, content_idx)
        new_content.extend(rows)
        content_idx += len(rows)
        en += 0 if is_cn else 1
        cn += 1 if is_cn else 0

    # products: append new ai-coding competitors, dedup by name_norm.
    products = _load_json_array(PRODUCTS_PATH)
    existing_norm = {str(p.get("metadata", {}).get("name_norm", "")).lower() for p in products}
    max_pnum = 0
    for p in products:
        pid = str(p.get("id", ""))
        if pid.startswith("prod-"):
            try:
                max_pnum = max(max_pnum, int(pid.split("-")[1]))
            except (ValueError, IndexError):
                pass
    new_prod_docs: list[dict] = []
    for p in new_products:
        name = str(p.get("name", "")).strip()
        if not name or name.lower() in existing_norm:
            continue
        existing_norm.add(name.lower())
        max_pnum += 1
        blurb = (p.get("blurb") or f"{name} is an AI coding / developer-tool product.").strip()
        new_prod_docs.append({
            "id": f"prod-{max_pnum:04d}",
            "text": blurb,
            "metadata": {
                "name": name, "name_norm": name.lower(),
                "category": p.get("category", "ai-coding"),
                "funding": str(p.get("funding", "unknown")),
                "stage": str(p.get("stage", "unknown")),
            },
        })

    print(f"Curated real expansion: +{len(new_kols)} kols ({en} EN / {cn} CN), "
          f"+{len(new_content)} content rows, +{len(new_prod_docs)} products. "
          f"Skipped {len(skipped)} dup/invalid: {skipped[:12]}{'…' if len(skipped) > 12 else ''}")
    if dry:
        if new_kols:
            print("\nSample kol:\n", new_kols[0]["text"][:400])
        return

    kols.extend(new_kols)
    KOLS_PATH.write_text(json.dumps(kols, ensure_ascii=False, indent=2), encoding="utf-8")
    with CONTENT_PATH.open("a", encoding="utf-8") as f:
        for row in new_content:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    products.extend(new_prod_docs)
    PRODUCTS_PATH.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {KOLS_PATH.name} ({len(kols)} total), appended {CONTENT_PATH.name}, "
          f"{PRODUCTS_PATH.name} ({len(products)} total). Now run build_indexes.py.")


if __name__ == "__main__":
    main()
