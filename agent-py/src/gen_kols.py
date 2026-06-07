"""Generate a synthetic but realistic 1000-KOL dataset for the KOL-finder agent.

Writes ``agent-py/kols.json`` — a list of documents shaped for Moss:

    { "id": "kol-0001", "text": "<rich natural-language profile>", "metadata": {...} }

The ``text`` is written as a natural-language profile (with niche synonyms baked
in) so Moss semantic search surfaces the right creators from conversational
queries. ``metadata`` carries clean categorical facets (platform, niche, tier,
region, language) for exact-match filtering, plus display fields.

Deterministic: seeded RNG so re-runs produce the same dataset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

try:
    from avatar import avatar_url
    from kol_scoring import precompute_scores
except ImportError:  # pragma: no cover - run from elsewhere
    from src.avatar import avatar_url  # type: ignore
    from src.kol_scoring import precompute_scores  # type: ignore

OUT_PATH = Path(__file__).resolve().parent.parent / "kols.json"
SEED = 42
N = 1000

PLATFORMS = ["YouTube", "Instagram", "TikTok", "X", "Twitch", "Bilibili", "LinkedIn"]
# Per-platform weighting so the mix feels real (IG/TikTok/YouTube dominate).
PLATFORM_WEIGHTS = [0.24, 0.26, 0.24, 0.1, 0.06, 0.06, 0.04]

# niche -> keyword bag woven into the profile text so semantic queries hit.
NICHES = {
    "tech": [
        "technology",
        "gadgets",
        "software",
        "AI",
        "coding",
        "consumer electronics",
        "product reviews",
    ],
    "gaming": [
        "video games",
        "esports",
        "live streaming",
        "game reviews",
        "let's plays",
        "FPS",
        "RPG",
    ],
    "beauty": [
        "makeup",
        "skincare",
        "cosmetics",
        "beauty tutorials",
        "GRWM",
        "product hauls",
    ],
    "fashion": [
        "style",
        "outfits",
        "OOTD",
        "streetwear",
        "luxury fashion",
        "thrift hauls",
    ],
    "fitness": [
        "workout",
        "gym",
        "bodybuilding",
        "home workouts",
        "nutrition",
        "weight loss",
        "wellness",
    ],
    "food": [
        "cooking",
        "recipes",
        "restaurant reviews",
        "baking",
        "mukbang",
        "food challenges",
    ],
    "finance": [
        "personal finance",
        "investing",
        "stocks",
        "budgeting",
        "side hustles",
        "FIRE",
        "money tips",
    ],
    "travel": [
        "travel vlogs",
        "destinations",
        "backpacking",
        "luxury travel",
        "travel tips",
        "digital nomad",
    ],
    "education": [
        "study tips",
        "science explainers",
        "language learning",
        "tutorials",
        "edutainment",
    ],
    "music": [
        "singing",
        "covers",
        "music production",
        "instrument tutorials",
        "songwriting",
    ],
    "comedy": ["sketches", "stand-up", "memes", "parody", "relatable humor"],
    "lifestyle": [
        "daily vlogs",
        "productivity",
        "minimalism",
        "self-improvement",
        "routines",
    ],
    "parenting": [
        "motherhood",
        "family vlogs",
        "kids activities",
        "parenting tips",
        "pregnancy",
    ],
    "automotive": [
        "cars",
        "car reviews",
        "detailing",
        "EVs",
        "motorsport",
        "garage builds",
    ],
    "business": [
        "entrepreneurship",
        "startups",
        "marketing",
        "SaaS",
        "B2B",
        "career growth",
    ],
    "crypto": ["cryptocurrency", "bitcoin", "web3", "DeFi", "NFTs", "blockchain"],
    "art": ["digital art", "illustration", "painting", "design", "animation", "crafts"],
    "sustainability": [
        "eco-friendly",
        "zero waste",
        "climate",
        "sustainable living",
        "green tech",
    ],
    "home": ["interior design", "home decor", "DIY", "organization", "renovation"],
    "pets": ["dogs", "cats", "pet care", "animal rescue", "training tips"],
}

# Region -> (likely languages, weight)
REGIONS = {
    "United States": (["English"], 0.30),
    "United Kingdom": (["English"], 0.08),
    "Canada": (["English"], 0.05),
    "Australia": (["English"], 0.04),
    "India": (["English", "Hindi"], 0.10),
    "Germany": (["German", "English"], 0.05),
    "France": (["French"], 0.04),
    "Spain": (["Spanish"], 0.04),
    "Brazil": (["Portuguese"], 0.06),
    "Mexico": (["Spanish"], 0.04),
    "Japan": (["Japanese"], 0.05),
    "South Korea": (["Korean"], 0.04),
    "China": (["Chinese"], 0.07),
    "Singapore": (["English", "Chinese"], 0.04),
}

FIRST = [
    "Alex",
    "Jordan",
    "Taylor",
    "Morgan",
    "Riley",
    "Casey",
    "Jamie",
    "Avery",
    "Quinn",
    "Sky",
    "Maya",
    "Liam",
    "Noah",
    "Emma",
    "Olivia",
    "Sophia",
    "Lucas",
    "Mia",
    "Ethan",
    "Zoe",
    "Aria",
    "Kai",
    "Leo",
    "Nina",
    "Ravi",
    "Priya",
    "Chen",
    "Yuki",
    "Hana",
    "Diego",
    "Lena",
    "Marco",
    "Sofia",
    "Omar",
    "Aisha",
    "Mateo",
    "Ana",
    "Hugo",
    "Ivy",
    "Theo",
]
LAST = [
    "Reed",
    "Cruz",
    "Park",
    "Lee",
    "Kim",
    "Nguyen",
    "Patel",
    "Khan",
    "Silva",
    "Costa",
    "Müller",
    "Dubois",
    "Garcia",
    "Rossi",
    "Tanaka",
    "Sato",
    "Wang",
    "Li",
    "Chen",
    "Zhang",
    "Brooks",
    "Hayes",
    "Ford",
    "Wells",
    "Bennett",
    "Carter",
    "Foster",
    "Grant",
    "Hale",
    "James",
    "Okafor",
    "Mensah",
    "Ahmed",
    "Haidar",
    "Novak",
    "Ivanov",
    "Santos",
    "Mendoza",
    "Flores",
    "Reyes",
]

ADJ = [
    "the go-to",
    "a rising",
    "a trusted",
    "a top",
    "a fast-growing",
    "an established",
    "a niche",
    "a beloved",
]
FORMATS = [
    "short-form videos",
    "long-form tutorials",
    "live streams",
    "daily stories",
    "weekly deep-dives",
    "reels and carousels",
    "vlogs",
    "Q&A sessions",
]


def tier_for(followers: int) -> str:
    if followers < 10_000:
        return "nano"
    if followers < 100_000:
        return "micro"
    if followers < 500_000:
        return "mid"
    if followers < 1_000_000:
        return "macro"
    return "mega"


def weighted_choice(rng, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def humanize(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _real_kol_docs(used_handles: set[str]) -> list[dict]:
    """Build Moss docs for the genuine AI-coding creators from the spreadsheet.

    Public fields only (name/handle/followers/niche/region/tier) — NO pricing.
    A synthetic per-post engagement is assigned (the spreadsheet doesn't carry
    it), but the real price/CPM never cross over. These docs are prepended so the
    demo's hero pool is real people; their handles are reserved so the synthetic
    loop never collides.
    """
    try:
        from ingest_real import real_kols
    except ImportError:  # pragma: no cover - run from elsewhere
        from src.ingest_real import real_kols  # type: ignore

    rng = random.Random(SEED + 7)  # separate stream so synthetic block stays stable
    docs: list[dict] = []
    for i, k in enumerate(real_kols(), start=1):
        handle = k["handle"]
        used_handles.add(handle)
        tier = k["tier"]
        followers = k["followers"]
        base_eng = {"nano": 7.5, "micro": 5.5, "mid": 3.8, "macro": 2.6, "mega": 1.6}[
            tier
        ]
        engagement = round(max(0.4, rng.gauss(base_eng, 0.9)), 1)
        # Synthetic sponsorship rate for scoring's price_norm denominator only;
        # this is NOT the real quote (which never leaves the aggregator).
        price = max(
            50, int(followers / 1000 * rng.uniform(8, 22) * (1 + engagement / 20))
        )
        brands = (
            ", ".join(k["brands"]) if k["brands"] else "AI coding and developer tools"
        )
        text = (
            f"{k['name']} (@{handle}) is a trusted {k['niche']} creator on "
            f"{k['platform']} with {humanize(followers)} followers, based in "
            f"{k['region']}, posting in {k['language']}. Their content covers "
            f"AI coding tools, software engineering, developer workflows, and "
            f"product reviews, mostly as long-form tutorials and reviews. "
            f"Audience tier: {tier}. They have promoted brands such as {brands}. "
            f"A strong fit for AI coding and developer-tool brands looking to reach "
            f"an English-speaking developer audience."
        )
        scores = precompute_scores(followers, engagement, tier)
        docs.append(
            {
                "id": f"kol-real-{i:04d}",
                "text": text,
                "metadata": {
                    "name": k["name"],
                    "handle": handle,
                    "platform": k["platform"],
                    "niche": k["niche"],
                    "tier": tier,
                    "region": k["region"],
                    "language": k["language"],
                    "followers": followers,
                    "engagement_pct": engagement,
                    "price_usd": price,
                    # PUBLIC avatar (real face via unavatar) + precomputed signals.
                    "avatar_url": k.get("avatar_url")
                    or avatar_url(handle, k["platform"], k["name"]),
                    **scores,
                    "source": "real",
                },
            }
        )
    return docs


def main() -> None:
    rng = random.Random(SEED)
    niche_keys = list(NICHES.keys())
    region_keys = list(REGIONS.keys())
    region_weights = [REGIONS[r][1] for r in region_keys]

    docs = []
    used_handles: set[str] = set()

    # Inject the real AI-coding KOLs first (demo main-character pool).
    real_docs = _real_kol_docs(used_handles)
    docs.extend(real_docs)

    for i in range(1, N + 1):
        first = rng.choice(FIRST)
        last = rng.choice(LAST)
        niche = rng.choice(niche_keys)
        platform = weighted_choice(rng, PLATFORMS, PLATFORM_WEIGHTS)
        region = weighted_choice(rng, region_keys, region_weights)
        language = rng.choice(REGIONS[region][0])

        # Followers: log-uniform from 5k to 18M, so the long tail is realistic.
        followers = int(10 ** rng.uniform(3.7, 7.26))
        followers = max(5_000, min(followers, 18_000_000))
        tier = tier_for(followers)

        # Engagement inversely correlates with size (smaller = more engaged).
        base_eng = {"nano": 7.5, "micro": 5.5, "mid": 3.8, "macro": 2.6, "mega": 1.6}[
            tier
        ]
        engagement = round(max(0.4, rng.gauss(base_eng, 0.9)), 1)

        # Price scales with reach and engagement.
        price = int(followers / 1000 * rng.uniform(8, 22) * (1 + engagement / 20))
        price = max(50, price)

        # Build a unique handle.
        base_handle = f"{first}{last}".lower()
        handle = base_handle
        suffix = 0
        while handle in used_handles:
            suffix += 1
            handle = f"{base_handle}{suffix}"
        used_handles.add(handle)

        kws = rng.sample(NICHES[niche], k=min(4, len(NICHES[niche])))
        name = f"{first} {last}"
        fmt = rng.choice(FORMATS)
        adj = rng.choice(ADJ)

        text = (
            f"{name} (@{handle}) is {adj} {niche} creator on {platform} with "
            f"{humanize(followers)} followers, based in {region}, posting in {language}. "
            f"Their content covers {', '.join(kws)}, mostly as {fmt}. "
            f"Average engagement rate {engagement}%. Audience tier: {tier}. "
            f"Typical sponsorship rate around ${price:,} per post. "
            f"A strong fit for brands in {niche} and adjacent categories "
            f"looking to reach a {language}-speaking audience in {region}."
        )

        scores = precompute_scores(followers, engagement, tier)
        docs.append(
            {
                "id": f"kol-{i:04d}",
                "text": text,
                "metadata": {
                    "name": name,
                    "handle": handle,
                    "platform": platform,
                    "niche": niche,
                    "tier": tier,
                    "region": region,
                    "language": language,
                    "followers": followers,
                    "engagement_pct": engagement,
                    "price_usd": price,
                    # PUBLIC avatar (initials/proxy by handle) + precomputed signals.
                    "avatar_url": avatar_url(handle, platform, name),
                    **scores,
                },
            }
        )

    OUT_PATH.write_text(
        json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Quick distribution summary.
    from collections import Counter

    plat = Counter(d["metadata"]["platform"] for d in docs)
    nich = Counter(d["metadata"]["niche"] for d in docs)
    tiers = Counter(d["metadata"]["tier"] for d in docs)
    n_real = sum(1 for d in docs if d["metadata"].get("source") == "real")
    print(
        f"Wrote {len(docs)} KOLs to {OUT_PATH}  ({n_real} real + {len(docs) - n_real} synthetic)"
    )
    print("Platforms:", dict(plat))
    print("Tiers:", dict(tiers))
    print("Niches:", dict(sorted(nich.items())))
    print("\nSample (real):\n", docs[0]["text"])


if __name__ == "__main__":
    main()
