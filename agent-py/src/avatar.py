"""Deterministic, public-only avatar URL derivation for KOL docs (BUILD-2).

Avatars are derived 100% offline from a creator's PUBLIC platform handle — no
scraping, no API keys, no network at build time. The browser fetches the image
directly when it renders the evidence card.

Strategy (in priority order, all public):
1. ``unavatar.io/{platform}/{handle}`` — proxies the real platform avatar
   (YouTube channel, GitHub, etc.). Verified live: YouTube @handles + channel
   IDs resolve 200. This is the REAL face of real creators.
2. ``ui-avatars.com`` initials avatar — a deterministic, always-200 fallback the
   frontend can swap in via ``onerror`` if the proxy 404s (e.g. a synthetic
   handle that no real account backs).

CONFIDENTIALITY: an avatar URL is public profile data only. No pricing, no
private contact info ever touches this module.
"""

from __future__ import annotations

import re
from urllib.parse import quote

# Map our display platform labels onto unavatar provider slugs.
# unavatar supports: youtube, twitter/x, github, instagram, telegram, ...
_PLATFORM_PROVIDER = {
    "youtube": "youtube",
    "x": "twitter",
    "twitter": "twitter",
    "instagram": "instagram",
    "tiktok": "tiktok",
    "twitch": "twitch",
    "github": "github",
    "linkedin": "github",  # no LinkedIn provider on unavatar; degrade to initials-friendly
    "bilibili": "github",  # ditto — initials fallback will carry these
}

# Brand-neutral palette per platform for the initials fallback (hex w/o '#').
_PLATFORM_BG = {
    "youtube": "c0392b",
    "instagram": "c13584",
    "tiktok": "1f6f78",
    "x": "2b2b2b",
    "twitter": "2b2b2b",
    "twitch": "7d3cdb",
    "bilibili": "2d7fb8",
    "linkedin": "2d6cb8",
    "default": "3a4a63",
}


def _initials_url(name: str, platform: str) -> str:
    """Deterministic ui-avatars initials URL (always 200) — the safety net."""
    bg = _PLATFORM_BG.get(platform.lower(), _PLATFORM_BG["default"])
    safe_name = quote((name or "KOL").strip())
    return (
        f"https://ui-avatars.com/api/?name={safe_name}"
        f"&size=128&background={bg}&color=fff&bold=true&format=png"
    )


def avatar_url(handle: str, platform: str, name: str = "") -> str:
    """Return a real public avatar URL for this creator.

    Prefers the live platform avatar via unavatar (the real face); falls back to
    a deterministic initials avatar for platforms/handles unavatar can't proxy.
    The URL is stable across runs (deterministic) so index rebuilds don't churn.
    """
    plat = (platform or "").strip().lower()
    h = re.sub(r"^@+", "", (handle or "").strip())
    provider = _PLATFORM_PROVIDER.get(plat)
    if provider and h:
        # unavatar lets us declare a fallback so a 404 still yields an image.
        fallback = _initials_url(name or h, plat)
        return f"https://unavatar.io/{provider}/{quote(h)}?fallback={quote(fallback, safe='')}"
    # No proxy for this platform (linkedin/bilibili w/o github) -> initials.
    return _initials_url(name or h, plat)
