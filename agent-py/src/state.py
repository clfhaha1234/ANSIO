"""Cross-turn session state for the ANSIO agent (PRD digest ⑤ + T9).

State lives in ``session.userdata`` (NOT Moss): the five gating slots plus the
wide candidate pool cached from ``find_similar_kols(top_k=80)`` so budget/weight
changes re-rank in pure Python with **no re-query** (PRD T1). Hang up -> the
session (and this object) is discarded, so nothing persists across calls.

KISS: a plain dataclass with small, pure methods. No Moss, no livekit imports —
importable offline for unit tests (PRD T10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The five required slots that gate "speak the recommendation out loud"
# (PRD digest ⑤). Mirrors {产品品类, 目标受众, 平台, 预算, 目标}.
REQUIRED_SLOTS = ("category", "audience", "platform", "budget", "goal")

# Recommendation is forced no later than this user turn (PRD gate condition 4).
MAX_TURNS_BEFORE_FORCE = 5


@dataclass
class AnsioState:
    """Per-session gating + candidate cache. Stored on ``session.userdata``."""

    slots: dict = field(default_factory=dict)
    # Wide pool from find_similar_kols(top_k=80), cached for Python re-rank.
    candidate_pool: list[dict] = field(default_factory=list)
    # Last ranked top-5 (list of stable ids) — for top-5 overlap stability.
    last_top5_ids: list[str] = field(default_factory=list)
    weights: dict | None = None
    turn_count: int = 0
    recommended: bool = False

    # -- slot management ---------------------------------------------------
    def update_slots(self, **kwargs) -> None:
        """Merge non-empty slot values; never clobber a set slot with None."""
        for k, v in kwargs.items():
            if v is None or v == "":
                continue
            self.slots[k] = v

    def slots_complete(self) -> bool:
        return all(self.slots.get(s) not in (None, "") for s in REQUIRED_SLOTS)

    def missing_slots(self) -> list[str]:
        return [s for s in REQUIRED_SLOTS if self.slots.get(s) in (None, "")]

    # -- candidate cache ---------------------------------------------------
    def set_pool(self, candidates: list[dict]) -> None:
        self.candidate_pool = list(candidates or [])

    def has_pool(self) -> bool:
        return bool(self.candidate_pool)

    # -- gating ------------------------------------------------------------
    def note_top5(self, ranked: list[dict]) -> int:
        """Record current top-5, return overlap (0..5) with the previous top-5."""
        ids = [_doc_id(c) for c in (ranked or [])][:5]
        prev = set(self.last_top5_ids)
        overlap = len(prev & set(ids)) if prev else 0
        self.last_top5_ids = ids
        return overlap

    def should_recommend(self, user_urged: bool, top5_overlap: int) -> bool:
        """Gate: speak the recommendation if ANY condition holds (PRD ⑤)."""
        if self.recommended:
            return True
        if user_urged:
            return True
        if self.slots_complete():
            return True
        if top5_overlap >= 4:  # candidates stable across two turns
            return True
        return self.turn_count >= MAX_TURNS_BEFORE_FORCE  # forced at turn limit


def _doc_id(c: dict) -> str:
    md = c.get("metadata", c) if isinstance(c, dict) else {}
    return str(md.get("handle") or md.get("name") or md.get("id") or id(c))
