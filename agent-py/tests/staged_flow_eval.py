"""Staged consultation-flow eval — do cards reveal IN ORDER like the BEATS arc?

Drives the REAL production system prompt (``agent._INSTRUCTIONS``: the 5-step
discovery arc) through a scripted founder conversation against the live brain
(EVAL_LLM_PROVIDER, default inference/gpt-4.1-mini). The 5 ANSIO tools are
stubs that record call order, so the only variable is the LLM's staging
discipline. Asserts the reveal precedence the user demo requires:

  find_competitors -> find_kols_who_promoted -> find_similar_kols
      -> (ONLY THEN) recommend_kols

Run:  ENV_FILE=.env uv run python tests/staged_flow_eval.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_FILE", ".env"))
load_dotenv(".env.local")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from livekit.agents import Agent, AgentSession, RunContext, function_tool  # noqa: E402

from agent import _INSTRUCTIONS  # noqa: E402 — the REAL staged-flow prompt
from llm_factory import build_llm, model_label, resolve_provider  # noqa: E402

PROVIDER = resolve_provider(
    os.getenv("EVAL_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "inference"
)

# The founder arc (mirrors the standalone BEATS user lines).
SCRIPT = [
    "Hi! I'm building an AI coding assistant for developers and indie "
    "hackers, and I want to grow fast.",
    "Whoa, is that Cursor up top? They're massive — can we really learn "
    "from them?",
    "I knew it! Okay — so who did they actually work with?",
    "These look great. How do we pick ours?",
    "That's the good stuff. Give me the final shortlist!",
]

STAGE_ORDER = [
    "find_competitors",
    "find_kols_who_promoted",
    "find_similar_kols",
    "recommend_kols",
]


class StagedAgent(Agent):
    """Production instructions + recording stub tools (no Moss, no network)."""

    def __init__(self) -> None:
        super().__init__(llm=build_llm(PROVIDER), instructions=_INSTRUCTIONS)
        self.calls: list[str] = []

    def _rec(self, name: str) -> None:
        self.calls.append(name)

    @function_tool()
    async def find_competitors(
        self, context: RunContext, product_desc: str, top_k: int = 5
    ) -> str:
        """Find competing products/brands in the user's space."""
        self._rec("find_competitors")
        return (
            "Cursor (AI code editor, $60M), Codeium ($65M), Replit, GitHub "
            "Copilot — Cursor is the closest trajectory."
        )

    @function_tool()
    async def find_kols_who_promoted(
        self,
        context: RunContext,
        brand: str = "",
        kol_handle: str = "",
        product_desc: str = "",
    ) -> str:
        """Find sponsorship history: who promoted BRAND / what @HANDLE promoted."""
        self._rec("find_kols_who_promoted")
        return (
            "Cursor grew via creators, not ads: theobennett1 (3 posts, 410k "
            "views), mikeynocode (2 posts, 300k views) on YouTube and X."
        )

    @function_tool()
    async def get_kol_profile(self, context: RunContext, handle: str) -> str:
        """Pull one creator's full profile by handle."""
        self._rec("get_kol_profile")
        return "Theo Bennett: tech YouTube, 41,832 followers, 6.1% engagement."

    @function_tool()
    async def find_similar_kols(
        self,
        context: RunContext,
        profile_text: str,
        niche: str = "",
        platform: str = "",
        top_k: int = 80,
    ) -> str:
        """Find undervalued creators similar to a brief (wide recall)."""
        self._rec("find_similar_kols")
        return (
            "Found 80 similar creators. Top: Theo Bennett, Mikey No Code, "
            "Priya Grant — same developer audience, far lower cost."
        )

    @function_tool()
    async def recommend_kols(
        self,
        context: RunContext,
        product_desc: str,
        niche: str = "",
        platform: str = "",
        budget: float | None = None,
    ) -> str:
        """Run the full recall chain and return the final ranked shortlist."""
        self._rec("recommend_kols")
        return (
            "Top picks: Theo Bennett (41,832 followers, est. $900), Mikey No "
            "Code (85,600, est. $1,400), Priya Grant (11,265, est. $250)."
        )

    @function_tool()
    async def search_playbook(
        self, context: RunContext, question: str, doc_type: str = ""
    ) -> str:
        """Search the ANSIO methodology playbook."""
        self._rec("search_playbook")
        return "Playbook: start with two or three micro creators to validate."


def first_call_order(calls: list[str]) -> list[str]:
    seen: list[str] = []
    for c in calls:
        if c in STAGE_ORDER and c not in seen:
            seen.append(c)
    return seen


async def main() -> None:
    print(f"Staged-flow eval — provider={PROVIDER} ({model_label(PROVIDER)})")
    print("=" * 72)
    agent = StagedAgent()
    per_turn: list[list[str]] = []
    async with AgentSession() as session:
        await session.start(agent)
        for i, utt in enumerate(SCRIPT, 1):
            before = len(agent.calls)
            await session.run(user_input=utt)
            fired = agent.calls[before:]
            per_turn.append(fired)
            print(f"T{i}: {utt[:58]!r:60} -> {fired or '(no tool)'}")

    order = first_call_order(agent.calls)
    print("\nfirst-call order:", order)

    failures: list[str] = []
    # A) recommend never before the three discovery stages.
    if "recommend_kols" in agent.calls:
        ri = agent.calls.index("recommend_kols")
        for pre in STAGE_ORDER[:3]:
            if pre not in agent.calls[:ri]:
                failures.append(f"recommend_kols fired before {pre}")
    else:
        failures.append("recommend_kols never fired (T5 demanded the list)")
    # B) turn 1 must not recommend.
    if "recommend_kols" in per_turn[0]:
        failures.append("turn 1 jumped straight to recommend_kols")
    # C) discovery stages appear in arc order.
    stage_only = [c for c in order if c in STAGE_ORDER]
    if stage_only != [s for s in STAGE_ORDER if s in stage_only]:
        failures.append(f"stage order violated: {stage_only}")

    print("\nVERDICT:", "PASS — cards reveal in the required order" if not failures
          else "FAIL — " + "; ".join(failures))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    asyncio.run(main())
