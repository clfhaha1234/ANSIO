"""PRD §4 routing eval — does the LLM call the RIGHT tool for each utterance?

This validates the central PRD claim ("每一句话都触发一次真实检索") WITHOUT
implementing the product. The 5 PRD tools are *stubs* that only record the call
and return plausible canned data, so the only variable under test is the LLM's
function-call routing given the PRD's tool descriptions + system prompt.

Run (needs LIVEKIT_* for the Inference LLM; Moss not required — tools are stubbed):

    ENV_FILE=../.env.local uv run python tests/prd_routing_eval.py
    # or from the repo with creds:
    uv run python tests/prd_routing_eval.py

It drives a single multi-turn conversation through the PRD §4.1 main script plus
a few §4.2 off-script turns, and prints, per turn, the EXPECTED tool vs the tool
the LLM ACTUALLY called, with a final routing hit-rate.

NOTE: routing is probabilistic. Treat the hit-rate as a signal, not a gate. The
goal is to surface turns where the prompt/tool-descriptions mis-route, so the
implementer can fix descriptions before building the real retrieval.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import textwrap

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_FILE", ".env.local"))

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from livekit.agents import Agent, AgentSession, RunContext, function_tool

from llm_factory import build_llm, model_label, resolve_provider

# --- LLM selection (env-driven, via shared src/llm_factory.py) ----------------
# Provider is one of: minimax | claude | inference. Override per run with the
# EVAL_LLM_PROVIDER env var (falls back to LLM_PROVIDER, then "inference" here so
# the historical baseline is the default when nothing is set).
#
#   # MiniMax-M2 (OpenAI-compatible):
#   ENV_FILE=.env EVAL_LLM_PROVIDER=minimax uv run python tests/prd_routing_eval.py
#   # Claude via the local sub2api 9090 relay:
#   ENV_FILE=.env EVAL_LLM_PROVIDER=claude uv run python tests/prd_routing_eval.py
#   # gpt-4.1-mini via LiveKit Inference (baseline):
#   ENV_FILE=.env EVAL_LLM_PROVIDER=inference uv run python tests/prd_routing_eval.py
#
# Set TTFT_SAMPLES=N (default 5) to control how many TTFT samples are drawn.
EVAL_LLM_PROVIDER = resolve_provider(
    os.getenv("EVAL_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "inference"
)


def build_eval_llm():
    """Return an LLM for the eval via the shared factory, by EVAL_LLM_PROVIDER."""
    return build_llm(EVAL_LLM_PROVIDER)


# Human-readable label for the header / output, no secrets.
MODEL = model_label(EVAL_LLM_PROVIDER)

# Canned data so the multi-hop conversation can proceed coherently. The handles
# match nothing real — routing, not retrieval, is under test.
CANNED_COMPETITORS = (
    "Cursor (AI code editor, $400M funding); GitHub Copilot (Microsoft); "
    "Replit (cloud IDE, $250M); Codeium (free AI autocomplete)."
)
CANNED_PROMOTERS = (
    "buildwithsam — 3 Cursor posts, 1.6M total views; "
    "tldevtips — 2 posts, 740k views; aigrindexyz — 2 posts, 520k views."
)
CANNED_PROFILE = (
    "buildwithsam: 220k subs on YouTube, audience 60% developers / 25% founders, "
    "daily coding-workflow demos, 6.2% engagement, ~$1,800/video."
)
CANNED_SIMILAR = (
    "20 similar dev creators incl. tldevtips (sim .91), shipfastleo (.88), "
    "codewithana (.86), indiehacker_jo (.84)."
)
CANNED_PLAYBOOK = (
    "Case: an AI dev tool ran 5 micro dev-creators, $4k budget -> 2,200 trials, "
    "110 paid (source: internal_casebook_2024)."
)


class PRDAgent(Agent):
    """Agent wired with the 5 PRD tools as call-recording stubs."""

    def __init__(self) -> None:
        super().__init__(
            llm=build_eval_llm(),
            instructions=textwrap.dedent(
                """\
                You are ANSIO, a voice-based growth advisor for startup founders.
                A founder describes their product by voice; you find under-valued
                KOLs (creators) for their campaign by SEARCHING as you talk.

                Your loop is: Conversation -> Retrieval -> Reasoning -> Decision.
                Almost every founder utterance should trigger exactly one tool
                call. Pick the tool that fits the founder's CURRENT question:

                - The founder describes their product / category, or asks "who are
                  our competitors": call find_competitors.
                - The founder asks who promoted / partnered with / worked with a
                  specific brand (or asks what a specific @handle has promoted):
                  call find_kols_who_promoted.
                - You need one creator's audience / style / history before
                  comparing them: call get_kol_profile.
                - The founder asks for "more like them" / similar creators, or adds
                  a platform/niche constraint to the candidate set: call
                  find_similar_kols.
                - The founder pushes back, asks "why", asks whether it's worth it,
                  asks about ROI, or asks about content strategy/methodology: call
                  search_playbook (it returns grounded cases and playbooks; never
                  invent ROI or persuasion talking points).

                Do NOT call a tool only when the founder changes a number you
                already have (budget, weights) and just wants a re-rank, or makes
                pure small talk.

                Speak in one to three short, natural sentences (this is voice).
                Never read JSON, lists, or tool names aloud.
                """
            ),
        )
        self.calls: list[tuple[str, dict]] = []

    def _rec(self, name: str, **kw) -> None:
        self.calls.append((name, kw))

    @function_tool()
    async def find_competitors(
        self, context: RunContext, product_desc: str, top_k: int = 5
    ) -> str:
        """Find competitor products in the same category as the founder's product.

        Call this first, as soon as the founder describes what they build, to map
        the competitive landscape before looking at creators.

        Args:
            product_desc: Natural-language description of the founder's product.
        """
        self._rec("find_competitors", product_desc=product_desc, top_k=top_k)
        return CANNED_COMPETITORS

    @function_tool()
    async def find_kols_who_promoted(
        self, context: RunContext, brand: str, product_desc: str = ""
    ) -> str:
        """Find which creators have already promoted / partnered with a brand.

        Use when the founder asks who a competitor works with, or (reverse) what a
        specific creator handle has promoted.

        Args:
            brand: The brand name to look up (e.g. "cursor"). Lowercase it.
            product_desc: Optional context to rank the most relevant content first.
        """
        self._rec("find_kols_who_promoted", brand=brand, product_desc=product_desc)
        return CANNED_PROMOTERS

    @function_tool()
    async def get_kol_profile(self, context: RunContext, handle: str) -> str:
        """Pull one creator's full profile (audience, style, history) by handle.

        Use before comparing or recommending a specific creator.

        Args:
            handle: The creator handle, e.g. "buildwithsam" (no @).
        """
        self._rec("get_kol_profile", handle=handle)
        return CANNED_PROFILE

    @function_tool()
    async def find_similar_kols(
        self,
        context: RunContext,
        profile_text: str,
        niche: str = "",
        platform: str = "",
        top_k: int = 20,
    ) -> str:
        """Find more creators similar to a profile, optionally constrained.

        Use when the founder asks for "more like them" or adds a platform/niche
        constraint.

        Args:
            profile_text: Description of the kind of creator to find more of.
            niche: Optional category filter.
            platform: Optional platform filter (e.g. YouTube).
        """
        self._rec(
            "find_similar_kols",
            profile_text=profile_text,
            niche=niche,
            platform=platform,
            top_k=top_k,
        )
        return CANNED_SIMILAR

    @function_tool()
    async def search_playbook(
        self, context: RunContext, question: str, doc_type: str = ""
    ) -> str:
        """Search grounded playbooks, strategy docs, and historical case studies.

        Use for objections ("why bother?"), "why these creators?", content
        strategy, and ROI questions — so answers cite real cases, not guesses.

        Args:
            question: What to look up.
            doc_type: Optional one of qa / strategy / case.
        """
        self._rec("search_playbook", question=question, doc_type=doc_type)
        return CANNED_PLAYBOOK


# (utterance, expected_tool_or_None, note). None = expect NO tool call (re-rank/small talk).
SCRIPT = [
    ("We're building an AI coding tool and our growth has stalled.",
     "find_competitors", "Step 1: product described -> landscape"),
    ("Who are all of those competitors actually partnering with?",
     "find_kols_who_promoted", "Step 2: who promoted"),
    ("Cursor is way bigger than us — does benchmarking against them even make sense?",
     "search_playbook", "Step 2b: objection -> grounded"),
    ("Okay, break down who buildwithsam actually is.",
     "get_kol_profile", "Step 3: pull profile"),
    ("Are there others like them?",
     "find_similar_kols", "Step 4: expand similar"),
    ("Actually, only people on YouTube.",
     "find_similar_kols", "Step 4b off-script: add platform constraint"),
    ("Change the budget to five hundred dollars per video.",
     None, "Off-script: pure re-rank, NO retrieval"),
    ("Honestly is this even worth the spend?",
     "search_playbook", "Step 8: ROI -> case"),
    ("What has buildwithsam promoted before?",
     "find_kols_who_promoted", "Off-script: reverse lookup by handle"),
    ("Why these two creators specifically?",
     "search_playbook", "Off-script: justify -> strategy doc"),
]


async def sample_ttft(n: int) -> dict:
    """Measure time-to-first-token over `n` cold-ish calls for EVAL_LLM_PROVIDER.

    TTFT = wall time from `llm.chat(...)` to the first streamed chunk. We use a
    realistic system prompt + the 5 PRD tool schemas as context so the number
    reflects the demo's true first-token cost (tool-call prompts are heavier than
    a bare chat). Returns mean / p95 (ms) plus the raw samples. Failures are
    surfaced rather than swallowed so a dead relay is obvious.
    """
    import statistics
    import time

    from livekit.agents.llm import ChatContext

    llm = build_eval_llm()
    # Reuse the PRD agent's tools so the prompt weight matches the routing eval.
    tools = list(PRDAgent().tools) if hasattr(PRDAgent(), "tools") else None

    samples: list[float] = []
    errors: list[str] = []
    for _ in range(n):
        ctx = ChatContext.empty()
        ctx.add_message(
            role="system",
            content=(
                "You are ANSIO, a voice growth advisor. Reply in one short "
                "natural sentence; call a tool only when warranted."
            ),
        )
        ctx.add_message(
            role="user",
            content="We're building an AI coding tool and growth has stalled.",
        )
        t0 = time.perf_counter()
        stream = llm.chat(chat_ctx=ctx, tools=tools) if tools else llm.chat(chat_ctx=ctx)
        first: float | None = None
        try:
            async for _chunk in stream:
                if first is None:
                    first = (time.perf_counter() - t0) * 1000.0
                    break  # TTFT only — stop after the first token
        except Exception as exc:  # noqa: BLE001 — surface relay/auth failures
            errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
        if first is not None:
            samples.append(first)

    out = {"samples": samples, "errors": errors}
    if samples:
        ordered = sorted(samples)
        out["mean"] = statistics.mean(samples)
        # p95 via nearest-rank (small-n safe).
        idx = max(0, min(len(ordered) - 1, round(0.95 * len(ordered) + 0.5) - 1))
        out["p95"] = ordered[idx]
        out["min"] = ordered[0]
        out["max"] = ordered[-1]
    return out


async def main() -> None:
    print(f"PRD routing eval — provider={EVAL_LLM_PROVIDER} LLM {MODEL}\n" + "=" * 78)
    agent = PRDAgent()
    hits = 0
    scored = 0
    async with AgentSession() as session:
        await session.start(agent)
        for i, (utt, expected, note) in enumerate(SCRIPT, 1):
            before = len(agent.calls)
            await session.run(user_input=utt)
            fired = [c[0] for c in agent.calls[before:]]
            actual = fired[0] if fired else None

            scored += 1
            ok = actual == expected
            hits += int(ok)
            mark = "✓" if ok else "✗"
            exp = expected or "(no tool)"
            act = " + ".join(fired) if fired else "(no tool)"
            print(f"\n[{i}] {mark}  {note}")
            print(f'    user: "{utt}"')
            print(f"    expected: {exp:24s} actual: {act}")
            if fired:
                # show args of the first fired tool for inspection
                args = agent.calls[before][1]
                shown = {k: v for k, v in args.items() if v not in ("", 20, 5)}
                print(f"    args: {shown}")

    print("\n" + "=" * 78)
    print(f"Routing hit-rate: {hits}/{scored}")
    print("Note: routing is probabilistic — investigate ✗ rows; tune tool "
          "descriptions/prompt, not the test.")

    # --- TTFT sampling (n>=5 by default) --------------------------------------
    n = int(os.getenv("TTFT_SAMPLES", "5"))
    print("\n" + "=" * 78)
    print(f"TTFT sampling — provider={EVAL_LLM_PROVIDER} LLM {MODEL} (n={n})")
    stats = await sample_ttft(n)
    if stats.get("errors"):
        for e in stats["errors"][:3]:
            print(f"    ttft error: {e}")
    if stats.get("samples"):
        s = stats["samples"]
        print("    samples(ms): " + ", ".join(f"{x:.0f}" for x in s))
        print(
            f"    TTFT mean={stats['mean']:.0f}ms  p95={stats['p95']:.0f}ms  "
            f"min={stats['min']:.0f}ms  max={stats['max']:.0f}ms  ok={len(s)}/{n}"
        )
    else:
        print("    TTFT: no successful samples (provider unreachable?).")


if __name__ == "__main__":
    asyncio.run(main())
