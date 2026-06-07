"""ANSIO end-to-end perceived-latency harness — "user input -> first byte".

This complements ``tests/latency_bench.py`` (which times the Moss-only recall
chain and MiniMax-only TTFT in isolation). Here we measure the *perceived*
end-to-end latency a judge feels: from a simulated completed user turn to the
first audible TTS byte, across the three LLM providers (minimax / claude /
inference) so the team can pick the demo's main brain by REAL numbers.

No microphone is required. We simulate the LiveKit turn pipeline's text path:

  1. on_user_turn_completed pre-retrieval  — one Moss query (IDX_KOLS top_k=3),
     exactly as agent.py injects context BEFORE the LLM runs. This overlaps with
     the LLM in production (preemptive injection), so it is reported separately
     and NOT double-counted into the perceived number.
  2. LLM TTFT  — first streamed token from build_llm(provider) given the brief +
     a function tool, tool_choice forced so we measure function-calling TTFT
     (the real per-turn routing cost), per provider.
  3. First TTS byte (TTFB)  — MiniMax TTS.synthesize() first audio frame for a
     short spoken reply. This is the same TTS used for every provider, so it is
     measured ONCE and added to each provider's perceived total.

  perceived_e2e (streaming, overlapped) ~= LLM_TTFT + TTS_TTFB
  (retrieval is pre-injected and overlaps the LLM; STT + turn-detect happen
  before this harness's "completed user turn" start point — see 05-latency.md
  §4: streaming overlaps to ~max(stages), and the LLM stage dominates.)

Run (ENV_FILE convention, openai plugin needed for minimax/claude paths)::

    ENV_FILE=.env uv run --with livekit-plugins-openai python tests/latency_e2e.py
    # restrict providers / sample count:
    E2E_PROVIDERS=claude,minimax E2E_N=3 ENV_FILE=.env uv run \
        --with livekit-plugins-openai python tests/latency_e2e.py

Graceful degradation (CLAUDE.md "any external call needs a fallback"):
  - A provider whose key/relay is missing or errors is SKIPPED + annotated,
    never fatal — the other providers still produce real numbers.
  - Moss creds missing -> retrieval hop SKIPPED, LLM/TTS still measured.
  - MiniMax TTS unavailable -> TTS hop SKIPPED, LLM TTFT still reported.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_FILE", ".env"))
load_dotenv(".env.local", override=False)

# Make src/ importable (reuse the real llm_factory + scoring, no duplication).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# --- config (names only, no secrets) -----------------------------------------
PROVIDERS = [
    p.strip().lower()
    for p in os.getenv("E2E_PROVIDERS", "claude,minimax,inference").split(",")
    if p.strip()
]
N = int(os.getenv("E2E_N", "5"))            # LLM TTFT samples per provider
N_TTS = int(os.getenv("E2E_TTS_N", "3"))    # TTS first-byte samples
IDX_KOLS = os.getenv("ANSIO_KOLS_INDEX", "ansio_kols")

# A realistic completed user turn (the demo's opening brief) + a short spoken
# reply for the TTS first-byte measurement.
USER_BRIEF = (
    "We're building an AI coding assistant for indie developers and our growth "
    "has stalled. Find the right creators to partner with."
)
SPOKEN_REPLY = (
    "I looked at creators in the developer-tools space and found three "
    "undervalued channels that over-index on engagement. Let me walk you "
    "through them."
)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _print_table(header: list[str], rows: list[list[str]]) -> None:
    cols = len(header)
    widths = [len(h) for h in header]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])) if i < len(r) else 0)
    print("  " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  " + "-+-".join("-" * widths[i] for i in range(cols)))
    for r in rows:
        print(
            "  "
            + " | ".join(
                str(r[i] if i < len(r) else "").ljust(widths[i]) for i in range(cols)
            )
        )


# =========================================================================
# Hop 1 — pre-retrieval (on_user_turn_completed simulation)
# =========================================================================
async def measure_retrieval() -> tuple[float | None, str]:
    """Return (retrieval_wall_ms_P50, note). Mirrors agent.on_user_turn_completed."""
    pid, key = os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")
    if not pid or not key:
        return None, "SKIPPED: MOSS_PROJECT_ID/KEY not set"
    try:
        from moss import MossClient, QueryOptions

        c = MossClient(pid, key)
        # Resolve to the real index, else substitute an existing one (real number).
        try:
            existing = [getattr(i, "name", i) for i in await c.list_indexes()]
        except Exception:
            existing = []
        target = (
            IDX_KOLS
            if IDX_KOLS in existing
            else ("kols" if "kols" in existing else (existing[0] if existing else None))
        )
        if not target:
            return None, "SKIPPED: no Moss index available"
        await c.load_index(target)
        # warm once, then sample
        await c.query(target, USER_BRIEF, QueryOptions(top_k=3))
        walls: list[float] = []
        for _ in range(max(5, N)):
            t0 = time.perf_counter()
            await c.query(target, USER_BRIEF, QueryOptions(top_k=3))
            walls.append((time.perf_counter() - t0) * 1000.0)
        note = "" if target == IDX_KOLS else f" (substitute index [{target}])"
        return _pct(walls, 50), f"P50 over {len(walls)} samples{note}"
    except Exception as e:
        return None, f"ERROR {type(e).__name__}: {str(e)[:60]}"


# =========================================================================
# Hop 2 — LLM function-calling TTFT per provider (via build_llm)
# =========================================================================
async def measure_llm_ttft(provider: str) -> dict:
    """Return {provider, model, ttft_p50, ttft_min, n_ok, tool_rate, note}."""
    from llm_factory import model_label  # real factory, no duplication

    result = {
        "provider": provider,
        "model": model_label(provider),
        "ttft_p50": None,
        "ttft_min": None,
        "n_ok": 0,
        "tool_rate": "0/0",
        "note": "",
    }

    # Provider readiness pre-check (degrade, never fatal).
    if provider == "minimax" and not os.getenv("MINIMAX_API_KEY"):
        result["note"] = "SKIPPED: MINIMAX_API_KEY not set"
        return result
    if provider == "claude" and not os.getenv("ANTHROPIC_AUTH_TOKEN"):
        result["note"] = "SKIPPED: ANTHROPIC_AUTH_TOKEN not set"
        return result

    try:
        from livekit.agents import llm as lk_llm

        from llm_factory import build_llm
    except Exception as e:
        result["note"] = (
            f"SKIPPED: import failed ({type(e).__name__}); "
            "rerun with --with livekit-plugins-openai"
        )
        return result

    @lk_llm.function_tool()
    async def recommend_kols(brief: str) -> str:
        """Recommend the best undervalued creators for a product brief.

        Args:
            brief: The founder's product, audience, and goal in one line.
        """
        return ""

    chat_ctx = lk_llm.ChatContext()
    chat_ctx.add_message(role="user", content=USER_BRIEF)

    samples: list[float] = []
    tool_seen = 0
    for i in range(N):
        try:
            client = build_llm(provider)
            t0 = time.perf_counter()
            try:
                stream = client.chat(
                    chat_ctx=chat_ctx, tools=[recommend_kols], tool_choice="required"
                )
            except Exception:
                stream = client.chat(chat_ctx=chat_ctx, tools=[recommend_kols])
            ttft = None
            saw_tool = False
            async for chunk in stream:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                delta = getattr(chunk, "delta", None)
                if delta is not None and getattr(delta, "tool_calls", None):
                    saw_tool = True
            try:
                await stream.aclose()
            except Exception:
                pass
            if ttft is not None:
                samples.append(ttft)
                tool_seen += int(saw_tool)
                print(
                    f"    [{provider}] {i + 1}/{N}: TTFT={ttft:.0f}ms tool={saw_tool}"
                )
            else:
                print(f"    [{provider}] {i + 1}/{N}: no chunks")
        except Exception as e:
            print(
                f"    [{provider}] {i + 1}/{N}: ERROR "
                f"{type(e).__name__}: {str(e)[:60]}"
            )

    if samples:
        result["ttft_p50"] = _pct(samples, 50)
        result["ttft_min"] = min(samples)
        result["n_ok"] = len(samples)
        result["tool_rate"] = f"{tool_seen}/{len(samples)}"
    else:
        result["note"] = result["note"] or "FAILED: no TTFT samples"
    return result


# =========================================================================
# Hop 3 — MiniMax TTS first-byte (shared across all providers)
# =========================================================================
async def measure_tts_ttfb() -> tuple[float | None, str]:
    """Return (tts_ttfb_p50_ms, note). First audio frame from MiniMax.synthesize."""
    if not os.getenv("MINIMAX_API_KEY"):
        return None, "SKIPPED: MINIMAX_API_KEY not set"
    try:
        from livekit.agents.utils import http_context
        from livekit.plugins import minimax
    except Exception as e:
        return None, f"SKIPPED: minimax plugin import failed ({type(e).__name__})"

    samples: list[float] = []
    # The minimax plugin grabs its aiohttp session from the agent's http_context,
    # which only exists inside a job. Running standalone (this script), we open
    # one explicitly — exactly what the plugin's RuntimeError instructs.
    async with http_context.open():
        try:
            tts = minimax.TTS(
                model=os.getenv("ANSIO_TTS_MODEL", "speech-2.8-turbo"),
                voice=os.getenv("ANSIO_TTS_VOICE", "English_Persuasive_Man"),
                language_boost="auto",
                sample_rate=24000,
            )
        except Exception as e:
            return None, (
                f"SKIPPED: TTS construct failed ({type(e).__name__}: {str(e)[:50]})"
            )

        for i in range(N_TTS):
            try:
                t0 = time.perf_counter()
                stream = tts.synthesize(SPOKEN_REPLY)
                first = None
                async for _frame in stream:
                    first = (time.perf_counter() - t0) * 1000.0
                    break
                try:
                    await stream.aclose()
                except Exception:
                    pass
                if first is not None:
                    samples.append(first)
                    print(f"    [tts] {i + 1}/{N_TTS}: first byte={first:.0f}ms")
                else:
                    print(f"    [tts] {i + 1}/{N_TTS}: no audio frame")
            except Exception as e:
                print(
                    f"    [tts] {i + 1}/{N_TTS}: ERROR "
                    f"{type(e).__name__}: {str(e)[:60]}"
                )
        try:
            await tts.aclose()
        except Exception:
            pass

    if not samples:
        return None, "FAILED: no TTS first-byte samples"
    return _pct(samples, 50), f"P50 over {len(samples)} samples"


# =========================================================================
async def main() -> None:
    print("=" * 78)
    print("ANSIO end-to-end perceived-latency harness (user turn -> first byte)")
    print("=" * 78)
    print(f"  providers under test: {PROVIDERS}")
    print(f"  N(llm)={N}  N(tts)={N_TTS}")

    print("\n-- Hop 1: pre-retrieval (on_user_turn_completed) ----------------")
    retr_p50, retr_note = await measure_retrieval()
    if retr_p50 is not None:
        print(f"  Moss retrieval P50: {retr_p50:.1f}ms")
        print(f"  ({retr_note})")
    else:
        print(f"  Moss retrieval: {retr_note}")

    print("\n-- Hop 2: LLM function-calling TTFT (per provider) --------------")
    llm_rows = []
    for p in PROVIDERS:
        print(f"  provider: {p}")
        llm_rows.append(await measure_llm_ttft(p))

    print("\n-- Hop 3: MiniMax TTS first byte (shared) -----------------------")
    tts_p50, tts_note = await measure_tts_ttfb()
    if tts_p50 is not None:
        print(f"  TTS first-byte P50: {tts_p50:.0f}ms ({tts_note})")
    else:
        print(f"  TTS first-byte: {tts_note}")

    # ----- composed perceived end-to-end table -----
    print("\n" + "=" * 78)
    print("PERCEIVED END-TO-END (streaming, overlapped)  =  LLM_TTFT + TTS_TTFB")
    print("retrieval is pre-injected (overlaps LLM) -> shown but not added")
    print("=" * 78)
    header = [
        "provider",
        "model",
        "LLM TTFT P50",
        "LLM min",
        "+TTS TTFB",
        "perceived P50",
        "tool",
        "note",
    ]
    rows = []
    for r in llm_rows:
        ttft = r["ttft_p50"]
        if ttft is not None:
            perceived = ttft + (tts_p50 or 0.0)
            rows.append([
                r["provider"],
                r["model"],
                f"{ttft:.0f}ms",
                f"{r['ttft_min']:.0f}ms",
                f"{tts_p50:.0f}ms" if tts_p50 is not None else "n/a",
                f"{perceived:.0f}ms",
                r["tool_rate"],
                r["note"],
            ])
        else:
            rows.append([
                r["provider"],
                r["model"],
                "n/a",
                "n/a",
                f"{tts_p50:.0f}ms" if tts_p50 is not None else "n/a",
                "n/a",
                r["tool_rate"],
                r["note"],
            ])
    _print_table(header, rows)

    print("\n  budget reference (05-latency.md §4): streaming target 400-800ms.")
    if retr_p50 is not None:
        print(f"  pre-retrieval P50 {retr_p50:.1f}ms overlaps the LLM (not summed).")
    print("  copy this table into .omc/research/ansio-v4/i-e2e.md.")


if __name__ == "__main__":
    asyncio.run(main())
