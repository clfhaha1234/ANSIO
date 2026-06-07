"""End-to-end eval of the KOL agent: classify -> search_kols -> recommend.

Drives the real `Assistant` through LiveKit's AgentSession in text mode (no
audio), for a set of realistic campaign briefs. For each brief it reports:
  - the filters the agent classified the request into (platform / niche),
  - the search_kols query it issued,
  - the creators it recommended (final spoken text),
  - latency: the full agent turn, and the raw Moss query underneath it.

Run: uv run python src/eval_kols.py
Needs LIVEKIT_* (for Inference LLM/judge) + MOSS_* in agent-py/.env.local.
"""

from __future__ import annotations

import asyncio
import os
import time

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

load_dotenv(".env.local")

from livekit.agents import AgentSession  # noqa: E402

from agent import KOL_INDEX, Assistant  # noqa: E402

BRIEFS = [
    # explicit platform + niche
    "I'm launching a protein powder and want a fitness influencer on Instagram to promote it.",
    # vague — agent must classify the niche itself
    "We make a budgeting app for young professionals. Who should we work with?",
    # platform-specific, tech
    "Need a YouTuber to review our new mechanical keyboard for gamers and developers.",
    # region/language signal
    "Looking for a Japanese food creator to feature our premium kitchen knives.",
    # crypto
    "We're a web3 wallet startup and want an educational creator on X to explain it.",
    # beauty, audience-size signal
    "A skincare brand wants a big beauty creator, ideally over a million followers.",
]


def _extract(history) -> tuple[dict, str, str]:
    """Pull (search_kols args, search result text, final assistant text) from history."""
    args: dict = {}
    tool_out = ""
    final_msg = ""
    for item in history.items:
        itype = getattr(item, "type", None)
        if itype == "function_call" and getattr(item, "name", "") == "search_kols":
            raw = getattr(item, "arguments", "") or ""
            import json

            with __import__("contextlib").suppress(Exception):
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        elif itype == "function_call_output":
            out = getattr(item, "output", "") or ""
            tool_out = out if isinstance(out, str) else str(out)
        elif itype == "message" and getattr(item, "role", "") == "assistant":
            content = getattr(item, "content", None)
            if isinstance(content, list):
                final_msg = " ".join(str(c) for c in content)
            else:
                final_msg = str(content)
    return args, tool_out, final_msg


async def main() -> None:
    print(f"KOL agent eval — index '{KOL_INDEX}', LLM gpt-4.1-mini\n" + "=" * 72)
    turn_times = []
    hits = 0

    for i, brief in enumerate(BRIEFS, 1):
        async with AgentSession() as session:
            await session.start(Assistant())  # no room -> publish/memory no-op

            t0 = time.perf_counter()
            result = await session.run(user_input=brief)
            turn_ms = (time.perf_counter() - t0) * 1000
            turn_times.append(turn_ms)

            args, tool_out, final_msg = _extract(session.history)

            searched = bool(args)
            hits += int(searched)
            print(f"\n[{i}] BRIEF: {brief}")
            print(
                f"    classified -> platform={args.get('platform','') or '-':10s} "
                f"niche={args.get('niche','') or '-':12s} searched={searched}"
            )
            if args.get("query"):
                print(f"    search query: \"{args['query']}\"")
            print(f"    latency: turn={turn_ms:6.0f}ms (Moss + LLM tool round-trip)")
            if tool_out:
                first = tool_out.splitlines()[0] if tool_out.splitlines() else ""
                print(f"    top match returned: {first[:110]}")
            print(f"    RECOMMENDS: {final_msg[:300]}")

    print("\n" + "=" * 72)
    print(
        f"Searched in {hits}/{len(BRIEFS)} briefs.  "
        f"Turn latency: avg {sum(turn_times)/len(turn_times):.0f}ms  "
        f"min {min(turn_times):.0f}ms  max {max(turn_times):.0f}ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
