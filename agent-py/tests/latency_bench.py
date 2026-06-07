"""ANSIO latency baseline harness — the source of truth for the HUD millisecond
numbers and the pitch latency claims.

This is NOT a unit test. It needs real Moss credentials (and, for group ③, a
MiniMax API key + livekit-plugins-openai). It measures three things and prints
each as an aligned table you can screenshot for the team:

  ① Per-index Moss query latency P50/P95 (n>=20) for the three ansio_ indexes,
     contrasting warm (load_index done) vs cold first-hop (right after load).
  ② The 5-hop recall chain (Moss-only part) run SERIALLY, simulating the PRD
     §1 retrieval flow: products semantic -> content $eq brand -> kols $eq
     handle point-lookup -> kols semantic-similar -> local scoring (sleep(0)
     placeholder). Prints each hop + the total.
  ③ MiniMax-M2 single function-calling TTFT sampling (n>=5), reusing B2's access
     path (openai.LLM with MINIMAX_BASE_URL/MINIMAX_MODEL/MINIMAX_API_KEY). If
     MiniMax is unavailable it is SKIPPED and annotated, never fatal.

Run (ENV_FILE convention matches the rest of the repo's harnesses):

    ENV_FILE=.env uv run python tests/latency_bench.py
    # group ③ needs the openai plugin (B2 path A):
    ENV_FILE=.env uv run --with livekit-plugins-openai python tests/latency_bench.py

Graceful degradation (per CLAUDE.md "any external call needs a fallback"):
  - ansio_* indexes not built yet (fei project 3/3 full, see prep-c3-build.md)
    -> group ① falls back to whatever real indexes exist (e.g. `kols`) so the
       numbers are still real, and clearly labels the substitution + BLOCKED.
  - MiniMax key / plugin missing or call errors -> group ③ SKIPPED, annotated.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_FILE", ".env.local"))

from moss import MossClient, QueryOptions

# --- config (no secrets; only names) ----------------------------------------
ANSIO_INDEXES = ["ansio_products", "ansio_content", "ansio_kols"]
N_LATENCY = int(os.getenv("BENCH_N", "20"))          # group ① samples per index
N_TTFT = int(os.getenv("BENCH_TTFT_N", "5"))         # group ③ samples
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2")

# Representative warm query reused across hops.
SEMANTIC_QUERY = "developer tools creator for an AI coding startup"


def _pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile on wall-clock ms samples."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


async def _existing_indexes(c: MossClient) -> list[str]:
    try:
        idx = await c.list_indexes()
        return [getattr(i, "name", i) for i in idx]
    except Exception as e:  # noqa: BLE001 — never fatal, degrade gracefully
        print(f"  ! list_indexes failed ({type(e).__name__}); assuming none")
        return []


async def _time_query(
    c: MossClient, name: str, q: str, **opts
) -> tuple[float, int | None, int]:
    """Return (wall_ms, server_ms, ndocs) for one query."""
    t0 = time.perf_counter()
    r = await c.query(name, q, QueryOptions(**opts))
    wall = (time.perf_counter() - t0) * 1000.0
    return wall, getattr(r, "time_taken_ms", None), len(r.docs)


# =========================================================================
# Group ① — per-index P50/P95, warm vs cold first-hop
# =========================================================================
async def group1_index_latency(c: MossClient, existing: list[str]) -> None:
    print("\n" + "=" * 78)
    print("GROUP ① — per-index Moss query latency  (n=%d each, top_k=5)" % N_LATENCY)
    print("=" * 78)

    targets: list[tuple[str, str | None]] = []  # (logical_label, actual_index)
    for want in ANSIO_INDEXES:
        if want in existing:
            targets.append((want, want))
        else:
            # ansio_* not built (BLOCKED) -> substitute a real index so the
            # number is real, not fabricated. Prefer `kols`, else first existing.
            sub = "kols" if "kols" in existing else (existing[0] if existing else None)
            targets.append((f"{want} -> [{sub or 'NONE'}]*", sub))

    rows: list[list[str]] = []
    header = ["index", "cold1_wall", "warm_P50", "warm_P95", "srv_P50", "srv_P95"]
    for label, name in targets:
        if not name:
            rows.append([label, "n/a", "n/a", "n/a", "n/a", "n/a"])
            print(f"  {label:34s}  no index available — SKIP")
            continue
        try:
            # cold first hop: load, then immediately query once (the first query
            # after load pays the cache-warming cost).
            await c.load_index(name)
            cold_wall, _, _ = await _time_query(c, name, SEMANTIC_QUERY, top_k=5)

            walls: list[float] = []
            servers: list[float] = []
            for _ in range(N_LATENCY):
                w, s, _ = await _time_query(c, name, SEMANTIC_QUERY, top_k=5)
                walls.append(w)
                if s is not None:
                    servers.append(float(s))
            rows.append([
                label,
                f"{cold_wall:.1f}",
                f"{_pct(walls, 50):.1f}",
                f"{_pct(walls, 95):.1f}",
                f"{_pct(servers, 50):.1f}" if servers else "n/a",
                f"{_pct(servers, 95):.1f}" if servers else "n/a",
            ])
        except Exception as e:  # noqa: BLE001
            rows.append([label, "ERR", "ERR", "ERR", "ERR", "ERR"])
            print(f"  {label:34s}  query failed: {type(e).__name__}: {str(e)[:60]}")

    _print_table(header, rows)
    print("  * substituted index: ansio_* not built (fei project 3/3 full, see")
    print("    prep-c3-build.md). Numbers are REAL but measured on the substitute.")
    print("  cold1_wall = first query right after load_index (cache-warming hop).")
    print("  warm_* = wall-clock incl. python; srv_* = Moss server time_taken_ms.")


# =========================================================================
# Group ② — 5-hop recall chain, serial, Moss-only part
# =========================================================================
async def group2_recall_chain(c: MossClient, existing: list[str]) -> None:
    print("\n" + "=" * 78)
    print("GROUP ② — 5-hop recall chain serial total (Moss-only, +local scoring)")
    print("=" * 78)

    def resolve(want: str) -> str | None:
        if want in existing:
            return want
        return "kols" if "kols" in existing else (existing[0] if existing else None)

    products = resolve("ansio_products")
    content = resolve("ansio_content")
    kols = resolve("ansio_kols")
    substituted = any(w not in existing for w in ANSIO_INDEXES)

    # Preload everything used so the chain measures steady-state hop latency,
    # not load cost (matches agent on_enter preloading the indexes, T7).
    for nm in {x for x in (products, content, kols) if x}:
        try:
            await c.load_index(nm)
        except Exception as e:  # noqa: BLE001
            print(f"  ! load_index({nm}) failed: {type(e).__name__}")

    # Pick a real handle value from kols data when substituting, so the $eq
    # point-lookup hop actually matches (else it measures an empty-filter query).
    brand_val = "cursor"
    handle_val = None
    if kols:
        try:
            probe = await c.query(kols, SEMANTIC_QUERY, QueryOptions(top_k=1))
            if probe.docs:
                md = probe.docs[0].metadata or {}
                handle_val = md.get("handle") or md.get("kol_handle")
        except Exception:  # noqa: BLE001
            pass

    hops: list[tuple[str, str | None, dict]] = [
        ("1 products semantic", products, {"top_k": 5}),
        ("2 content $eq brand", content,
         {"top_k": 50,
          "filter": {"field": "brand", "condition": {"$eq": brand_val}}}),
        ("3 kols $eq handle (point)", kols,
         ({"top_k": 1,
           "filter": {"field": "handle", "condition": {"$eq": handle_val}}}
          if handle_val else {"top_k": 1})),
        ("4 kols semantic similar (wide)", kols, {"top_k": 80}),
    ]

    rows: list[list[str]] = []
    header = ["hop", "wall_ms", "srv_ms", "ndocs"]
    total_wall = 0.0
    for label, name, opts in hops:
        if not name:
            rows.append([label, "n/a", "n/a", "0"])
            continue
        try:
            wall, srv, nd = await _time_query(c, name, SEMANTIC_QUERY, **opts)
        except Exception:  # noqa: BLE001 — a bad filter shouldn't kill the chain
            opts2 = {k: v for k, v in opts.items() if k != "filter"}
            try:
                wall, srv, nd = await _time_query(c, name, SEMANTIC_QUERY, **opts2)
                label += " (no-filter retry)"
            except Exception as e2:  # noqa: BLE001
                rows.append([label, "ERR", "ERR", "0"])
                print(f"  ! hop {label} failed: {type(e2).__name__}")
                continue
        total_wall += wall
        rows.append([label, f"{wall:.1f}",
                     str(srv) if srv is not None else "n/a", str(nd)])

    # hop 5 = local scoring placeholder (sleep(0)); measure its wall cost.
    t0 = time.perf_counter()
    await asyncio.sleep(0)  # placeholder for score_and_rank Python work (T1)
    score_wall = (time.perf_counter() - t0) * 1000.0
    total_wall += score_wall
    rows.append(["5 local scoring (sleep(0))", f"{score_wall:.3f}", "0", "-"])
    rows.append(["TOTAL (serial)", f"{total_wall:.1f}", "-", "-"])

    _print_table(header, rows)
    if substituted:
        print("  * one or more ansio_* indexes absent; substituted real index")
        print("    (kols) for the missing ones — total is a real serial measurement")
        print("    on substitutes, NOT the final ansio_* topology.")
    print("  hop 5 is a sleep(0) placeholder for Python score_and_rank (T1).")


# =========================================================================
# Group ③ — MiniMax-M2 function-calling TTFT
# =========================================================================
async def group3_minimax_ttft() -> None:
    print("\n" + "=" * 78)
    print("GROUP ③ — MiniMax-M2 single function-calling TTFT  (n=%d)" % N_TTFT)
    print("=" * 78)

    if not os.getenv("MINIMAX_API_KEY"):
        print("  SKIPPED: MINIMAX_API_KEY not set.")
        return

    try:
        from livekit.plugins import openai
    except Exception as e:  # noqa: BLE001
        print(f"  SKIPPED: livekit-plugins-openai not importable ({type(e).__name__}).")
        print("           rerun with: uv run --with livekit-plugins-openai ...")
        return

    from livekit.agents import llm as lk_llm

    # Real decorated function tool (FunctionTool is a Protocol, not constructable;
    # the decorator is the supported way to hand the LLM a callable tool schema).
    @lk_llm.function_tool()
    async def find_competitors(product_desc: str) -> str:
        """Find competitor products for a startup product description.

        Args:
            product_desc: Natural-language description of the founder's product.
        """
        return ""

    chat_ctx = lk_llm.ChatContext()
    chat_ctx.add_message(
        role="user",
        content="We're building an AI coding tool and growth stalled. "
                "Find our competitors using the tool.",
    )

    # tool_choice="required" forces a function call so we truly measure
    # function-calling TTFT (the agent's per-turn retrieval-routing cost), not a
    # plain chat completion. Falls back to free choice if the arg is unsupported.
    samples: list[float] = []
    fc_seen = 0
    for i in range(N_TTFT):
        try:
            client = openai.LLM(
                model=MINIMAX_MODEL,
                base_url=MINIMAX_BASE_URL,
                api_key=os.getenv("MINIMAX_API_KEY"),
            )
            t0 = time.perf_counter()
            try:
                stream = client.chat(
                    chat_ctx=chat_ctx, tools=[find_competitors],
                    tool_choice="required",
                )
            except Exception:  # noqa: BLE001
                stream = client.chat(chat_ctx=chat_ctx, tools=[find_competitors])
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
            except Exception:  # noqa: BLE001
                pass
            if ttft is not None:
                samples.append(ttft)
                fc_seen += int(saw_tool)
                print(f"  sample {i+1}/{N_TTFT}: TTFT={ttft:.0f}ms  tool_call={saw_tool}")
            else:
                print(f"  sample {i+1}/{N_TTFT}: no chunks received")
        except Exception as e:  # noqa: BLE001
            print(f"  sample {i+1}/{N_TTFT}: ERROR {type(e).__name__}: {str(e)[:70]}")

    if not samples:
        print("  FAILED: no successful TTFT samples (see errors above).")
        return

    rows = [
        ["TTFT min", f"{min(samples):.0f} ms"],
        ["TTFT P50", f"{_pct(samples, 50):.0f} ms"],
        ["TTFT P95", f"{_pct(samples, 95):.0f} ms"],
        ["TTFT mean", f"{statistics.mean(samples):.0f} ms"],
        ["tool_call rate", f"{fc_seen}/{len(samples)}"],
    ]
    _print_table(["metric", "value"], rows)


# --- table printer -----------------------------------------------------------
def _print_table(header: list[str], rows: list[list[str]]) -> None:
    cols = len(header)
    widths = [len(h) for h in header]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])) if i < len(r) else 0)
    print("  " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  " + "-+-".join("-" * widths[i] for i in range(cols)))
    for r in rows:
        print("  " + " | ".join(
            str(r[i] if i < len(r) else "").ljust(widths[i]) for i in range(cols)))


async def main() -> None:
    pid, key = os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")
    if not pid or not key:
        raise SystemExit("Set MOSS_PROJECT_ID / MOSS_PROJECT_KEY (ENV_FILE=...).")
    c = MossClient(pid, key)

    print("ANSIO latency baseline harness")
    print(f"  Moss project creds: {'set' if pid and key else 'MISSING'}")
    existing = await _existing_indexes(c)
    print(f"  Existing indexes: {existing}")
    ansio_built = [x for x in ANSIO_INDEXES if x in existing]
    print(f"  ansio_* built: {ansio_built or 'NONE (BLOCKED — see prep-c3-build.md)'}")

    await group1_index_latency(c, existing)
    await group2_recall_chain(c, existing)
    await group3_minimax_ttft()

    print("\n" + "=" * 78)
    print("done. copy the tables above into prep-d-latency.md.")


if __name__ == "__main__":
    asyncio.run(main())
