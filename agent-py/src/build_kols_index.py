"""Build the Moss ``kols`` index from kols.json and measure query latency.

Run via: uv run python src/build_kols_index.py
Needs MOSS_PROJECT_ID / MOSS_PROJECT_KEY in agent-py/.env.local.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from moss import DocumentInfo, MossClient, QueryOptions

AGENT_DIR = Path(__file__).resolve().parent.parent
KOLS_PATH = AGENT_DIR / "kols.json"
ENV_PATH = AGENT_DIR / ".env.local"
load_dotenv(ENV_PATH)

INDEX = os.getenv("MOSS_KOL_INDEX_NAME", "kols")
MODEL_ID = os.getenv("MOSS_MODEL_ID", "moss-minilm")


def load_docs() -> list[DocumentInfo]:
    data = json.loads(KOLS_PATH.read_text(encoding="utf-8"))
    docs = []
    for e in data:
        md = {str(k): str(v) for k, v in (e.get("metadata") or {}).items()}
        docs.append(DocumentInfo(id=str(e["id"]), text=str(e["text"]), metadata=md))
    return docs


async def main() -> None:
    pid, key = os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")
    if not pid or not key:
        raise SystemExit("Missing MOSS_PROJECT_ID / MOSS_PROJECT_KEY in .env.local")

    docs = load_docs()
    client = MossClient(pid, key)

    print(f"Creating Moss index '{INDEX}' with {len(docs)} KOLs (model={MODEL_ID})...")
    t0 = time.perf_counter()
    res = await client.create_index(INDEX, docs, MODEL_ID)
    print(f"  indexed in {time.perf_counter() - t0:.1f}s "
          f"(job={res.job_id}, docs={res.doc_count})")

    # Load the index for querying, then benchmark query latency.
    await client.load_index(INDEX)

    probes = [
        "fitness influencer on instagram for a protein supplement brand",
        "tech reviewer on youtube to launch a new laptop",
        "beauty creator for a skincare campaign in the US",
        "crypto and web3 educator on x with an engaged audience",
        "gaming streamer on twitch for an esports energy drink",
        "sustainable living creator for an eco-friendly home brand",
        "finance youtuber to talk about a budgeting app",
        "food creator in japan for a kitchen appliance launch",
    ]

    print("\nQuery latency (top_k=5):")
    timings = []
    for q in probes:
        t0 = time.perf_counter()
        r = await client.query(INDEX, q, QueryOptions(top_k=5))
        dt = (time.perf_counter() - t0) * 1000
        timings.append(dt)
        server_ms = getattr(r, "time_taken_ms", None)
        top = (getattr(r, "docs", None) or [])[:1]
        top_name = ""
        if top:
            md = getattr(top[0], "metadata", {}) or {}
            top_name = f"{md.get('name','?')} [{md.get('niche','?')}/{md.get('platform','?')}]"
        print(f"  {dt:6.1f}ms wall (server {server_ms}ms)  '{q[:42]}...' -> {top_name}")

    print(f"\nAvg wall latency: {sum(timings)/len(timings):.1f}ms  "
          f"min {min(timings):.1f}ms  max {max(timings):.1f}ms")

    # Demonstrate a filtered query (exact-match facet).
    print("\nFiltered query (platform=YouTube, niche=tech):")
    t0 = time.perf_counter()
    r = await client.query(
        INDEX,
        "best creator to review a flagship smartphone",
        QueryOptions(
            top_k=5,
            filter={"field": "platform", "condition": {"$eq": "YouTube"}},
        ),
    )
    dt = (time.perf_counter() - t0) * 1000
    for d in (getattr(r, "docs", None) or [])[:5]:
        md = getattr(d, "metadata", {}) or {}
        print(f"   {md.get('name'):20s} {md.get('platform'):10s} {md.get('niche'):8s} "
              f"{md.get('followers'):>9s} followers  score={getattr(d,'score',None)}")
    print(f"  filtered query wall latency: {dt:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
