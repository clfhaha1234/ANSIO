"""Moss contract checks — verify the Moss behaviors the 5 PRD tools depend on.

These are NOT unit tests (they need real Moss credentials and a built index).
They are a pre-flight the implementer runs after building an index, to confirm
Moss behaves the way the tool code assumes BEFORE wiring it into the agent.

Every assumption here was verified live against the `kols` index on 2026-06-06;
re-run against your own indexes to catch data/schema drift.

Run:
    ENV_FILE=../.env.local uv run python tests/moss_contract.py            # uses `kols`
    INDEX=content ENV_FILE=../.env.local uv run python tests/moss_contract.py

What it checks (each prints PASS / FAIL / SKIP with evidence):
    1. load_index is mandatory — filters need a locally-loaded index, and the
       un-loaded cloud path is unreliable (observed HTTP 503).
    2. $eq exact match works.
    3. $eq is CASE- and CHARACTER-sensitive (wrong case -> 0 docs). [landmine]
    4. $and compound filter works.
    5. Numeric range ($gte/$lt) works ON STRING-stored metadata. [PRD principle
       #3 is wrong: you CAN push budget filters into Moss]
    6. KV point-lookup: top_k=1 + $eq on an id-like field returns the exact doc.
    7. Query latency is in the single-digit ms range (the <10ms pitch number).
"""

from __future__ import annotations

import asyncio
import os
import time

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_FILE", ".env.local"))

from moss import MossClient, QueryOptions

INDEX = os.getenv("INDEX") or os.getenv("MOSS_KOL_INDEX_NAME") or "kols"

_PASS, _FAIL, _SKIP = 0, 0, 0


def _result(ok, label, detail=""):
    global _PASS, _FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  [{tag}] {label}" + (f"  — {detail}" if detail else ""))


def _skip(label, detail=""):
    global _SKIP
    _SKIP += 1
    print(f"  [SKIP] {label}" + (f"  — {detail}" if detail else ""))


async def _q(c, q, **opts):
    return await c.query(INDEX, q, QueryOptions(**opts))


async def main() -> None:
    pid, key = os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")
    if not pid or not key:
        raise SystemExit("Set MOSS_PROJECT_ID / MOSS_PROJECT_KEY (ENV_FILE=...).")
    c = MossClient(pid, key)

    print(f"Moss contract checks against index '{INDEX}'\n" + "=" * 70)

    # Discover the schema so checks adapt to any index.
    await c.load_index(INDEX)
    probe = await _q(c, "creator developer tool", top_k=5)
    if not probe.docs:
        raise SystemExit(f"Index '{INDEX}' returned no docs — build/seed it first.")
    sample_md = probe.docs[0].metadata or {}
    fields = set(sample_md)
    print(f"Discovered metadata fields: {sorted(fields)}\n")

    def pick_str_field():
        for f in ("platform", "niche", "category", "brand", "doc_type", "tier"):
            if f in fields:
                return f
        return next(iter(fields), None)

    def pick_num_field():
        for f in ("price_usd", "followers", "views", "engagement_pct"):
            if f in fields:
                return f
        return None

    def pick_id_field():
        for f in ("handle", "kol_handle", "kol_id", "id", "name"):
            if f in fields:
                return f
        return None

    # --- 1. load_index mandatory + cloud path unreliable -----------------
    print("1. load_index is mandatory (filters + reliability)")
    sfield = pick_str_field()
    val = sample_md.get(sfield)
    try:
        await c.unload_index(INDEX)
        r = await _q(c, "creator", top_k=5,
                     filter={"field": sfield, "condition": {"$eq": val}})
        plats = [(d.metadata or {}).get(sfield) for d in r.docs]
        honored = plats and all(p == val for p in plats)
        _result(not honored or len(r.docs) == 0,
                "un-loaded query does NOT reliably honor filter",
                f"{sfield}={val!r} -> {plats} (mixed/empty => must load_index)")
    except Exception as e:
        _result(True, "un-loaded cloud query path is unreliable",
                f"raised {type(e).__name__}: {str(e)[:80]} (=> always load_index)")
    finally:
        await c.load_index(INDEX)

    # --- 2. $eq exact match ---------------------------------------------
    print(f"\n2. $eq exact match on '{sfield}'")
    r = await _q(c, "creator", top_k=5,
                 filter={"field": sfield, "condition": {"$eq": val}})
    got = [(d.metadata or {}).get(sfield) for d in r.docs]
    _result(bool(r.docs) and all(g == val for g in got),
            f"$eq {sfield}={val!r}", f"n={len(r.docs)} all=={val}: {all(g==val for g in got)}")

    # --- 3. $eq case-sensitivity (landmine) ------------------------------
    print(f"\n3. $eq case-sensitivity (LLM-supplied values must be normalized)")
    if isinstance(val, str) and val.lower() != val:
        r = await _q(c, "creator", top_k=5,
                     filter={"field": sfield, "condition": {"$eq": val.lower()}})
        _result(len(r.docs) == 0,
                f"wrong-case $eq {sfield}={val.lower()!r} returns 0",
                f"n={len(r.docs)} (0 => case-sensitive; normalize before filtering)")
    else:
        _skip("no mixed-case value to probe", f"{sfield}={val!r}")

    # --- 4. $and compound ------------------------------------------------
    print("\n4. $and compound filter")
    f2 = [f for f in (pick_str_field(), "niche", "tier", "platform") if f in fields]
    f2 = list(dict.fromkeys(f2))[:2]
    if len(f2) == 2:
        conds = [{"field": f, "condition": {"$eq": sample_md.get(f)}} for f in f2]
        r = await _q(c, "creator", top_k=5, filter={"$and": conds})
        ok = all((d.metadata or {}).get(f2[0]) == sample_md.get(f2[0])
                 and (d.metadata or {}).get(f2[1]) == sample_md.get(f2[1])
                 for d in r.docs)
        _result(ok, f"$and [{f2[0]}, {f2[1]}]", f"n={len(r.docs)} all-match={ok}")
    else:
        _skip("need two string fields for $and")

    # --- 5. numeric range on string metadata (PRD #3 correction) ---------
    nfield = pick_num_field()
    print(f"\n5. numeric range on string-stored '{nfield}' (PRD principle #3)")
    if nfield:
        vals = sorted(
            int(float((d.metadata or {})[nfield]))
            for d in probe.docs if (d.metadata or {}).get(nfield)
        )
        mid = vals[len(vals) // 2] if vals else 1000
        r = await _q(c, "creator", top_k=10,
                     filter={"field": nfield, "condition": {"$gte": mid}})
        got = [int(float((d.metadata or {})[nfield])) for d in r.docs]
        _result(bool(got) and all(g >= mid for g in got),
                f"$gte:{mid} on {nfield}",
                f"n={len(got)} all>= {mid}: {all(g>=mid for g in got)} "
                f"=> CAN push range filters into Moss")
    else:
        _skip("no numeric field found")

    # --- 6. KV point-lookup ---------------------------------------------
    idf = pick_id_field()
    print(f"\n6. KV point-lookup: top_k=1 + $eq on '{idf}'")
    if idf and sample_md.get(idf):
        idv = sample_md[idf]
        r = await _q(c, idv, top_k=1,
                     filter={"field": idf, "condition": {"$eq": idv}})
        ok = len(r.docs) == 1 and (r.docs[0].metadata or {}).get(idf) == idv
        _result(ok, f"point-lookup {idf}={idv!r}", f"n={len(r.docs)}")
        # handle @-prefix landmine
        if idf in ("handle", "kol_handle") and not str(idv).startswith("@"):
            r2 = await _q(c, idv, top_k=1,
                          filter={"field": idf, "condition": {"$eq": "@" + str(idv)}})
            _result(len(r2.docs) == 0,
                    "handle WITH '@' returns 0 (strip @ before lookup!)",
                    f"stored={idv!r}, queried '@{idv}' -> n={len(r2.docs)}")
    else:
        _skip("no id-like field found")

    # --- 7. latency ------------------------------------------------------
    print("\n7. query latency (<10ms pitch claim)")
    walls = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = await _q(c, "developer tools creator for a startup", top_k=5)
        walls.append((time.perf_counter() - t0) * 1000)
    server = getattr(r, "time_taken_ms", None)
    avg = sum(walls) / len(walls)
    _result(avg < 50 and (server is None or server < 10),
            f"warm latency", f"server~{server}ms, wall avg {avg:.1f}ms (incl. python)")

    print("\n" + "=" * 70)
    print(f"contract: {_PASS} pass, {_FAIL} fail, {_SKIP} skip")
    if _FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
