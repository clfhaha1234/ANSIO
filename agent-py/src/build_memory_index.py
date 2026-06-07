"""Build and smoke-test the ANSIO per-user profile (memory) index in Moss.

This is a field-ops script for hackathon day. It:

1. Loads Moss credentials the same way the rest of the agent does.
2. Lists existing indexes so you can see what is already there.
3. Creates the profile index (``ANSIO_MEMORY_INDEX``, default ``ansio_memory``)
   if it does not already exist. If creation fails because the free tier index
   limit is hit, it prints the explicit downgrade instruction and exits non-zero.
4. Runs a full roundtrip smoke test: ``load_index`` -> ``add_docs`` a profile
   doc -> re-``load_index`` -> ``query`` with a per-user filter -> assert the
   text comes back -> ``delete_docs`` to clean up.

Run it from agent-py:

    uv run python src/build_memory_index.py

If the 4th index cannot be created on the free tier, fall back to packing
profiles into the content index:

    ANSIO_MEMORY_INDEX=ansio_content uv run python src/build_memory_index.py

Every external Moss call is wrapped so a transient failure prints a clear
message instead of dumping a raw traceback on the operator.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from moss import DocumentInfo, MossClient, QueryOptions

# Resolve paths relative to this file (src/ -> agent-py/) so the script runs
# regardless of the current working directory.
AGENT_DIR = Path(__file__).resolve().parent.parent

# Match the agent's env-loading convention: a primary env file (overridable via
# ENV_FILE) plus a local override file.
PRIMARY_ENV = os.getenv("ENV_FILE", str(AGENT_DIR / ".env"))
LOCAL_ENV = str(AGENT_DIR / ".env.local")
load_dotenv(PRIMARY_ENV)
load_dotenv(LOCAL_ENV)

DEFAULT_MEMORY_INDEX = "ansio_memory"
PROFILE_DOC_TYPE = "user_profile"

# Sentinel text written when a profile is "cleared" but delete_docs is
# unavailable; loaders treat this as "no profile".
EMPTY_PROFILE_SENTINEL = "__EMPTY_PROFILE__"

# Seed + smoke-test identifiers. Stable doc ids => upsert semantics.
SEED_USER_ID = "_seed"
SEED_DOC_ID = f"profile-{SEED_USER_ID}"
SMOKE_USER_ID = "_smoke"
SMOKE_DOC_ID = f"profile-{SMOKE_USER_ID}"
SMOKE_PROFILE_TEXT = (
    "ANSIO smoke profile: likes outdoor gear and budget travel; "
    "prefers Xiaohongshu KOLs in the 5k-50k follower range."
)


def _profile_doc(user_id: str, text: str) -> DocumentInfo:
    """Build a profile DocumentInfo with the ANSIO metadata convention.

    Moss metadata values must be strings, so everything is coerced to str.
    """
    return DocumentInfo(
        id=f"profile-{user_id}",
        text=text,
        metadata={"user_id": str(user_id), "doc_type": PROFILE_DOC_TYPE},
    )


def _user_filter(user_id: str) -> dict:
    """Metadata filter scoping a query to a single user's profile."""
    return {"field": "user_id", "condition": {"$eq": str(user_id)}}


def _require_credentials() -> tuple[str, str]:
    project_id = os.getenv("MOSS_PROJECT_ID")
    project_key = os.getenv("MOSS_PROJECT_KEY")
    missing = [
        name
        for name, value in (
            ("MOSS_PROJECT_ID", project_id),
            ("MOSS_PROJECT_KEY", project_key),
        )
        if not value
    ]
    if missing:
        print(
            "[FATAL] Missing Moss credentials: "
            + ", ".join(missing)
            + f"\n        Set them in {PRIMARY_ENV} or {LOCAL_ENV} and retry.",
            file=sys.stderr,
        )
        sys.exit(2)
    assert project_id is not None
    assert project_key is not None
    return project_id, project_key


async def _list_existing(client: MossClient) -> set[str]:
    """List existing index names; degrade to empty set on failure."""
    print("\n[1/4] Listing existing indexes...")
    try:
        indexes = await client.list_indexes()
    except Exception as exc:
        print(f"  [WARN] list_indexes failed ({exc!r}); assuming index is absent.")
        return set()
    names = set()
    for info in indexes:
        name = getattr(info, "name", None)
        doc_count = getattr(info, "doc_count", "?")
        status = getattr(info, "status", "?")
        if name:
            names.add(name)
        print(f"  - {name} (docs={doc_count}, status={status})")
    if not names:
        print("  (no indexes found)")
    return names


async def _ensure_index(
    client: MossClient, index_name: str, existing: set[str]
) -> None:
    """Create the profile index if absent. Exit non-zero on quota failure."""
    print(f"\n[2/4] Ensuring profile index '{index_name}' exists...")
    if index_name in existing:
        print(f"  Index '{index_name}' already exists; skipping create.")
        return

    seed_docs = [_profile_doc(SEED_USER_ID, "seed")]
    try:
        result = await client.create_index(index_name, seed_docs)
    except Exception as exc:
        message = str(exc).lower()
        quota_hit = any(
            token in message
            for token in ("limit", "quota", "maximum", "exceed", "too many", "plan")
        )
        print(
            f"  [ERROR] create_index('{index_name}') failed: {exc!r}", file=sys.stderr
        )
        if quota_hit:
            print(
                "\n  >>> Looks like the free-tier index limit was hit.\n"
                "  >>> DOWNGRADE PATH: pack profiles into the content index instead.\n"
                "  >>> Re-run with:\n"
                "  >>>     ANSIO_MEMORY_INDEX=ansio_content "
                "uv run python src/build_memory_index.py\n"
                "  >>> And at runtime export ANSIO_MEMORY_INDEX=ansio_content.",
                file=sys.stderr,
            )
        else:
            print(
                "  >>> Not obviously a quota error. Check credentials/connectivity, "
                "then retry.",
                file=sys.stderr,
            )
        sys.exit(3)

    job_id = getattr(result, "job_id", "?")
    doc_count = getattr(result, "doc_count", "?")
    print(f"  Created '{index_name}' (job={job_id}, docs={doc_count}).")


async def _roundtrip(client: MossClient, index_name: str) -> bool:
    """add -> reload -> query -> assert -> cleanup. Returns True on PASS."""
    print(f"\n[3/4] Roundtrip smoke test on '{index_name}'...")
    smoke_doc = _profile_doc(SMOKE_USER_ID, SMOKE_PROFILE_TEXT)

    # --- write the smoke profile (upsert by stable id) ---
    try:
        await client.add_docs(index_name, [smoke_doc])
        print(f"  add_docs OK (id={SMOKE_DOC_ID}).")
    except Exception as exc:
        print(f"  [FAIL] add_docs failed: {exc!r}", file=sys.stderr)
        return False

    # --- reload so the new doc is queryable ---
    try:
        await client.load_index(index_name)
        print("  load_index OK (index reloaded after write).")
    except Exception as exc:
        print(f"  [FAIL] load_index failed: {exc!r}", file=sys.stderr)
        await _cleanup(client, index_name)
        return False

    # --- query scoped to the smoke user, timed ---
    started = time.perf_counter()
    try:
        result = await client.query(
            index_name,
            "user profile",
            QueryOptions(top_k=1, filter=_user_filter(SMOKE_USER_ID)),
        )
    except Exception as exc:
        print(f"  [FAIL] query failed: {exc!r}", file=sys.stderr)
        await _cleanup(client, index_name)
        return False
    latency_ms = (time.perf_counter() - started) * 1000.0

    docs = getattr(result, "docs", []) or []
    if not docs:
        print("  [FAIL] query returned no docs for the smoke user.", file=sys.stderr)
        await _cleanup(client, index_name)
        return False

    top = docs[0]
    got_text = getattr(top, "text", "")
    got_meta = getattr(top, "metadata", {}) or {}
    score = getattr(top, "score", None)
    if got_text != SMOKE_PROFILE_TEXT:
        print(
            "  [FAIL] roundtrip text mismatch.\n"
            f"        expected: {SMOKE_PROFILE_TEXT!r}\n"
            f"        got:      {got_text!r}",
            file=sys.stderr,
        )
        await _cleanup(client, index_name)
        return False

    print(
        f"  PASS roundtrip: text matched, score={score}, "
        f"user_id={got_meta.get('user_id')}, doc_type={got_meta.get('doc_type')}, "
        f"query latency={latency_ms:.1f} ms."
    )

    await _cleanup(client, index_name)
    return True


async def _cleanup(client: MossClient, index_name: str) -> None:
    """Delete the smoke doc. Degrade gracefully if delete is unavailable."""
    print(f"\n[4/4] Cleaning up smoke doc '{SMOKE_DOC_ID}'...")
    try:
        await client.delete_docs(index_name, [SMOKE_DOC_ID])
        print("  delete_docs OK (smoke profile removed).")
    except Exception as exc:
        print(
            f"  [WARN] delete_docs failed ({exc!r}); writing empty-profile sentinel "
            "as downgrade so the doc reads as cleared.",
            file=sys.stderr,
        )
        try:
            await client.add_docs(
                index_name,
                [_profile_doc(SMOKE_USER_ID, EMPTY_PROFILE_SENTINEL)],
            )
            print("  Wrote sentinel-cleared smoke profile.")
        except Exception as exc2:
            print(
                f"  [WARN] sentinel write also failed ({exc2!r}); "
                "smoke doc may linger. Not fatal.",
                file=sys.stderr,
            )


async def main() -> None:
    index_name = os.getenv("ANSIO_MEMORY_INDEX", DEFAULT_MEMORY_INDEX)
    print("=" * 60)
    print("ANSIO profile (memory) index builder + smoke test")
    print(f"  target index: {index_name}")
    print(f"  env files:    {PRIMARY_ENV} , {LOCAL_ENV}")
    print("=" * 60)

    project_id, project_key = _require_credentials()
    client = MossClient(project_id, project_key)

    existing = await _list_existing(client)
    await _ensure_index(client, index_name, existing)
    passed = await _roundtrip(client, index_name)

    print("\n" + "=" * 60)
    if passed:
        print(f"RESULT: PASS — profile index '{index_name}' is live and roundtrips.")
        print("=" * 60)
        return
    print(f"RESULT: FAIL — roundtrip on '{index_name}' did not pass. See logs above.")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
