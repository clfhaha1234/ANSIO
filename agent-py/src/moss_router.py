"""Local-session router for the KOL index (cloud ingest-quota escape hatch).

Context (demo day): the Moss project's 50MB monthly ingest quota is exhausted
and the cloud ``ansio_kols`` rebuild died server-side holding its build lock —
the index cannot be recreated in the cloud today. ``MossClient.session()``
gives us the same Rust index fully **in-memory**: ``add_docs``/``query`` run
locally (~1-10ms, zero cloud calls, zero ingest quota). We embed ``kols.json``
locally with moss-minilm at first use and never call ``push_index``.

``MossRouter`` is a drop-in facade over ``MossClient`` for the Assistant: any
call targeting the KOL index is served by the local session index; every other
index (ansio_content / ansio_products — both healthy in the cloud, and the
memory profiles ride in ansio_content) passes straight through.

Bonus: this is Moss's flagship "on-device retrieval" capability — the HUD
latency numbers drop from cloud RTT to single-digit milliseconds.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from pathlib import Path

from moss import DocumentInfo, MossClient

logger = logging.getLogger("moss_router")

AGENT_DIR = Path(__file__).resolve().parent.parent
KOLS_PATH = AGENT_DIR / "kols.json"

# One local session per worker process (each LiveKit job is its own process).
# The build (~30s local embedding) runs in a DAEMON THREAD kicked off by
# prewarm_session(): LiveKit's process-initialize timeout (~10s) kills any
# prewarm that blocks, so prewarm must return instantly (learned live — a
# blocking prewarm made every job spawn fail). threading.Lock (not
# asyncio.Lock) so the builder thread and any event loop can share the guard;
# SessionIndex methods are plain to_thread wrappers, hence loop-agnostic.
_session = None
_build_lock = threading.Lock()
_build_started = threading.Event()  # idempotence gate for prewarm_session


# Local pool size: full kols.json (1088 docs) embeds in ~165s and saturates
# the CPU; the demo only needs the elite pool. Top-N by precomputed alpha with
# the demo protagonists force-included builds in well under a minute.
LOCAL_DOC_LIMIT = int(os.getenv("ANSIO_LOCAL_KOL_DOCS", "300"))
_MUST_HANDLES = {"theobennett1", "mikeynocode", "priyagrant", "malvaai"}


def _alpha_of(md: dict) -> float:
    for k in ("alpha", "alpha_score", "cs_value_score"):
        try:
            return float(md.get(k))
        except (TypeError, ValueError):
            continue
    return 0.0


def _load_kols_docs() -> list[DocumentInfo]:
    """kols.json -> top-N DocumentInfo list (demo protagonists always included)."""
    data = json.loads(KOLS_PATH.read_text(encoding="utf-8"))
    data.sort(
        key=lambda e: (
            (e.get("metadata") or {}).get("handle", "").lower() in _MUST_HANDLES,
            _alpha_of(e.get("metadata") or {}),
        ),
        reverse=True,
    )
    docs: list[DocumentInfo] = []
    for e in data[:LOCAL_DOC_LIMIT]:
        md = {str(k): str(v) for k, v in (e.get("metadata") or {}).items()}
        docs.append(DocumentInfo(id=str(e["id"]), text=str(e["text"]), metadata=md))
    return docs


def _build_session_sync(index_name: str):
    """Blocking, thread-safe, idempotent local-index build (own event loop)."""
    global _session
    with _build_lock:
        if _session is not None:
            return _session

        async def _go():
            client = MossClient(
                os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")
            )
            sess = await client.session(index_name)
            if sess.doc_count == 0:  # cloud index absent -> embed locally
                await sess.add_docs(_load_kols_docs())
            return sess

        sess = asyncio.run(_go())
        logger.info(
            "Local session index '%s' ready: %d docs (on-device)",
            index_name, sess.doc_count,
        )
        _session = sess
        return _session


async def _ensure_session(client: MossClient, index_name: str):
    """Async accessor: instant when built; otherwise waits off-loop for the
    builder thread (or builds itself) without ever blocking the event loop."""
    if _session is not None:
        return _session
    return await asyncio.to_thread(_build_session_sync, index_name)


def _localize_filter(options):
    """Adapt cloud-style filters for the local session parser.

    Live-verified: the local Rust parser accepts ``{"$and": [{"field": F,
    "condition": C}, ...]}`` but rejects a BARE ``{"field": F, "condition": C}``
    (it requires the top level to be ``$and``/``$or``). The agent's point
    lookups pass the bare form, so wrap it. Returns ``options`` untouched when
    there is nothing to adapt; never raises (a bad filter degrades to None,
    matching the agent's filter-degrade discipline).
    """
    try:
        filt = getattr(options, "filter", None)
        if isinstance(filt, dict) and "field" in filt and "condition" in filt:
            options.filter = {"$and": [filt]}
    except Exception:
        with contextlib.suppress(Exception):
            options.filter = None
    return options


class MossRouter:
    """Drop-in MossClient facade: KOL index -> local session, rest -> cloud."""

    def __init__(self, cloud: MossClient, kols_index: str) -> None:
        self._cloud = cloud
        self._kols = kols_index

    def __getattr__(self, name):
        # Full drop-in transparency: anything the router doesn't override
        # (incl. test-fake attributes like ``results_by_index``) delegates to
        # the wrapped cloud client. Only fires for attributes not found here.
        return getattr(self._cloud, name)

    # -- loading -----------------------------------------------------------
    # CLOUD-FIRST: the healthy path is the cloud index (teammate's project).
    # Only when the cloud KOL index fails (missing index / quota outage) does
    # the router arm the local on-device session as a zero-quota fallback.
    # The local build never blocks (daemon thread); queries during the build
    # window fail fast into the tools' existing degrade paths.
    async def load_index(self, name: str):
        try:
            return await self._cloud.load_index(name)
        except Exception:
            if name != self._kols:
                raise
            logger.warning("cloud load failed for %s; arming local fallback", name)
            prewarm_session(name)

    async def load_indexes(self, names: list[str]):
        try:
            return await self._cloud.load_indexes(names)
        except Exception:
            # Per-index retry so one bad index doesn't block the rest; the KOL
            # index additionally arms its local fallback.
            for n in names:
                with contextlib.suppress(Exception):
                    await self.load_index(n)

    # -- query -------------------------------------------------------------
    async def query(self, name: str, text: str, options=None):
        if name != self._kols:
            return await self._cloud.query(name, text, options)
        try:
            return await self._cloud.query(name, text, options)
        except Exception:
            if _session is None:
                prewarm_session(name)  # arm fallback; fail fast this turn
                raise
            return await _session.query(text, _localize_filter(options))

    # -- passthrough mutations (memory profiles live in ansio_content) ------
    async def add_docs(self, name: str, docs, options=None):
        return await self._cloud.add_docs(name, docs, options)

    async def delete_docs(self, name: str, doc_ids):
        return await self._cloud.delete_docs(name, doc_ids)

    async def list_indexes(self):
        return await self._cloud.list_indexes()


def prewarm_session(index_name: str | None = None) -> None:
    """Kick off the local KOL index build in a daemon thread (returns instantly).

    Called from agent.py's ``prewarm``. MUST NOT block: LiveKit kills job
    processes whose initialize exceeds its timeout (~10s), and the build takes
    ~30s. The thread builds in the background; a call arriving earlier waits in
    ``_ensure_session`` (off-loop) for the same lock instead of re-building.
    """
    name = index_name or os.getenv("ANSIO_KOLS_INDEX", "ansio_kols")
    if _session is not None or _build_started.is_set():
        return  # ready or already building -> no thread pile-up
    _build_started.set()

    def _run() -> None:
        try:
            _build_session_sync(name)
        except Exception:
            _build_started.clear()  # allow a retry on the next trigger
            logger.exception("local index prewarm build failed (lazy path remains)")

    threading.Thread(target=_run, daemon=True, name="moss-local-index-build").start()
