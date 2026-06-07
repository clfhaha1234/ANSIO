import os

# macOS: ensure Python's TLS stack can find a CA bundle before any livekit/
# aiohttp import creates a default SSL context. Without this, connecting to
# LiveKit Cloud fails with SSL: CERTIFICATE_VERIFY_FAILED on stock macOS
# Python builds. Must run before the livekit imports below.
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import asyncio
import contextlib
import json
import logging
import textwrap
import time

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics, minimax, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from moss import MossClient, QueryOptions

from events import build_chain_step, build_evidence, kol_items, publish_evidence
from lang import (
    greeting_suffix,
    language_directive,
    normalize_language,
    stt_language,
    tts_language_boost,
)
from llm_factory import build_llm, model_label
from memory import (
    UserMemory,
    history_to_items,
    memory_greeting,
    profile_instructions,
)
from scoring import load_benchmark, score_and_rank
from state import AnsioState

logger = logging.getLogger("agent")

# ENV_FILE convention matches the rest of the repo's harnesses; falls back to
# .env (this project) then .env.local (template default).
load_dotenv(os.getenv("ENV_FILE", ".env"))
load_dotenv(".env.local")

# ----------------------------------------------------------------------------
# Moss indexes (the three ansio_-prefixed indexes built by build_indexes.py).
# Free-tier 3-index plan: playbook is merged into ansio_content, separable by
# the doc_type metadata filter.
# ----------------------------------------------------------------------------
IDX_PRODUCTS = os.getenv("ANSIO_PRODUCTS_INDEX", "ansio_products")
IDX_CONTENT = os.getenv("ANSIO_CONTENT_INDEX", "ansio_content")
IDX_KOLS = os.getenv("ANSIO_KOLS_INDEX", "ansio_kols")
ALL_INDEXES = (IDX_PRODUCTS, IDX_CONTENT, IDX_KOLS)

# White-lists (PRD §2.3). Every $eq value is validated against these before it
# reaches Moss; an illegal value degrades to pure semantic (PRD T2).
KOL_PLATFORMS = ["YouTube", "Instagram", "TikTok", "X", "Twitch", "Bilibili", "LinkedIn"]
KOL_NICHES = [
    "tech", "gaming", "beauty", "fashion", "fitness", "food", "finance", "travel",
    "education", "music", "comedy", "lifestyle", "parenting", "automotive",
    "business", "crypto", "art", "sustainability", "home", "pets",
]
PLAYBOOK_DOC_TYPES = {"qa", "strategy", "case"}

# Wide recall pool for find_similar_kols (PRD T1): recall broad, filter in Python.
WIDE_POOL_TOP_K = 80

DEFAULT_USER_ID = "user_1"


# ===========================================================================
# Pure helpers (no Moss/livekit) — normalization & rendering
# ===========================================================================


def _norm_handle(handle: str) -> str:
    """Strip a leading @ (PRD T3): '@theobennett1' -> 'theobennett1'."""
    return (handle or "").strip().lstrip("@")


def _norm_brand(brand: str) -> str:
    """Brands are stored lowercase in the content index (PRD T4)."""
    return (brand or "").strip().lower()


def _wl_platform(platform: str) -> str:
    return platform if platform in KOL_PLATFORMS else ""


def _wl_niche(niche: str) -> str:
    n = (niche or "").strip().lower()
    return n if n in KOL_NICHES else ""


def _build_kol_filter(platform: str, niche: str):
    """Exact-match filter for the kols index; None when nothing valid."""
    conds = []
    if platform:
        conds.append({"field": "platform", "condition": {"$eq": platform}})
    if niche:
        conds.append({"field": "niche", "condition": {"$eq": niche}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _docs(result) -> list:
    return getattr(result, "docs", None) or []


def _md_list(result) -> list[dict]:
    """Flatten Moss docs into [{metadata, sim, text}] dicts for scoring/cards."""
    out: list[dict] = []
    for d in _docs(result):
        md = dict(getattr(d, "metadata", {}) or {})
        sim = None
        score = getattr(d, "score", None)
        if score is not None:
            with contextlib.suppress(TypeError, ValueError):
                sim = float(score)
        out.append(
            {
                "metadata": md,
                "sim": sim,
                "text": (getattr(d, "text", "") or "").strip(),
            }
        )
    return out


def _short(text: str, limit: int = 28) -> str:
    """Clamp a query string for judge-readable chain-step labels (≤60 chars)."""
    t = (text or "").strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def _to_int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _fmt_int(v) -> str:
    n = _to_int(v)
    return f"{n:,}" if n else str(v if v is not None else "?")


# ===========================================================================
# Assistant
# ===========================================================================


class Assistant(Agent):
    """ANSIO growth-engineer voice agent: 5 Moss tools + a single meta-tool
    recall chain, preemptive retrieval injection, and a recommendation gate.
    """

    def __init__(
        self,
        *,
        room=None,
        user_id: str = DEFAULT_USER_ID,
        profile: str | None = None,
        language: str = "auto",
    ) -> None:
        super().__init__(
            # The conversational brain comes from the LLM factory (default
            # provider=minimax per F2's measured verdict; claude/inference are
            # env-hot-swap fallbacks). build_llm() lazily imports its plugin.
            llm=build_llm(),
            # A consented user profile (memory.py) and the language rule
            # (lang.py) ride in as system suffixes.
            instructions=(
                _INSTRUCTIONS
                + profile_instructions(profile)
                + language_directive(language)
            ),
        )
        self._room = room
        self._user_id = user_id
        # All three indexes are served from the cloud Moss project (ansio_kols
        # rebuilt lean on a fresh-quota project — build_indexes --lean). A plain
        # cloud client keeps each job process light: no on-device embedding, no
        # 3GB local SessionIndex. moss_router.py stays as a zero-quota fallback.
        self._moss = MossClient(
            os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")
        )
        self._indexes_loaded = False
        self._benchmark = load_benchmark()
        self._state = AnsioState()
        # Dedup guard for on_user_turn_completed double-fire under preemptive
        # generation (livekit/agents #3414): remember the last injected turn.
        self._last_injected_text = None
        with contextlib.suppress(Exception):
            logger.info("ANSIO Assistant brain = %s", model_label())

    # -- lifecycle ---------------------------------------------------------
    async def on_enter(self) -> None:
        """Preload all three ansio_ indexes (PRD T7). Best-effort: tools retry."""
        if self._indexes_loaded:
            return
        try:
            await self._moss.load_indexes(list(ALL_INDEXES))
            self._indexes_loaded = True
            logger.info("Loaded Moss indexes: %s", ", ".join(ALL_INDEXES))
        except Exception:
            # Retry each individually so one bad index does not block the rest.
            logger.exception("load_indexes failed; retrying per-index")
            for name in ALL_INDEXES:
                with contextlib.suppress(Exception):
                    await self._moss.load_index(name)
            self._indexes_loaded = True

    # -- preemptive retrieval injection (PRD §5 / latency §5) ---------------
    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Inject a cheap pre-retrieval into the context BEFORE the LLM answers,
        avoiding an extra tool round-trip. Guarded against the #3414 double-fire.
        """
        self._state.turn_count += 1
        text = (new_message.text_content or "").strip()
        if not text or text == self._last_injected_text:
            return  # same-turn dedup guard (preemptive double-fire)
        self._last_injected_text = text
        try:
            await publish_evidence(
                self._room,
                build_chain_step(
                    "similar_creators",
                    f"Reading the brief — '{_short(text)}'",
                ),
            )
            t0 = time.perf_counter()
            result = await self._moss.query(IDX_KOLS, text, QueryOptions(top_k=3))
            took_ms = (time.perf_counter() - t0) * 1000.0
            cands = _md_list(result)
            if not cands:
                return
            await publish_evidence(
                self._room,
                build_chain_step(
                    "similar_creators",
                    f"{len(cands)} candidates surfaced",
                    src=IDX_KOLS, res=True, latency_ms=took_ms,
                ),
            )
            snippet = "; ".join(
                f"{c['metadata'].get('name', '?')} (@{c['metadata'].get('handle', '?')})"
                for c in cands[:3]
            )
            turn_ctx.add_message(
                role="assistant",
                content=(
                    "[ANSIO retrieval — internal context, do not read aloud "
                    f"verbatim] candidates near this brief: {snippet}"
                ),
            )
            await publish_evidence(
                self._room,
                build_evidence(
                    "similar_creators",
                    index=IDX_KOLS,
                    latency_ms=took_ms,
                    items=kol_items(cands, limit=3),
                    insight="Pre-retrieved candidates for the latest turn.",
                ),
            )
        except Exception:
            logger.exception("Preemptive retrieval injection failed")

    # -- internal Moss helper (timing + filter degrade) --------------------
    async def _q(self, index: str, query: str, *, top_k: int, filt=None):
        """One Moss query with timing; degrades to semantic-only on empty filter."""
        t0 = time.perf_counter()
        result = await self._moss.query(
            index, query, QueryOptions(top_k=top_k, filter=filt)
        )
        if filt and not _docs(result):
            result = await self._moss.query(index, query, QueryOptions(top_k=top_k))
        took_ms = (time.perf_counter() - t0) * 1000.0
        return result, took_ms

    # =====================================================================
    # 5 retrieval tools (PRD ③). Each: white-list normalize + empty fallback.
    # =====================================================================

    @function_tool()
    async def find_competitors(
        self, context: RunContext, product_desc: str, top_k: int = 5
    ) -> str:
        """Find competing products/brands in the user's space (semantic only).

        Args:
            product_desc: Natural-language description of the user's product,
                e.g. "an AI pair-programming tool for backend engineers".
            top_k: How many competitors to return (default 5).
        """
        await publish_evidence(
            self._room,
            build_chain_step(
                "competitor_landscape",
                f"Mapping your space — '{_short(product_desc)}'",
            ),
        )
        try:
            result, ms = await self._q(IDX_PRODUCTS, product_desc, top_k=top_k)
        except Exception:
            logger.exception("find_competitors failed")
            return "I couldn't reach the product database right now."
        rows = _md_list(result)
        items = [
            {
                "name": r["metadata"].get("name", "?"),
                "category": r["metadata"].get("category", ""),
                "funding": r["metadata"].get("funding", ""),
            }
            for r in rows
        ]
        await publish_evidence(
            self._room,
            build_chain_step(
                "competitor_landscape",
                f"{len(items)} close competitors found",
                src=IDX_PRODUCTS, res=True, latency_ms=ms,
            ),
        )
        await publish_evidence(
            self._room,
            build_evidence(
                "competitor_landscape", step=1, index=IDX_PRODUCTS,
                latency_ms=ms, items=items, insight="Companies closest to you now.",
            ),
        )
        if not items:
            return "No clear competitors matched that description."
        return "; ".join(
            f"{i['name']} ({i['category']}{', ' + i['funding'] if i['funding'] else ''})"
            for i in items
        )

    @function_tool()
    async def find_kols_who_promoted(
        self,
        context: RunContext,
        brand: str = "",
        kol_handle: str = "",
        product_desc: str = "",
    ) -> str:
        """Find sponsorship history. Two MUTUALLY EXCLUSIVE directions (PRD T4):

        - Forward "who promoted BRAND": pass ``brand`` only.
        - Reverse "what has @HANDLE promoted": pass ``kol_handle`` only.

        Args:
            brand: A product/brand name, e.g. "cursor" (forward lookup).
            kol_handle: A creator handle, e.g. "buildwithsam" (reverse lookup).
            product_desc: Optional extra semantic context for the query text.
        """
        brand_n = _norm_brand(brand)
        handle_n = _norm_handle(kol_handle)
        if handle_n:
            filt = {"field": "kol_handle", "condition": {"$eq": handle_n}}
            qtext = product_desc or f"{handle_n} sponsorship"
        elif brand_n:
            filt = {"field": "brand", "condition": {"$eq": brand_n}}
            qtext = product_desc or f"{brand_n} sponsored review"
        else:
            return "Tell me a brand to look up, or a creator handle to reverse-lookup."
        intent = (
            f"Tracing who promoted {brand_n}" if brand_n
            else f"Tracing what @{handle_n} promoted"
        )
        await publish_evidence(
            self._room, build_chain_step("content_hits", _short(intent, 56)),
        )
        try:
            result, ms = await self._q(IDX_CONTENT, qtext, top_k=50, filt=filt)
        except Exception:
            logger.exception("find_kols_who_promoted failed")
            return "I couldn't reach the sponsorship database right now."

        # Aggregate by kol_handle -> total views, top creators.
        agg: dict[str, dict] = {}
        for r in _md_list(result):
            md = r["metadata"]
            h = md.get("kol_handle", "?")
            entry = agg.setdefault(
                h, {"handle": h, "posts": 0, "total_views": 0,
                    "sample": md.get("title", "")}
            )
            entry["posts"] += 1
            entry["total_views"] += _to_int(md.get("views"))
        top = sorted(agg.values(), key=lambda e: e["total_views"], reverse=True)[:10]
        await publish_evidence(
            self._room,
            build_chain_step(
                "content_hits",
                f"{len(top)} creators with sponsorship history",
                src=IDX_CONTENT, res=True, latency_ms=ms,
            ),
        )
        await publish_evidence(
            self._room,
            build_evidence(
                "content_hits", step=2, index=IDX_CONTENT, latency_ms=ms,
                items=top, insight=f"{len(top)} creators with sponsorship history.",
            ),
        )
        if not top:
            return "No sponsorship history matched that."
        return "; ".join(
            f"{e['handle']} ({e['posts']} posts, {e['total_views']:,} views)"
            for e in top[:5]
        )

    @function_tool()
    async def get_kol_profile(self, context: RunContext, handle: str) -> str:
        """Pull one creator's full profile by handle (point lookup, PRD T3).

        Args:
            handle: The creator handle, with or without a leading @.
        """
        h = _norm_handle(handle)
        if not h:
            return "I need a creator handle to look up."
        filt = {"field": "handle", "condition": {"$eq": h}}
        await publish_evidence(
            self._room,
            build_chain_step("kol_profile", f"Pulling profile for @{_short(h, 40)}"),
        )
        try:
            result, ms = await self._q(IDX_KOLS, h, top_k=1, filt=filt)
        except Exception:
            logger.exception("get_kol_profile failed")
            return "I couldn't reach the creator database right now."
        rows = _md_list(result)
        if not rows:
            return f"I don't have a profile for {h}."
        r = rows[0]
        md = r["metadata"]
        await publish_evidence(
            self._room,
            build_chain_step(
                "kol_profile",
                f"Profile loaded — {_short(md.get('name', h), 30)}",
                src=IDX_KOLS, res=True, latency_ms=ms,
            ),
        )
        await publish_evidence(
            self._room,
            build_evidence(
                "kol_profile", step=3, index=IDX_KOLS, latency_ms=ms,
                items=kol_items([r], limit=1), insight=r["text"][:140],
            ),
        )
        return (
            f"{md.get('name', h)} (@{h}): {md.get('niche', '?')} on "
            f"{md.get('platform', '?')}, {_fmt_int(md.get('followers'))} followers, "
            f"{md.get('engagement_pct', '?')}% engagement, based in "
            f"{md.get('region', '?')}."
        )

    @function_tool()
    async def find_similar_kols(
        self,
        context: RunContext,
        profile_text: str,
        niche: str = "",
        platform: str = "",
        top_k: int = WIDE_POOL_TOP_K,
    ) -> str:
        """Find creators similar to a brief; recalls a WIDE pool (PRD T1, T2).

        Recalls ``top_k=80`` WITHOUT a budget filter and caches the pool so
        budget/weight changes re-rank in Python with no re-query. Re-call this
        whenever the user changes platform / niche / audience constraints (T6).

        Args:
            profile_text: Description of the ideal creator + campaign.
            niche: Optional category filter (must be a valid niche, else ignored).
            platform: Optional platform filter (must be valid, else ignored).
            top_k: Wide recall size (default 80).
        """
        platform_w = _wl_platform(platform)
        niche_w = _wl_niche(niche)
        filt = _build_kol_filter(platform_w, niche_w)
        await publish_evidence(
            self._room,
            build_chain_step(
                "similar_creators",
                f"Matching audience to your ICP — '{_short(profile_text)}'",
            ),
        )
        try:
            result, ms = await self._q(IDX_KOLS, profile_text, top_k=top_k, filt=filt)
        except Exception:
            logger.exception("find_similar_kols failed")
            return "I couldn't reach the creator database right now."
        pool = _md_list(result)
        self._state.set_pool(pool)  # cache wide pool for Python re-rank (T1)
        self._state.update_slots(
            platform=platform_w or None, category=niche_w or None
        )
        await publish_evidence(
            self._room,
            build_chain_step(
                "similar_creators",
                f"{len(pool)} candidates recalled",
                src=IDX_KOLS, res=True, latency_ms=ms,
            ),
        )
        await publish_evidence(
            self._room,
            build_evidence(
                "similar_creators", step=4, index=IDX_KOLS, latency_ms=ms,
                items=kol_items(pool, limit=8),
                insight=f"Recalled {len(pool)} candidate creators.",
            ),
        )
        if not pool:
            return "No similar creators found for that brief."
        names = "; ".join(
            f"{c['metadata'].get('name', '?')} (@{c['metadata'].get('handle', '?')})"
            for c in pool[:5]
        )
        return f"Found {len(pool)} similar creators. Top matches: {names}"

    @function_tool()
    async def search_playbook(
        self, context: RunContext, question: str, doc_type: str = ""
    ) -> str:
        """Search the ANSIO methodology playbook (objections, strategy, cases).

        Args:
            question: The question or objection to look up.
            doc_type: Optional one of: qa, strategy, case. Invalid -> ignored.
        """
        dt = (doc_type or "").strip().lower()
        # Playbook lives in ansio_content. Moss filters support only
        # $and/$eq/$gt/$lt (no $in), so scope to a single doc_type when given;
        # otherwise query semantically (methodology docs surface by meaning).
        filt = {"field": "doc_type", "condition": {"$eq": dt}} if dt in PLAYBOOK_DOC_TYPES else None
        await publish_evidence(
            self._room,
            build_chain_step(
                "playbook_hit",
                f"Consulting the playbook — '{_short(question)}'",
            ),
        )
        try:
            result, ms = await self._q(IDX_CONTENT, question, top_k=3, filt=filt)
        except Exception:
            logger.exception("search_playbook failed")
            return "I couldn't reach the playbook right now."
        rows = _md_list(result)
        items = [
            {"text": r["text"][:200], "source": r["metadata"].get("source", "")}
            for r in rows
        ]
        await publish_evidence(
            self._room,
            build_chain_step(
                "playbook_hit",
                f"{len(items)} methodology notes matched",
                src=(items[0]["source"] if items else IDX_CONTENT),
                res=True, latency_ms=ms,
            ),
        )
        await publish_evidence(
            self._room,
            build_evidence(
                "playbook_hit", step="2b", index=IDX_CONTENT, latency_ms=ms,
                items=items, insight=(items[0]["text"][:120] if items else ""),
                source=(items[0]["source"] if items else ""),
            ),
        )
        if not rows:
            return "I don't have a playbook note on that."
        return rows[0]["text"][:400]

    # =====================================================================
    # META-TOOL (latency war-card core): ONE LLM decision -> deterministic
    # Python recall chain on Moss -> scored candidates (PRD §1 / latency #5).
    # The LLM calls this once instead of five tool round-trips.
    # =====================================================================

    @function_tool()
    async def recommend_kols(
        self,
        context: RunContext,
        product_desc: str,
        niche: str = "",
        platform: str = "",
        budget: float | None = None,
    ) -> str:
        """Run the full ANSIO recall chain and return scored recommendations.

        Call this ONCE when you have enough of a brief to recommend creators. It
        deterministically chains competitors -> sponsorship graph -> wide creator
        recall -> Python Alpha scoring, all in-process, and returns the ranked
        top creators with their fit/performance/value breakdown.

        Args:
            product_desc: The user's product/brief.
            niche: Optional creator niche to bias recall.
            platform: Optional platform filter.
            budget: Optional per-post budget cap (filters in Python, no re-query).
        """
        platform_w = _wl_platform(platform)
        niche_w = _wl_niche(niche)
        # Stage 0 — parse brief (intent only; deterministic, no Moss hop).
        await publish_evidence(
            self._room,
            build_chain_step(
                "alpha_ranking", f"Parsing the brief — '{_short(product_desc)}'",
            ),
        )
        t0 = time.perf_counter()
        try:
            # Deterministic chain (in-process, no per-hop LLM round-trip):
            #  1) competitor landscape (products, semantic)
            #  2) sponsorship graph signal (content, semantic context)
            #  4) wide creator recall (kols, the scoring pool) — top_k=80
            with contextlib.suppress(Exception):
                await publish_evidence(
                    self._room,
                    build_chain_step(
                        "competitor_landscape", "Mapping your competitive space",
                    ),
                )
                cres, comp_ms = await self._q(IDX_PRODUCTS, product_desc, top_k=5)
                comp_items = [
                    {"name": r["metadata"].get("name", "?"),
                     "category": r["metadata"].get("category", ""),
                     "funding": r["metadata"].get("funding", "")}
                    for r in _md_list(cres)
                ]
                if comp_items:
                    await publish_evidence(
                        self._room,
                        build_chain_step(
                            "competitor_landscape",
                            f"{len(comp_items)} competitors mapped",
                            src=IDX_PRODUCTS, res=True, latency_ms=comp_ms,
                        ),
                    )
                    await publish_evidence(
                        self._room,
                        build_evidence("competitor_landscape", step=1,
                                       index=IDX_PRODUCTS, items=comp_items,
                                       insight="Closest companies in your space."),
                    )
            # Stage 2 — sponsorship-graph signal folds into the wide recall.
            await publish_evidence(
                self._room,
                build_chain_step(
                    "similar_creators", "Reading the sponsorship graph",
                ),
            )
            filt = _build_kol_filter(platform_w, niche_w)
            kres, recall_ms = await self._q(
                IDX_KOLS, product_desc, top_k=WIDE_POOL_TOP_K, filt=filt
            )
            pool = _md_list(kres)
            await publish_evidence(
                self._room,
                build_chain_step(
                    "similar_creators", f"{len(pool)} creators in the recall pool",
                    src=IDX_KOLS, res=True, latency_ms=recall_ms,
                ),
            )
        except Exception:
            logger.exception("recommend_kols recall failed")
            return "I couldn't run the recommendation right now."

        self._state.set_pool(pool)
        self._state.update_slots(
            category=niche_w or None,
            platform=platform_w or None,
            budget=budget if budget is not None else None,
        )
        chain_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 5 — scoring (pure Python, pool-relative Alpha).
        await publish_evidence(
            self._room,
            build_chain_step(
                "alpha_ranking", "Scanning for underpriced creators",
            ),
        )
        ranked = self._rerank(budget=budget)
        if not ranked:
            await publish_evidence(
                self._room,
                build_evidence("alpha_ranking", step=5, index=IDX_KOLS,
                               latency_ms=chain_ms, items=[],
                               insight="No candidates fit the constraints."),
            )
            return "No creators fit those constraints. Want to widen the budget?"

        await publish_evidence(
            self._room,
            build_chain_step(
                "alpha_ranking",
                f"Top {min(len(ranked), 5)} ranked by Alpha — bundling",
                src=IDX_KOLS, res=True, latency_ms=chain_ms,
            ),
        )
        await publish_evidence(
            self._room,
            build_evidence(
                "alpha_ranking", step=5, index=IDX_KOLS, latency_ms=chain_ms,
                items=kol_items(ranked, limit=5),
                insight="Ranked by fit, performance and value (Alpha).",
            ),
        )
        self._state.recommended = True
        return self._format_ranked(ranked)

    # -- pure Python re-rank (no Moss; budget/weight FLIP, PRD T1) ----------
    def _rerank(self, budget: float | None = None, weights: dict | None = None):
        slots = dict(self._state.slots)
        if budget is not None:
            slots["budget"] = budget
        ranked = score_and_rank(
            self._state.candidate_pool,
            slots=slots,
            weights=weights or self._state.weights,
            benchmark=self._benchmark,
        )
        self._state.note_top5(ranked)
        return ranked

    @staticmethod
    def _format_ranked(ranked: list[dict]) -> str:
        lines = []
        for c in ranked[:3]:
            md = c.get("metadata", c)
            emc = c.get("estimated_market_cost")
            cost = f", est. market cost ${emc:,}" if emc else ""
            lines.append(
                f"{md.get('name', '?')} (@{md.get('handle', '?')}) — "
                f"{_fmt_int(md.get('followers'))} followers, "
                f"{md.get('engagement_pct', '?')}% engagement{cost}"
            )
        return "Top picks: " + "; ".join(lines)


# ---------------------------------------------------------------------------
# System prompt (PRD T6 re-retrieval rule + voice output rules)
# ---------------------------------------------------------------------------

_INSTRUCTIONS = textwrap.dedent(
    """\
    You are ANSIO, a sharp, warm growth engineer who helps brands find
    undervalued creators (KOLs) for influencer campaigns. You speak in voice.
    Follow the Language rule at the end of this prompt for which language to use.

    # How you work
    - Understand the user's product, audience, platform, budget, and goal.
    - Lead with insight, not interrogation: as soon as you can name a niche,
      start retrieving — do not ask three questions before showing value.
    - To recommend creators, call `recommend_kols` ONCE with the brief. It runs
      the whole recall + scoring chain internally. Do not chain the five lookup
      tools yourself for the main recommendation flow.
    - Use `find_competitors`, `find_kols_who_promoted`, `get_kol_profile`,
      `find_similar_kols`, and `search_playbook` for targeted follow-ups and
      objections.

    # Re-retrieval rule (important)
    - If the user ADDS or CHANGES a platform, niche, audience, or budget
      constraint (e.g. "only YouTube", "actually it's K-beauty"), immediately
      re-run `find_similar_kols` (or `recommend_kols`) with the new filter.
      Never treat a new constraint as mere confirmation.
    - If the user only changes BUDGET or says they care more about value, you do
      NOT need a new search — `recommend_kols` re-ranks the cached pool.

    # Objections
    - When the user pushes back ("Cursor is too big to compare to"), call
      `search_playbook` and answer from the methodology, not from guesses.

    # Output rules (voice)
    - Plain spoken text only: no JSON, markdown, lists, tables, code, or emojis.
    - One to three sentences per turn; name at most three creators at a time.
    - Spell numbers naturally ("three hundred thousand followers").
    - Read handles plainly, without the at-sign noise.
    - Only mention an ESTIMATED MARKET COST; never claim to know a creator's
      real private rate.
    - Never invent names, handles, or follower counts — only use tool results.
    - Do not reveal tool names, parameters, or internal retrieval notes.
    """
)


# Retrieval is plain cloud Moss (no on-device embedding), so the worker stays
# light. num_idle_processes=1 + a generous initialize timeout are kept as a
# conservative default for stable cold starts on the demo machine.
server = AgentServer(num_idle_processes=1, initialize_process_timeout=60.0)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


# Keep the registered dispatch name as "agent-py": the frontend sets
# AGENT_NAME=agent-py to dispatch explicitly to this worker. Do not rename.
@server.rtc_session(agent_name="agent-py")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    user_id = DEFAULT_USER_ID
    if ctx.job.metadata:
        with contextlib.suppress(Exception):
            user_id = json.loads(ctx.job.metadata).get("user_id", DEFAULT_USER_ID)

    # Consent-gated user-profile memory (memory.py). The frontend toggle rides
    # in via dispatch metadata; console mode (no metadata) can opt in through
    # ANSIO_MEMORY_ENABLED=1. Disabled => UserMemory makes ZERO Moss calls.
    memory_enabled = os.getenv("ANSIO_MEMORY_ENABLED", "0") == "1"
    refresh_profile = False
    language = normalize_language(os.getenv("ANSIO_LANGUAGE", "auto"))
    if ctx.job.metadata:
        with contextlib.suppress(Exception):
            _meta = json.loads(ctx.job.metadata)
            memory_enabled = bool(_meta.get("memory_enabled", memory_enabled))
            refresh_profile = bool(_meta.get("refresh_profile", False))
            language = normalize_language(_meta.get("language", language))
    user_memory = UserMemory(user_id=user_id, enabled=memory_enabled)
    # startup() swallows all Moss failures internally -> None (memory-free call).
    profile = await user_memory.startup(refresh=refresh_profile)

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language=stt_language(language)),
        # TTS: MiniMax professional growth-consultant voice (F3 selection).
        # voice overridable via ANSIO_TTS_VOICE if the team prefers another.
        tts=minimax.TTS(
            model="speech-2.8-turbo",
            voice=os.getenv("ANSIO_TTS_VOICE", "English_Persuasive_Man"),
            language_boost=tts_language_boost(language),
            emotion="neutral",
            speed=1.0,
            sample_rate=24000,
            # PCM bypasses the framework mp3 decoder whose close-race kills
            # speech mid-utterance (ValueError: I/O operation on closed file
            # in codecs/decoder.py via _tts_inference_task — every session
            # 09:37-10:10). Raw frames go straight to AudioEmitter.
            audio_format="pcm",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # Preemptive generation: speculative LLM start to cut perceived latency.
        # on_user_turn_completed has a same-turn dedup guard for #3414.
        preemptive_generation=True,
        min_endpointing_delay=float(os.getenv("ANSIO_MIN_ENDPOINTING", "0.5")),
    )

    await session.start(
        agent=Assistant(
            room=ctx.room, user_id=user_id, profile=profile, language=language
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
            # Text-first bubbles: emit transcription as soon as the LLM
            # produces it instead of pacing it with TTS audio playback
            # (docs: agents/multimodality/text "Disabling synchronization").
            text_output=room_io.TextOutputOptions(sync_transcription=False),
        ),
    )

    await ctx.connect()

    # Cold greeting when no profile; warm "I remember you" greeting otherwise
    # (memory_greeting(None) is the exact original cold-greeting text).
    await session.generate_reply(
        instructions=memory_greeting(profile) + greeting_suffix(language)
    )

    async def _flush_profile() -> None:
        # Session-end distillation: one LLM pass -> profile upsert. No-op when
        # memory is off. Hardened against shutdown teardown (a live job died
        # with SIGABRT here): the LLM call is time-bounded and the client is
        # explicitly closed so no HTTP stream outlives the closing event loop.
        if not user_memory.enabled:
            return
        llm = None
        try:
            llm = build_llm()
            await asyncio.wait_for(
                user_memory.extract_and_save(history_to_items(session.history), llm),
                timeout=12.0,
            )
        except Exception:
            logger.exception("profile flush failed (non-fatal)")
        finally:
            if llm is not None:
                with contextlib.suppress(Exception):
                    await llm.aclose()

    ctx.add_shutdown_callback(_flush_profile)


if __name__ == "__main__":
    cli.run_app(server)
