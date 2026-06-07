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
from livekit.agents import (
    tts as lk_tts,
)
from livekit.plugins import ai_coustics, minimax, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from moss import MossClient, QueryOptions

from events import EVIDENCE_TYPES, build_evidence, kol_items, publish_evidence
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
from moss_router import MossRouter
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


def _to_int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _fmt_int(v) -> str:
    n = _to_int(v)
    return f"{n:,}" if n else str(v if v is not None else "?")


def _looks_like_retrieval_turn(text: str) -> bool:
    """Cheap guard for pre-retrieval: search only when the turn has demo intent."""
    t = (text or "").strip().lower()
    if not t:
        return False
    smalltalk = {
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "好的",
        "谢谢", "你好", "您好", "明白", "可以", "没问题",
    }
    if t in smalltalk:
        return False
    retrieval_markers = (
        "ai coding", "coding tool", "developer", "indie hacker", "builder",
        "creator", "kol", "influencer", "campaign", "growth", "user growth",
        "competitor", "cursor", "copilot", "replit", "codeium", "youtube",
        "twitter", "x ", "audience", "budget", "roi", "达人", "创作者",
        "增长", "用户", "竞品", "对标", "投放", "营销", "开发者", "独立开发者",
        "预算", "转化", "合作", "报价",
    )
    return any(marker in t for marker in retrieval_markers)


def _llm_card_items(items_json: str, limit: int = 6) -> list[dict]:
    """Parse and sanitize LLM-authored card items before they hit the UI."""
    if not items_json:
        return []
    try:
        raw = json.loads(items_json)
    except json.JSONDecodeError:
        return []
    if isinstance(raw, dict):
        raw = raw.get("items", [])
    if not isinstance(raw, list):
        return []

    allowed = {
        "title", "name", "handle", "platform", "followers", "niche", "region",
        "sub", "source", "reason", "why", "takeaway", "score", "alpha", "match_score",
        "perf_score", "total_score", "engagement_pct", "estimated_market_cost",
        "posts", "total_views",
    }
    out: list[dict] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        clean = {}
        for key, value in item.items():
            if key not in allowed or value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = str(value)[:220] if isinstance(value, str) else value
        if clean:
            out.append(clean)
    return out


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
            # provider=inference / openai/gpt-4.1-mini; minimax and claude are
            # env-hot-swap alternatives). build_llm() lazily imports its plugin.
            llm=build_llm(),
            # A consented user profile (memory.py) and the language rule
            # (lang.py) ride in as system suffixes on the 8-step story prompt.
            instructions=(
                _INSTRUCTIONS
                + profile_instructions(profile)
                + language_directive(language)
            ),
        )
        self._room = room
        self._user_id = user_id
        # KOL traffic can route to a local Moss session index when the cloud
        # project is unavailable (see moss_router.py); content/products stay
        # on the cloud client unchanged.
        self._moss = MossRouter(
            MossClient(os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")),
            IDX_KOLS,
        )
        self._indexes_loaded = False
        self._benchmark = load_benchmark()
        self._state = AnsioState()
        self._turn_card_types: set[str] = set()
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
        self._turn_card_types.clear()
        if not _looks_like_retrieval_turn(text):
            return
        try:
            result = await self._moss.query(IDX_KOLS, text, QueryOptions(top_k=3))
            cands = _md_list(result)
            if not cands:
                return
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
            # Keep pre-retrieval invisible: it is context for lower latency, not a
            # right-rail demo step. Visible cards should advance only when the
            # LLM/tool route intentionally selects that step.
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
        """STEP 1 — Competitor Discovery.

        Use after a new product/growth brief, especially for "AI coding tool",
        "who should we benchmark", or "what companies already ran this motion".
        This should drive a right-rail "Competitor Landscape" card.

        Args:
            product_desc: Natural-language description of the user's product,
                e.g. "an AI pair-programming tool for backend engineers".
            top_k: How many competitors to return (default 5).
        """
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
                "stage": r["metadata"].get("stage", ""),
                # Company intro (公司介绍) from the product doc text + similarity,
                # so the competitor card explains what each company actually is.
                "desc": (r.get("text") or "")[:160],
                "sim": r.get("sim"),
            }
            for r in rows
        ]
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
        """STEP 2 — Competitor Campaign / Partnership Analysis.

        Use when the founder asks who a competitor worked with, what Cursor or
        Codeium did on Twitter/YouTube, or which creator campaigns performed.
        Two MUTUALLY EXCLUSIVE directions:

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
        """STEP 4 — Creator / Audience Intelligence for one creator.

        Use when the founder asks why a specific creator is good, wants audience
        fit, content style, engagement, region, or performance details.

        Args:
            handle: The creator handle, with or without a leading @.
        """
        h = _norm_handle(handle)
        if not h:
            return "I need a creator handle to look up."
        filt = {"field": "handle", "condition": {"$eq": h}}
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
        """STEP 3/4 — Creator Discovery and Similar Creator Expansion.

        Use when the founder asks what kind of creators to find, asks for people
        similar to Cursor's partners, or changes platform/niche/audience filters.

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
            build_evidence(
                "similar_creators", step=4, index=IDX_KOLS, latency_ms=ms,
                items=kol_items(pool, limit=8),
                insight=f"Recalled {len(pool)} candidate creators.",
            ),
        )
        if pool:
            await publish_evidence(
                self._room,
                build_evidence(
                    "kol_profile",
                    step=4,
                    index=IDX_KOLS,
                    items=kol_items(pool[:1], limit=1),
                    insight="Audience intelligence for the strongest matching archetype.",
                    title="Audience Intelligence",
                ),
            )
            self._turn_card_types.add("kol_profile")
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
        """Ground objections, strategy, content angles, and ROI claims.

        Use for objections like "Cursor is too big to compare to", methodology
        questions like "how do we know it worked", content strategy questions,
        and ROI/budget rationale before making a confident recommendation.

        Args:
            question: The question or objection to look up.
            doc_type: Optional one of: qa, strategy, case. Invalid -> ignored.
        """
        dt = (doc_type or "").strip().lower()
        # Playbook lives in ansio_content. Moss filters support only
        # $and/$eq/$gt/$lt (no $in), so scope to a single doc_type when given;
        # otherwise query semantically (methodology docs surface by meaning).
        filt = {"field": "doc_type", "condition": {"$eq": dt}} if dt in PLAYBOOK_DOC_TYPES else None
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
            build_evidence(
                "playbook_hit", step="2b", index=IDX_CONTENT, latency_ms=ms,
                items=items, insight=(items[0]["text"][:120] if items else ""),
                source=(items[0]["source"] if items else ""),
            ),
        )
        q_lower = (question or "").lower()
        if any(k in q_lower for k in ("content", "angle", "hook", "post", "script", "内容", "脚本")):
            await publish_evidence(
                self._room,
                build_evidence(
                    "content_strategy",
                    step=7,
                    index=IDX_CONTENT,
                    items=[
                        {
                            "title": "Workflow demo",
                            "platform": "YouTube / X",
                            "reason": "Show the tool solving a real coding task.",
                            "source": items[0]["source"] if items else "playbook",
                        },
                        {
                            "title": "Builder story",
                            "platform": "X",
                            "reason": "Frame the product through an indie builder's daily workflow.",
                            "source": items[0]["source"] if items else "playbook",
                        },
                    ],
                    insight="Recommended content angles for developer conversion.",
                    title="Content Strategy",
                ),
            )
            self._turn_card_types.add("content_strategy")
        if any(k in q_lower for k in ("roi", "return", "forecast", "reach", "trial", "conversion", "预算", "转化")):
            await publish_evidence(
                self._room,
                build_evidence(
                    "roi_forecast",
                    step=8,
                    index=IDX_CONTENT,
                    items=[
                        {
                            "title": "First test budget",
                            "estimated_market_cost": 5000,
                            "reason": "Keep the first round controlled before scaling.",
                        },
                        {
                            "title": "Expected trial lift",
                            "score": "conservative",
                            "reason": "Depends on creator fit, content quality, and product activation.",
                        },
                    ],
                    insight="Conservative ROI forecast for the first creator test.",
                    title="ROI Forecast",
                ),
            )
            self._turn_card_types.add("roi_forecast")
        if not rows:
            return "I don't have a playbook note on that."
        return rows[0]["text"][:400]

    @function_tool()
    async def publish_insight_card(
        self,
        context: RunContext,
        card_type: str,
        title: str,
        insight: str,
        items_json: str = "[]",
    ) -> str:
        """Publish one LLM-authored right-panel card for the demo chain.

        Use after retrieval or reasoning when the right panel should advance the
        eight-step story:
        1 competitor_landscape = Competitor Landscape
        2 content_hits = Campaign Timeline / Partnerships
        3 similar_creators = Creator Discovery
        4 kol_profile = Audience Intelligence
        5 alpha_ranking = Creator Alpha Ranking
        6 bundle = Campaign Bundle
        7 content_strategy = Content Strategy
        8 roi_forecast = ROI Forecast / Recommended Campaign

        Only use facts from the conversation or tool results. ``items_json``
        must be a JSON array of objects.

        Args:
            card_type: One of competitor_landscape, content_hits, playbook_hit,
                kol_profile, similar_creators, alpha_ranking, bundle,
                content_strategy, roi_forecast.
            title: Short card title.
            insight: One-sentence card subtitle or takeaway.
            items_json: JSON array of item objects with fields like title, name,
                handle, platform, followers, niche, score, alpha, reason,
                estimated_market_cost, posts, total_views.
        """
        ct = (card_type or "").strip()
        if ct not in EVIDENCE_TYPES:
            ct = "content_hits"
        if ct in self._turn_card_types:
            return f"Skipped duplicate {ct} card for this turn."
        items = _llm_card_items(items_json)
        await publish_evidence(
            self._room,
            build_evidence(
                ct,
                step="LLM",
                index="llm",
                items=items,
                insight=(insight or "")[:220],
                title=(title or ct.replace("_", " ").title())[:80],
            ),
        )
        self._turn_card_types.add(ct)
        return f"Published {ct} card with {len(items)} items."

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
        """STEP 5 — Creator Alpha Ranking and recommendation.

        Call this once the founder asks for recommended creators, underpriced
        alternatives, budget-backed picks, or "who would you choose". It runs
        competitor context -> wide creator recall -> Python Alpha scoring, then
        returns ranked creators with fit/performance/value breakdown.

        Args:
            product_desc: The user's product/brief.
            niche: Optional creator niche to bias recall.
            platform: Optional platform filter.
            budget: Optional per-post budget cap. Pass this ONLY if the founder
                explicitly stated a numeric cap; otherwise leave it empty.
        """
        platform_w = _wl_platform(platform)
        niche_w = _wl_niche(niche)
        t0 = time.perf_counter()
        try:
            # Deterministic chain (in-process, no per-hop LLM round-trip):
            #  1) competitor landscape (products, semantic)
            #  2) sponsorship graph signal (content, semantic context)
            #  4) wide creator recall (kols, the scoring pool) — top_k=80
            # Do not query/publish competitors here: this meta-tool is Step 5
            # (Creator Alpha). Step 1 comes only from the explicit competitor-
            # discovery route, otherwise the right rail jumps backward during
            # recommendation turns (and the products query would be wasted).
            filt = _build_kol_filter(platform_w, niche_w)
            kres, _ = await self._q(
                IDX_KOLS, product_desc, top_k=WIDE_POOL_TOP_K, filt=filt
            )
            pool = _md_list(kres)
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
            build_evidence(
                "alpha_ranking", step=5, index=IDX_KOLS, latency_ms=chain_ms,
                items=kol_items(ranked, limit=5),
                insight="Ranked by fit, performance and value (Alpha).",
            ),
        )
        await publish_evidence(
            self._room,
            build_evidence(
                "bundle",
                step=6,
                index=IDX_KOLS,
                items=kol_items(ranked, limit=3),
                insight="Two-creator test bundle with complementary developer reach.",
                title="Campaign Bundle",
            ),
        )
        self._turn_card_types.add("bundle")
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

    # Turn policy: classify first, then decide whether to retrieve
    Silently classify every founder turn before answering:
    - pure smalltalk / thanks / greeting: no retrieval. Greet briefly and ask
      for the campaign problem, target customer, or product category.
    - greeting plus product/growth/audience facts: NOT smalltalk. Treat it as
      the relevant business intent and retrieve immediately.
    - qualification: no retrieval unless the founder already names a product,
      audience, competitor, creator, budget, or growth goal. Ask at most one
      sharp question.
    - new growth brief: call `find_competitors` first, then answer with the
      benchmark logic. Do not ask for more info if the product category and
      growth goal are already present.
    - competitor campaign question ("who did Cursor work with", "what worked"):
      call `find_kols_who_promoted`.
    - objection / proof / "Cursor is too big" / "how do we know it worked":
      call `search_playbook` before making the strategic argument — but ONLY
      after `find_competitors` has already fired; before that, curiosity or
      doubt about competitors means run `find_competitors` first.
    - creator discovery / "what kind of达人" / "find similar people":
      call `find_similar_kols`.
    - specific creator / handle / audience detail: call `get_kol_profile`.
    - recommendation / "who would you pick" / underpriced creators:
      call `recommend_kols` once, then reason from the ranked result. Do not
      pass a budget unless the founder gave a numeric cap.
    - budget range question with no stated cap: recommend a first-test range of
      3000-5000 USD, then offer to refine. Do not pass a fake budget to tools.
    - content angle question: call `search_playbook`, then publish a
      `content_strategy` card with concrete angles.
    - ROI / expected reach / conversion question: call `search_playbook`, then
      publish a `roi_forecast` card with conservative ranges if supported by
      playbook or current plan context.
    - presentation step: when a right-rail card should advance the demo story,
      call `publish_insight_card` with your chosen card_type and item content.

    # Eight-step demo chain
    Make the conversation naturally drive this right rail:
    1 Competitor Landscape -> `competitor_landscape`
    2 Campaign Timeline / Partnerships -> `content_hits`
    3 Creator Discovery -> `similar_creators`
    4 Audience Intelligence -> `kol_profile`
    5 Creator Alpha Ranking -> `alpha_ranking`
    6 Campaign Bundle -> `bundle`
    7 Content Strategy -> `content_strategy`
    8 ROI Forecast / Recommended Campaign -> `roi_forecast`
    - Do not skip forward in the visible card chain. If the earlier evidence
      step has not appeared in the conversation, ask or retrieve that step
      before publishing a later card. In particular, show Campaign Timeline
      before Creator Discovery, and Creator Alpha before Bundle/Content/ROI.
    - Do not publish earlier-step cards after a later decision card; use spoken
      summary instead.
    - After Creator Discovery, publish Audience Intelligence (`kol_profile`) for
      the strongest creator or archetype before moving to Creator Alpha.
    - After Creator Alpha, publish Campaign Bundle (`bundle`) before moving to
      Content Strategy or ROI.

    # Flow discipline (HARD tool ordering — overrides turn-policy routing)
    The four retrieval stages MUST first-fire in exactly this order:
    `find_competitors` -> `find_kols_who_promoted` -> `find_similar_kols`
    -> `recommend_kols`.
    - The founder's FIRST growth brief (product + audience, even inside a
      greeting) means: call `find_competitors` in that SAME turn. The story
      cannot start without the competitor landscape.
    - If a turn suggests a later stage but an earlier stage has NEVER fired in
      this conversation, call the EARLIEST missing stage instead and narrate
      that step (e.g. a reaction like "can we learn from them?" right after a
      growth brief means: run `find_competitors` now if it has not run).
    - "How do we pick / choose / find OUR creators?" means `find_similar_kols`,
      NOT `recommend_kols`.
    - Never call `recommend_kols` before the first three stages have fired —
      unless the user explicitly demands the final list immediately.
    - One stage per turn: make the stage's tool call, reveal, then let the
      user react before the next stage.

    # How you work
    - Lead with insight, not interrogation: once you know the product category
      and audience, retrieve and show value.
    - Prefer one well-chosen retrieval tool per turn. Do not call every tool.
    - After retrieval, answer in the founder's language and, when appropriate,
      publish one polished right-rail card using `publish_insight_card`. Avoid
      duplicate cards of the same type in the same turn unless the LLM card adds
      a clearer decision layer.
    - Use only tool results or conversation facts in cards. Never invent names,
      handles, follower counts, prices, or ROI metrics.
    - If the user asks to proceed at the end, summarize the selected creators,
      budget, expected reach/trials/ROI, and publish an ROI Forecast card.
    - For the AI coding tool demo, keep the narrative close to:
      Cursor/Codeium benchmark -> creator partnerships -> similar creators ->
      Creator Alpha -> two-creator bundle -> workflow/demo content -> ROI.
    - If a turn asks both "who would you pick" and budget, call `recommend_kols`,
      then publish a `bundle` card with the two selected creators, budget split,
      and audience coverage before discussing content or ROI.

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
    - If the user only greets you, start with a clear friendly greeting before
      asking for their campaign or product details.
    - Spell numbers naturally ("three hundred thousand followers").
    - Read handles plainly, without the at-sign noise.
    - Only mention an ESTIMATED MARKET COST; never claim to know a creator's
      real private rate.
    - Never invent names, handles, or follower counts — only use tool results.
    - Do not reveal tool names, parameters, or internal retrieval notes.
    """
)


# num_idle_processes=1 + generous initialize timeout: conservative defaults for
# stable cold starts on the demo machine (a heavier prewarm or CPU contention
# must never starve sibling process imports past the spawn timeout).
server = AgentServer(num_idle_processes=1, initialize_process_timeout=60.0)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # NOTE: the local KOL fallback index (moss_router) arms itself lazily on
    # first cloud failure — no eager build here, so process warmup stays light.


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

    # Consent-gated user-profile memory + language mode (settings panel rides in
    # via dispatch metadata; console mode opts in via ANSIO_MEMORY_ENABLED=1).
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
        # Sentence-by-sentence TTS (user-chosen): StreamAdapter tokenizes the
        # reply into sentences and synthesizes EACH as its own complete request
        # — one buffer per sentence, played atomically, cleared when done. A
        # mid-utterance "slow generation" flush can therefore never truncate a
        # sentence; the worst case is a brief pause BETWEEN sentences.
        tts=lk_tts.StreamAdapter(
            tts=minimax.TTS(
                model="speech-2.8-turbo",
                voice=os.getenv("ANSIO_TTS_VOICE", "English_Persuasive_Man"),
                language_boost=tts_language_boost(language),
                emotion="neutral",
                speed=1.0,
                sample_rate=24000,
            ),
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
            # Text-FIRST captions (user requirement): publish the full reply
            # transcript immediately instead of word-syncing it to audio, so
            # the text is always on screen ahead of (and independent of) TTS.
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
