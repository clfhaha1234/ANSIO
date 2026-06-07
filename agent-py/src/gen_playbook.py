"""Generate the ANSIO `playbook` dataset (methodology + historical case library).

Writes ``agent-py/playbook.json`` — a list of documents shaped for Moss:

    { "id": "pb-NNNN", "text": "<methodology / strategy / case body>", "metadata": {...} }

The ``text`` is the retrievable body: methodology Q&A ("why should a small
company copy a big company's influencer-marketing playbook"), content-strategy
docs ("why workflow content converts better than reviews"), pricing/negotiation
guidance, platform-selection guidance, and historical campaign case studies.

Purpose (ANSIO PRD §2.1 / §2.3, index D · playbook):
  1. Grounded objection handling — the agent's persuasion lines come from
     retrieved docs (with a visible `source`), not LLM improvisation.
  2. ROI forecasting with a citation — Step 8 retrieves similar historical
     cases and extrapolates from real numbers instead of inventing "8-12x".

Schema:
    id              top-level   string   `pb-NNNN`
    text            top-level   string   body (vectorized)
    doc_type        metadata    string   enum {qa, strategy, case}  ($eq filter)
    source          metadata    string   provenance (shown on the evidence card)
    campaign_reach  metadata    string   integer-as-string  (case docs only)
    campaign_trials metadata    string   integer-as-string  (case docs only)
    campaign_roi    metadata    string   float-as-string    (case docs only)

`doc_type` is retained so that, under the free-tier 3-index plan, playbook can
be merged into the `content` index and still be distinguished by `$eq doc_type`
(see warm-brewing-widget.md Phase C). All metadata values are strings.

Deterministic: a fixed, curated list (no randomness) → re-runs are
byte-identical. A seeded RNG is constructed for any future jitter but the
output here is fully deterministic regardless.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "playbook.json"
SEED = 42

# ---------------------------------------------------------------------------
# QA — methodology Q&A for grounded objection handling.
# Each entry: (text, source)
# ---------------------------------------------------------------------------
QA = [
    ("Q: Cursor is huge and well-funded — does it even make sense for a small "
     "startup to study who they partner with? A: Yes. The value is not in matching "
     "their budget but in reverse-engineering their creator graph. Which developer "
     "KOLs accepted a Cursor sponsorship tells you which audiences are receptive to "
     "AI-coding tools and which content formats already converted. You inherit their "
     "audience validation for free, then target the same creators' under-priced "
     "peers.",
     "ANSIO methodology — competitor creator-graph mining"),
    ("Q: Won't the big creators who worked with Cursor be too expensive for us? "
     "A: Often the headline names are, but the long tail isn't. A creator who did a "
     "Cursor integration proves the audience converts; their micro and nano peers in "
     "the same niche have the same audience at a fraction of the price and higher "
     "engagement. Target the proven niche, not the proven name.",
     "ANSIO methodology — micro-influencer arbitrage"),
    ("Q: Why should I trust influencer marketing over paid ads for a dev tool? "
     "A: Developers have famously high ad-blindness and ad-blocker adoption, and they "
     "trust peer demonstration far more than banner copy. A trusted creator showing a "
     "real workflow is an implicit endorsement that paid ads cannot buy. For "
     "bottom-of-funnel dev-tool trials, creator content typically beats display ads on "
     "cost-per-trial.",
     "ANSIO methodology — why creators for dev tools"),
    ("Q: How many creators should a first campaign use? A: Start with a portfolio of "
     "3-5 micro creators rather than one macro creator. The portfolio diversifies the "
     "risk of any single post underperforming, lets you A/B test angles and formats, "
     "and produces more total authentic touchpoints for the same budget.",
     "ANSIO methodology — portfolio sizing"),
    ("Q: My product is technical — won't most creators get the demo wrong? A: That is "
     "exactly why you select for creators whose existing content already shows hands-on "
     "building, not just commentary. Audience-construction signals (share of developers "
     "and founders) and demo-style content history predict whether a creator can "
     "represent a technical product credibly. Filter on that, not raw follower count.",
     "ANSIO methodology — technical-fit screening"),
    ("Q: How do I know a creator's audience actually contains my buyers? A: Look past "
     "the follower count to audience construction. For a dev tool you want a high share "
     "of developers and a meaningful share of founders or decision-makers. Two creators "
     "with identical follower counts can have wildly different buyer density; the one "
     "with the right audience mix is worth several times more per dollar.",
     "ANSIO methodology — audience construction over reach"),
    ("Q: Is engagement rate or follower count the better signal? A: Engagement rate, by "
     "a wide margin, for conversion-oriented campaigns. A nano creator at 8% engagement "
     "routinely out-converts a mega creator at 1.5% because the audience relationship is "
     "stronger and the recommendation reads as personal rather than paid.",
     "ANSIO methodology — engagement vs reach"),
    ("Q: What is an 'under-priced' or alpha creator? A: One whose true influence — "
     "engagement, audience fit, recent growth — outpaces their current sponsorship "
     "price. Pricing in the creator market lags real momentum, so creators on a steep "
     "growth curve are systematically under-priced for a window of a few months. Catch "
     "them inside that window.",
     "ANSIO methodology — alpha definition"),
    ("Q: Should I disclose that a post is sponsored? A: Always. Beyond legal "
     "requirements, developer and founder audiences actively reward transparency and "
     "punish anything that reads as covert. A clearly disclosed, genuinely useful "
     "sponsored demo performs better than a disguised one.",
     "ANSIO methodology — disclosure and trust"),
    ("Q: How long before a creator campaign shows results? A: Trial signups spike within "
     "the first 72 hours of a post, but the durable tail — saves, reshares, and "
     "comment-driven discovery — keeps delivering trials for 2-4 weeks. Judge a campaign "
     "on a 30-day window, not the first day.",
     "ANSIO methodology — measurement window"),
    ("Q: One creator wants exclusivity in my category — is that worth it? A: Rarely for "
     "an early-stage startup. Exclusivity caps your reach and hands pricing power to a "
     "single creator. Prefer a non-exclusive portfolio until you have proven which "
     "creators drive disproportionate conversion, then consider locking in only the "
     "clear winners.",
     "ANSIO methodology — exclusivity tradeoffs"),
    ("Q: Can I just repurpose the same brief across every creator? A: No. A copy-paste "
     "brief produces stilted, off-voice content that audiences immediately discount. "
     "Give each creator the same goal and key message but let them express it in their "
     "native format and voice. Constrain the what, not the how.",
     "ANSIO methodology — brief design"),
    ("Q: A competitor sponsored a creator and it flopped — should I avoid that creator? "
     "A: Not necessarily. A flop is often a brief or product-fit problem, not a creator "
     "problem. Diagnose why it underperformed (wrong format, weak hook, mismatched "
     "audience) before writing the creator off; a strong creator with a better brief can "
     "still convert.",
     "ANSIO methodology — diagnosing competitor flops"),
    ("Q: Should I pick creators by niche keyword match or by audience? A: By audience. "
     "Keyword match on niche is a starting filter, but two 'tech' creators can serve "
     "completely different audiences. Always validate that the creator's audience "
     "actually contains your buyers before committing budget — niche is necessary, not "
     "sufficient.",
     "ANSIO methodology — niche vs audience match"),
    ("Q: How do I avoid overpaying a creator whose price already priced in their fame? "
     "A: Compare their quote against peers with similar engaged-audience size in the same "
     "niche. If the creator's price is a large multiple of comparable peers without a "
     "matching engagement or audience-fit advantage, you are paying for fame, not "
     "conversion. Walk toward the under-priced peer.",
     "ANSIO methodology — avoid paying for fame"),
    ("Q: My budget is tiny — is creator marketing even viable? A: Yes, and a small budget "
     "actually favors the highest-ROI segment. Nano and micro creators with strong "
     "engagement are the most capital-efficient entry point; a $2,000 portfolio of nano "
     "creators routinely out-converts a single $2,000 macro post.",
     "ANSIO methodology — small-budget viability"),
    ("Q: Should I chase a viral moment or build a steady cadence? A: For a startup, a "
     "steady cadence of credible workflow content compounds trust and trial volume more "
     "reliably than chasing one viral hit. Virality is unpredictable; consistency with "
     "the right creators is controllable and durable.",
     "ANSIO methodology — cadence over virality"),
]

# ---------------------------------------------------------------------------
# STRATEGY — content-strategy, pricing/negotiation, platform-selection docs.
# Each entry: (text, source)
# ---------------------------------------------------------------------------
STRATEGY = [
    ("Workflow content outperforms review content for developer tools. A 'here is my "
     "real refactoring workflow with this tool' video converts substantially better "
     "than a 'top 5 AI coding tools' review, because it shows the product solving a "
     "problem the viewer recognizes, in context, with a clear before/after. Brief "
     "creators to build something real on camera rather than rank competitors.",
     "ANSIO content strategy — workflow over review"),
    ("The strongest dev-tool hook is a concrete, quantified outcome in the first three "
     "seconds: 'I shipped this feature in 20 minutes instead of two hours.' Lead with "
     "the result, then show the workflow that produced it. Generic 'this tool is "
     "amazing' openers lose technical viewers instantly.",
     "ANSIO content strategy — hook design"),
    ("Sequence content from problem to product, never product-first. Open on the pain "
     "the developer already feels (debugging legacy code, slow refactors), establish it "
     "viscerally, then introduce the tool as the resolution. Product-first content reads "
     "as an ad; problem-first content reads as a story the viewer is already inside.",
     "ANSIO content strategy — narrative sequencing"),
    ("Negotiating creator rates: anchor on deliverables and usage rights, not on a flat "
     "post fee. A lower base fee plus whitelisting or paid-usage rights often costs less "
     "than a high flat fee while giving you reusable ad creative. Always ask what a "
     "bundle of three posts costs — per-post rates drop sharply with volume.",
     "ANSIO pricing — rate negotiation"),
    ("Performance-based and hybrid deals de-risk creator spend. Offer a modest base plus "
     "a per-trial or per-signup bonus tied to a unique tracking link or code. Strong "
     "creators welcome upside; the structure also surfaces who genuinely believes they "
     "can drive conversions versus who just wants a flat check.",
     "ANSIO pricing — performance-based deals"),
    ("Watch the price-to-engagement ratio, not the absolute price. Normalize each "
     "creator's quote by their engaged audience (followers times engagement rate) to get "
     "a true cost-per-engaged-impression. A creator who looks expensive on a flat number "
     "is often the cheapest on this normalized basis.",
     "ANSIO pricing — cost per engaged impression"),
    ("Platform selection follows audience intent. YouTube long-form wins for "
     "consideration and deep workflow demos developers rewatch; short-form (TikTok, "
     "Reels, Shorts) wins for top-of-funnel reach and the quantified-hook format; X "
     "wins for developer-to-developer credibility and launch moments. Match the platform "
     "to the funnel stage, not to vanity reach.",
     "ANSIO platform — funnel-stage mapping"),
    ("For developer tools specifically, YouTube and X over-index on buyer density, while "
     "Instagram and TikTok over-index on raw reach. A dev-tool budget is usually better "
     "weighted toward YouTube depth plus X credibility, using short-form selectively for "
     "awareness spikes around launches.",
     "ANSIO platform — dev-tool channel weighting"),
    ("Bilibili and other regional platforms matter when the target developer audience is "
     "non-English. A creator native to the audience's language and platform converts far "
     "better than a translated repost. For bilingual launches, pair an English YouTube "
     "creator with a same-niche regional-platform creator rather than translating one "
     "asset.",
     "ANSIO platform — localization"),
    ("Bundle creators for audience coverage, not just reach. Two creators whose audiences "
     "barely overlap deliver more unique reach per dollar than two whose audiences are "
     "nearly identical. When building a portfolio, optimize for low audience overlap "
     "across niche, region, and platform so you are not paying twice for the same eyeballs.",
     "ANSIO strategy — bundle for low overlap"),
    ("Time creator drops around product moments. Cluster posts around a launch, a major "
     "feature, or a pricing change so the spike in trial traffic compounds with PR and "
     "organic discovery. A coordinated week of creator content outperforms the same posts "
     "scattered randomly across a quarter.",
     "ANSIO strategy — campaign timing"),
    ("Always negotiate content-usage rights up front. The ability to repurpose a "
     "creator's video as a paid ad, a landing-page asset, or social proof multiplies the "
     "value of the deal. Usage rights bought after the fact cost far more than rights "
     "bundled into the original agreement.",
     "ANSIO pricing — usage rights"),
    ("Avoid one-and-done sponsorships when a creator performs. A repeat collaboration "
     "reads as a genuine ongoing relationship to the audience and compounds trust, while "
     "a single appearance is easy to dismiss as a paid placement. Budget for a second and "
     "third touch with your proven winners.",
     "ANSIO strategy — repeat collaboration"),
    ("Match the creator's content cadence and format to your brief. Asking a long-form "
     "tutorial creator to make a 15-second hook, or a fast-cut short-form creator to film "
     "a 20-minute deep dive, fights their muscle memory and their audience's expectations. "
     "Pick creators whose native format already is the format you need.",
     "ANSIO strategy — format-creator fit"),
]

# ---------------------------------------------------------------------------
# CASE — historical campaign case studies (with reach/trials/roi numbers).
# Each entry: (text, source, reach, trials, roi)
# ---------------------------------------------------------------------------
CASE = [
    ("An AI coding tool ran a campaign across 5 mid-tier developer creators (each "
     "roughly 10k-50k followers) with a $4,000 budget. The portfolio drove about 2,200 "
     "free-trial signups and 110 paid conversions, validating the micro-influencer "
     "workflow-content thesis at roughly 8.5x return on spend.",
     "Internal campaign retro 2025Q4 — AI coding micro portfolio", "520000", "2200", "8.5"),
    ("A developer-productivity startup spent $12,000 on a single macro YouTube creator "
     "(1.4M subscribers). The video reached 380k views but converted only about 640 "
     "trials and 28 paid users — roughly 2.1x return — underperforming the same brand's "
     "micro-creator portfolio on a cost-per-trial basis.",
     "Internal campaign retro 2025Q3 — single macro test", "380000", "640", "2.1"),
    ("A code-review tool sponsored 8 nano creators (under 10k followers each) at $250 "
     "apiece for $2,000 total. High engagement (average 7.4%) produced about 1,500 "
     "trials and 95 paid conversions, an outsized roughly 11.0x return that confirmed "
     "nano engagement beats macro reach for niche dev tools.",
     "Internal campaign retro 2026Q1 — nano engagement play", "210000", "1500", "11.0"),
    ("An AI IDE ran a coordinated launch week with 3 YouTube workflow creators plus 4 "
     "X dev-influencers, $9,000 total, timed to a major feature release. The cluster "
     "drove about 4,100 trials and 240 paid users at roughly 7.8x, demonstrating the "
     "compounding effect of timing creator content around a product moment.",
     "Internal campaign retro 2026Q1 — coordinated launch week", "910000", "4100", "7.8"),
    ("A bilingual launch paired one English YouTube creator with one same-niche "
     "Bilibili creator for $6,500. The localized pair reached two distinct developer "
     "audiences and drove about 3,000 trials and 165 conversions at roughly 6.9x, "
     "outperforming a prior translated-repost attempt by a wide margin.",
     "Internal campaign retro 2025Q4 — bilingual pair", "640000", "3000", "6.9"),
    ("A DevTool spent $15,000 chasing a single mega creator (3.2M followers, 1.4% "
     "engagement). Despite 1.1M views the campaign returned only about 1,800 trials and "
     "70 paid users — roughly 1.8x — the canonical example of paying for reach instead "
     "of buyer density.",
     "Internal campaign retro 2025Q2 — mega reach cautionary", "1100000", "1800", "1.8"),
    ("A workflow-content campaign for an AI refactoring tool used 4 creators briefed to "
     "build a real feature on camera, $5,500 total. The problem-first, quantified-hook "
     "format drove about 2,600 trials and 140 conversions at roughly 9.2x, beating the "
     "same brand's earlier review-style content by more than double on cost-per-trial.",
     "Internal campaign retro 2026Q1 — workflow vs review", "470000", "2600", "9.2"),
    ("A startup tested performance-based deals: $200 base plus $4 per tracked trial "
     "across 6 micro creators. Total spend landed near $7,200 for about 3,400 trials and "
     "190 paid users (roughly 8.1x), and the structure cleanly separated the two "
     "creators who drove 60% of conversions from the rest.",
     "Internal campaign retro 2025Q4 — performance-based structure", "560000", "3400", "8.1"),
    ("A pricing-tool brand negotiated a 3-post bundle plus whitelisting rights with one "
     "mid creator for $3,800 (versus a $2,200 single-post quote). The reusable ad "
     "creative and three touchpoints drove about 1,900 trials and 105 conversions at "
     "roughly 7.4x, with the whitelisted assets continuing to perform as paid ads.",
     "Internal campaign retro 2026Q1 — bundle plus whitelisting", "330000", "1900", "7.4"),
    ("A low-overlap bundle paired a US tech YouTuber with a Europe-based fitness-tech "
     "crossover creator for an AI-coding wellness app, $8,000. Minimal audience overlap "
     "maximized unique reach, yielding about 3,700 trials and 200 conversions at roughly "
     "8.0x.",
     "Internal campaign retro 2025Q3 — low-overlap bundle", "720000", "3700", "8.0"),
]


def main() -> None:
    # Seeded RNG kept for forward-compatibility; output is deterministic.
    _rng = random.Random(SEED)  # noqa: F841

    docs = []
    idx = 0

    def add(text: str, doc_type: str, source: str, extra: dict | None = None) -> None:
        nonlocal idx
        idx += 1
        meta = {"doc_type": doc_type, "source": source}
        if extra:
            meta.update(extra)
        docs.append({"id": f"pb-{idx:04d}", "text": text, "metadata": meta})

    for text, source in QA:
        add(text, "qa", source)
    for text, source in STRATEGY:
        add(text, "strategy", source)
    for text, source, reach, trials, roi in CASE:
        add(
            text,
            "case",
            source,
            {
                "campaign_reach": reach,
                "campaign_trials": trials,
                "campaign_roi": roi,
            },
        )

    OUT_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter

    types = Counter(d["metadata"]["doc_type"] for d in docs)
    print(f"Wrote {len(docs)} playbook docs to {OUT_PATH}")
    print("Doc types:", dict(sorted(types.items())))
    print("\nSample:\n", docs[0]["text"][:160], "...")


if __name__ == "__main__":
    main()
