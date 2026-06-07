"""Generate a synthetic-but-realistic products dataset for the ANSIO `products` index.

Writes ``agent-py/products.json`` — a list of documents shaped for Moss:

    { "id": "prod-NNNN", "text": "<positioning description>", "metadata": {...} }

The ``text`` is a natural-language product positioning blurb so Moss semantic
search surfaces the right competitors from conversational queries ("I'm building
an AI coding tool, growth stalled"). ``metadata`` carries clean facets for
display and (optionally) filtering.

Schema (ANSIO PRD §2.1 / §2.3, index A · products):
    id        top-level   string   `prod-NNNN`
    text      top-level   string   positioning description (vectorized)
    name      metadata    string   display name (original casing)
    name_norm metadata    string   lowercase name (cross-ref with content.brand)
    category  metadata    string   track tag, lowercase (e.g. `ai-coding`, `beauty`)
    funding   metadata    string   funding amount/round string (e.g. `400M`, `seed`)
    stage     metadata    string   `seed`/`series-a`/`series-c`/`public`/...

All metadata values are strings (Moss metadata is Dict[str, str]).

The AI-coding track is real products with real positioning (the demo main line:
Cursor / Copilot / Replit / Codeium / Windsurf ...). Other tracks (beauty /
fitness / productivity / ...) are seeded too, so judges can switch topics live
without an empty `products` index.

Deterministic: seeded RNG so re-runs produce byte-identical output.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "products.json"
SEED = 42

# ---------------------------------------------------------------------------
# Curated products. Each entry: (name, category, funding, stage, positioning).
# The positioning text is written so semantic search hits from natural queries.
# ---------------------------------------------------------------------------

# --- AI coding track (demo main line): real products + real positioning ---
AI_CODING = [
    (
        "Cursor",
        "ai-coding",
        "400M",
        "series-c",
        "Cursor is an AI-first code editor for professional developers, known for agentic editing, "
        "multi-file refactoring, and rapid iteration over large codebases. A fork of VS Code that "
        "puts an AI pair programmer at the center of the editing workflow.",
    ),
    (
        "GitHub Copilot",
        "ai-coding",
        "acquired",
        "public",
        "GitHub Copilot is an AI pair programmer built into the IDE, offering inline code completions, "
        "chat, and agentic workflows. Backed by GitHub and Microsoft, it is the most widely adopted "
        "AI coding assistant among enterprise developers.",
    ),
    (
        "Replit",
        "ai-coding",
        "97M",
        "series-b",
        "Replit is a browser-based collaborative coding platform with an AI agent that builds and "
        "deploys full applications from natural language prompts. Popular with learners, hobbyists, "
        "and rapid prototypers who want zero local setup.",
    ),
    (
        "Codeium",
        "ai-coding",
        "150M",
        "series-c",
        "Codeium is a free-tier AI code completion and chat assistant supporting dozens of IDEs and "
        "languages, with an enterprise self-hosted option. Known for fast autocomplete and broad "
        "editor coverage.",
    ),
    (
        "Windsurf",
        "ai-coding",
        "243M",
        "series-c",
        "Windsurf is an agentic AI IDE built by the Codeium team, featuring the Cascade flow that lets "
        "the AI read, edit, and run code across the whole project. Positioned as a deeply autonomous "
        "coding agent inside a purpose-built editor.",
    ),
    (
        "Tabnine",
        "ai-coding",
        "55M",
        "series-b",
        "Tabnine is a privacy-first AI code assistant offering autocomplete and chat that can run fully "
        "air-gapped or self-hosted. Targeted at security-conscious enterprises with strict code "
        "privacy requirements.",
    ),
    (
        "Amazon Q Developer",
        "ai-coding",
        "internal",
        "public",
        "Amazon Q Developer is AWS's AI coding assistant for the IDE and command line, with deep "
        "integration into AWS services, security scanning, and codebase agents. Aimed at teams "
        "building on the AWS cloud.",
    ),
    (
        "Sourcegraph Cody",
        "ai-coding",
        "175M",
        "series-d",
        "Cody by Sourcegraph is an AI coding assistant with whole-codebase context powered by code "
        "search, offering chat, autocomplete, and automated fixes across large enterprise "
        "repositories.",
    ),
    (
        "JetBrains AI Assistant",
        "ai-coding",
        "internal",
        "public",
        "JetBrains AI Assistant brings code generation, explanation, and refactoring natively into "
        "IntelliJ, PyCharm, and the rest of the JetBrains IDE family, leveraging deep static analysis "
        "and project understanding.",
    ),
    (
        "Aider",
        "ai-coding",
        "bootstrapped",
        "open-source",
        "Aider is an open-source AI pair programming tool that runs in the terminal and edits code in a "
        "local git repository, making commit-by-commit changes from natural language. Beloved by "
        "command-line and open-source developers.",
    ),
    (
        "Cline",
        "ai-coding",
        "32M",
        "seed",
        "Cline is an open-source autonomous coding agent that runs inside VS Code, planning and "
        "executing multi-step tasks, running terminal commands, and editing files with human approval "
        "at each step.",
    ),
    (
        "Bolt.new",
        "ai-coding",
        "106M",
        "series-b",
        "Bolt.new by StackBlitz is an AI web development agent that builds, runs, and deploys "
        "full-stack apps entirely in the browser from a prompt. A leading player in the prompt-to-app "
        "vibe-coding wave.",
    ),
    (
        "Lovable",
        "ai-coding",
        "200M",
        "series-a",
        "Lovable is an AI app builder that turns natural-language prompts into production-ready "
        "full-stack web applications, targeting founders and non-engineers who want to ship products "
        "without writing code.",
    ),
    (
        "v0 by Vercel",
        "ai-coding",
        "internal",
        "public",
        "v0 by Vercel is a generative UI tool that produces React and Tailwind components and full "
        "pages from prompts or images, tightly integrated with the Vercel deployment platform.",
    ),
    (
        "Devin",
        "ai-coding",
        "196M",
        "series-a",
        "Devin by Cognition is an autonomous AI software engineer that plans and completes entire "
        "engineering tasks end to end, from a ticket to a merged pull request. Positioned as a "
        "fully autonomous teammate rather than an in-editor assistant.",
    ),
    (
        "Augment Code",
        "ai-coding",
        "252M",
        "series-b",
        "Augment Code is an AI coding platform optimized for large, complex codebases, offering "
        "context-aware chat, completions, and agents tuned for professional engineering teams.",
    ),
    (
        "Continue",
        "ai-coding",
        "3M",
        "seed",
        "Continue is an open-source AI code assistant for VS Code and JetBrains that lets developers "
        "build and customize their own autocomplete and chat experiences with any model.",
    ),
    (
        "Supermaven",
        "ai-coding",
        "12M",
        "seed",
        "Supermaven is an ultra-fast AI autocomplete with an exceptionally long context window, "
        "focused on low-latency inline suggestions for power users. Later joined Cursor.",
    ),
    (
        "CodeRabbit",
        "ai-coding",
        "16M",
        "series-a",
        "CodeRabbit is an AI code review platform that posts contextual, line-by-line review comments "
        "on pull requests, helping teams catch bugs and ship faster.",
    ),
    (
        "Qodo",
        "ai-coding",
        "50M",
        "series-a",
        "Qodo, formerly Codium AI, is a code-integrity platform that generates tests, reviews code, and "
        "validates behavior, positioning itself around quality and test coverage rather than raw "
        "generation.",
    ),
    (
        "Zed",
        "ai-coding",
        "60M",
        "series-b",
        "Zed is a high-performance, collaborative code editor written in Rust with built-in AI "
        "assistance, targeting developers who want speed and real-time pair editing.",
    ),
    (
        "Pieces",
        "ai-coding",
        "12M",
        "seed",
        "Pieces is an on-device AI developer assistant that captures workflow context, snippets, and "
        "history to provide grounded, privacy-preserving help across the developer's tools.",
    ),
    (
        "Mintlify",
        "ai-coding",
        "21M",
        "series-a",
        "Mintlify is an AI-powered documentation platform that generates and maintains beautiful "
        "developer docs from the codebase, targeting devtool and API companies.",
    ),
    (
        "Warp",
        "ai-coding",
        "73M",
        "series-b",
        "Warp is a modern, AI-powered terminal with an agent mode that can run and chain commands, "
        "aimed at developers who live on the command line.",
    ),
    (
        "Phind",
        "ai-coding",
        "5M",
        "seed",
        "Phind is an AI answer engine for developers that searches the web and documentation to answer "
        "technical questions with cited, code-rich responses.",
    ),
]

# --- Other tracks: seeded so judges can switch topics live (anti-pivot). ---
OTHER_TRACKS = [
    # beauty
    ("Glossier", "beauty", "266M", "series-e",
     "Glossier is a direct-to-consumer beauty brand built on a minimalist, skin-first aesthetic and "
     "a community-driven, social-native marketing playbook."),
    ("Rare Beauty", "beauty", "celebrity", "private",
     "Rare Beauty is Selena Gomez's makeup line emphasizing inclusivity, mental-health advocacy, and "
     "lightweight, buildable formulas that went viral on TikTok."),
    ("Fenty Beauty", "beauty", "celebrity", "private",
     "Fenty Beauty is Rihanna's cosmetics brand famous for its inclusive 40+ foundation shade range "
     "that reshaped industry standards for diversity."),
    ("The Ordinary", "beauty", "acquired", "public",
     "The Ordinary is a skincare brand offering clinical, single-ingredient formulations at radically "
     "low prices, demystifying actives for an educated consumer."),
    ("Drunk Elephant", "beauty", "845M", "acquired",
     "Drunk Elephant is a clean clinical skincare brand built around biocompatible ingredients and a "
     "colorful, no-suspicious-six positioning."),
    ("Charlotte Tilbury", "beauty", "1.3B", "acquired",
     "Charlotte Tilbury is a luxury makeup and skincare brand built on celebrity-makeup-artist "
     "expertise and iconic hero products like Pillow Talk."),
    ("e.l.f. Cosmetics", "beauty", "public", "public",
     "e.l.f. Cosmetics is a value-driven beauty brand known for affordable dupes of premium products "
     "and a dominant, playful presence on TikTok."),
    # fitness
    ("Whoop", "fitness", "405M", "series-f",
     "Whoop is a subscription wearable that tracks recovery, strain, and sleep for serious athletes, "
     "selling a screenless band plus a data membership."),
    ("Peloton", "fitness", "public", "public",
     "Peloton is a connected-fitness company pairing premium cardio equipment with live and on-demand "
     "instructor-led classes and a strong community."),
    ("Strava", "fitness", "151M", "series-f",
     "Strava is a social fitness network for runners and cyclists, turning workouts into shareable, "
     "competitive social activity with segments and leaderboards."),
    ("Tonal", "fitness", "650M", "series-e",
     "Tonal is a wall-mounted smart home gym using digital weight and AI coaching to deliver "
     "strength training with adaptive resistance."),
    ("Future", "fitness", "125M", "series-c",
     "Future pairs members with a dedicated remote personal trainer who programs and coaches workouts "
     "through an app, blending human accountability with technology."),
    ("Ladder", "fitness", "20M", "series-a",
     "Ladder is a strength-training app offering coach-built programs and team-based motivation for "
     "people who want structured lifting plans."),
    # productivity
    ("Notion", "productivity", "343M", "series-c",
     "Notion is an all-in-one workspace combining notes, docs, databases, and wikis, with AI "
     "features, popular with startups and knowledge workers."),
    ("Linear", "productivity", "85M", "series-b",
     "Linear is a fast, opinionated issue tracker and project management tool built for high-velocity "
     "software teams who value speed and craft."),
    ("Superhuman", "productivity", "108M", "series-c",
     "Superhuman is a premium, keyboard-first email client with AI triage and shortcuts, marketed on "
     "the promise of inbox speed for busy professionals."),
    ("Raycast", "productivity", "30M", "series-b",
     "Raycast is an extensible Mac launcher and command bar that lets power users control apps, "
     "snippets, and AI from the keyboard."),
    ("Todoist", "productivity", "bootstrapped", "private",
     "Todoist is a cross-platform task manager with natural-language input and a clean system for "
     "personal and team productivity."),
    ("Obsidian", "productivity", "bootstrapped", "private",
     "Obsidian is a local-first markdown knowledge base with bidirectional links, beloved by "
     "note-takers building a personal knowledge graph."),
    # consumer / wellness
    ("Calm", "wellness", "218M", "series-c",
     "Calm is a meditation and sleep app offering guided sessions, sleep stories, and mindfulness "
     "content for stress reduction."),
    ("Headspace", "wellness", "320M", "acquired",
     "Headspace is a mindfulness and meditation app delivering structured courses and sleep content, "
     "now expanded into enterprise mental health."),
    ("Oura", "wellness", "550M", "series-d",
     "Oura makes a smart ring that tracks sleep, readiness, and heart-rate variability, selling "
     "hardware plus a health-insights membership."),
    ("AG1", "wellness", "115M", "private",
     "AG1, formerly Athletic Greens, is a subscription daily greens powder marketed heavily through "
     "podcast and creator sponsorships."),
    # finance
    ("Robinhood", "finance", "public", "public",
     "Robinhood is a commission-free investing app that popularized fractional shares and options "
     "trading with a mobile-first, gamified experience."),
    ("Mercury", "finance", "152M", "series-c",
     "Mercury is a digital bank built for startups, offering business banking, treasury, and "
     "spend management with a developer-friendly experience."),
    ("Ramp", "finance", "1.1B", "series-e",
     "Ramp is a corporate card and spend-management platform that automates expenses and saves "
     "companies money, growing through founder and finance creators."),
    # ecommerce / dtc
    ("Shopify", "ecommerce", "public", "public",
     "Shopify is the leading commerce platform letting merchants build and run online stores, with a "
     "vast app ecosystem and creator-commerce tooling."),
    ("Gumroad", "ecommerce", "16M", "series-a",
     "Gumroad is a simple platform for creators to sell digital products directly to their audience "
     "with minimal setup."),
    # edtech
    ("Duolingo", "education", "public", "public",
     "Duolingo is a gamified language-learning app with a famously bold social-media mascot strategy "
     "and streak-driven engagement."),
    ("Brilliant", "education", "47M", "series-b",
     "Brilliant teaches math, science, and computer science through interactive problem-solving, "
     "marketed heavily via educational creator sponsorships."),
]


def main() -> None:
    rng = random.Random(SEED)

    # Combine in a fixed order (AI-coding first, then other tracks), then a
    # single deterministic shuffle so the file isn't trivially grouped while
    # remaining byte-identical across runs.
    entries = list(AI_CODING) + list(OTHER_TRACKS)
    rng.shuffle(entries)

    docs = []
    for i, (name, category, funding, stage, text) in enumerate(entries, start=1):
        docs.append(
            {
                "id": f"prod-{i:04d}",
                "text": text,
                "metadata": {
                    # name kept for display; lowercase normalized name aids exact
                    # brand cross-reference with the content index `brand` field.
                    "name": name,
                    "name_norm": name.lower(),
                    "category": category,
                    "funding": funding,
                    "stage": stage,
                },
            }
        )

    OUT_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter

    cats = Counter(d["metadata"]["category"] for d in docs)
    n_ai = sum(1 for d in docs if d["metadata"]["category"] == "ai-coding")
    print(f"Wrote {len(docs)} products to {OUT_PATH}")
    print(f"AI-coding products: {n_ai}")
    print("Categories:", dict(sorted(cats.items())))
    print("\nSample:\n", docs[0]["text"])


if __name__ == "__main__":
    main()
