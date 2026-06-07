"""Regenerate the Claude precomputed KOL quality-score handoff artifact.

This is the *reproducible* path for the file an interactive Claude session
produces by hand: ``agent-py/kol_quality_scores.json``. It reads the KOL library
(``kols.json``), asks an OpenAI-compatible LLM to grade every creator on a fixed
5-field rubric, and writes the result under the same schema. The handcrafted JSON
and this script's output are interchangeable — same schema, same purpose — so the
file can always be rebuilt without a human in the loop.

Public fields only. Scoring sees ``name / handle / platform / niche / tier /
region / language / followers / engagement_pct`` from each KOL's metadata. It
never sees ``price_usd`` or any private/contact field, so a quality score can be
shown publicly without leaking pricing.

Output schema (``kol_quality_scores.json``)
-------------------------------------------
::

    {
      "version": "1",
      "generated_by": "claude-opus",      # or the SCORES_LLM_MODEL used here
      "rubric": "<one-paragraph description of the five scored dimensions>",
      "scores": {
        "<handle>": {
          "audience_quality": 8.5,        # 0-10 followers-vs-engagement health
          "niche_authority": 7.0,         # 0-10 vertical authority in its niche
          "brand_safety": 9.0,            # 0-10 brand-safety of category/platform
          "value_score": 8.0,             # 0-10 follower-engagement-cost value
          "one_liner": "..."              # <=15-word English recommendation blurb
        },
        ...
      }
    }

``scores`` is keyed by ``handle`` and covers every entry in ``kols.json``.

Merge convention (for the index-build terminal)
-----------------------------------------------
When ``build_indexes.py`` rebuilds the ``ansio_kols`` index, flatten
``scores[handle]`` into that KOL's metadata with a ``cs_`` prefix
(``cs_audience_quality``, ``cs_niche_authority``, ``cs_brand_safety``,
``cs_value_score``, ``cs_one_liner``) so the evidence card can read precomputed
Claude quality signals at zero query-time cost — Moss only retrieves, never
recomputes.

Usage
-----
::

    # validate the script + emitted schema WITHOUT calling any LLM:
    uv run python src/precompute_scores.py --dry-run

    # full regeneration (calls the LLM; needs SCORES_LLM_* env vars):
    SCORES_LLM_BASE_URL=... SCORES_LLM_API_KEY=... SCORES_LLM_MODEL=... \
        uv run python src/precompute_scores.py

    # write somewhere else / tune batch size:
    uv run python src/precompute_scores.py --out /tmp/scores.json --batch-size 40

Environment
-----------
- ``SCORES_LLM_BASE_URL`` — OpenAI-compatible ``/v1`` base URL.
- ``SCORES_LLM_API_KEY``  — bearer token for that endpoint.
- ``SCORES_LLM_MODEL``    — model id to grade with (also recorded in
  ``generated_by``). Defaults to ``claude-opus`` when unset.

Secret discipline: only variable NAMES appear here; values live in gitignored
``.env``. The output path is gitignored — the artifact is never committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

KOLS_PATH = Path(__file__).resolve().parent.parent / "kols.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "kol_quality_scores.json"

SCHEMA_VERSION = "1"
DEFAULT_MODEL = "claude-opus"
DEFAULT_BATCH_SIZE = 25

# 0-10 numeric dimensions every KOL must receive, in card-display order.
SCORE_FIELDS = (
    "audience_quality",
    "niche_authority",
    "brand_safety",
    "value_score",
)
# The free-text recommendation blurb shown on the card.
ONE_LINER_FIELD = "one_liner"
MAX_ONE_LINER_WORDS = 15

RUBRIC = (
    "Each KOL is graded on five card-ready dimensions from PUBLIC fields only "
    "(followers, engagement_pct, niche, tier, platform, region, language): "
    "audience_quality (0-10) rates how healthy the follower-size vs "
    "engagement-rate balance is; niche_authority (0-10) rates vertical "
    "credibility within the creator's niche; brand_safety (0-10) rates how "
    "brand-safe the category and platform are; value_score (0-10) rates the "
    "follower-engagement-to-cost value potential; one_liner is a <=15-word "
    "English recommendation blurb shown verbatim on the evidence card."
)

# Public-only fields handed to the grader. price_usd and contact fields excluded.
_PUBLIC_FIELDS = (
    "name",
    "handle",
    "platform",
    "niche",
    "tier",
    "region",
    "language",
    "followers",
    "engagement_pct",
)

_SYSTEM_PROMPT = (
    "You are a KOL (influencer) quality analyst. Grade each creator on the "
    "rubric below using ONLY the public fields provided. Return STRICT JSON, no "
    "prose.\n\n"
    f"{RUBRIC}\n\n"
    "Output a JSON object mapping each creator's handle to an object with keys: "
    f"{', '.join(SCORE_FIELDS)} (numbers 0-10, one decimal place), and "
    f"{ONE_LINER_FIELD} (string, at most {MAX_ONE_LINER_WORDS} English words). "
    "Include every handle you were given. Example: "
    '{"someHandle": {"audience_quality": 8.5, "niche_authority": 7.0, '
    '"brand_safety": 9.0, "value_score": 8.0, "one_liner": "Strong mid-tier '
    'creator with reliable engagement."}}'
)


def _load_kols() -> list[dict[str, Any]]:
    """Load and shallow-validate ``kols.json`` (a JSON array of KOL docs)."""
    if not KOLS_PATH.exists():
        raise FileNotFoundError(f"KOL data file not found: {KOLS_PATH}")
    data = json.loads(KOLS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{KOLS_PATH.name} must be a JSON array")
    return [d for d in data if isinstance(d, dict)]


def _public_view(meta: dict[str, Any]) -> dict[str, Any]:
    """Return only the public fields the grader is allowed to see."""
    return {k: meta[k] for k in _PUBLIC_FIELDS if k in meta}


def _handles(kols: list[dict[str, Any]]) -> list[str]:
    """Ordered, de-duplicated handles present in the library."""
    seen: set[str] = set()
    out: list[str] = []
    for kol in kols:
        handle = (kol.get("metadata") or {}).get("handle")
        if handle and handle not in seen:
            seen.add(handle)
            out.append(str(handle))
    return out


def _coerce_score_entry(handle: str, raw: Any) -> dict[str, Any]:
    """Validate/normalize one grader entry; fail fast on malformed output."""
    if not isinstance(raw, dict):
        raise ValueError(f"score entry for {handle!r} is not an object: {raw!r}")
    entry: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        if field not in raw:
            raise ValueError(f"score entry for {handle!r} missing {field!r}")
        value = float(raw[field])
        entry[field] = round(max(0.0, min(10.0, value)), 1)
    one_liner = str(raw.get(ONE_LINER_FIELD, "")).strip()
    if not one_liner:
        raise ValueError(f"score entry for {handle!r} missing {ONE_LINER_FIELD!r}")
    entry[ONE_LINER_FIELD] = one_liner
    return entry


def validate_scores(scores: dict[str, Any], handles: list[str]) -> None:
    """Raise if ``scores`` does not fully cover ``handles`` under the schema."""
    missing = [h for h in handles if h not in scores]
    if missing:
        raise ValueError(f"scores missing {len(missing)} handle(s), e.g. {missing[:5]}")
    for handle in handles:
        _coerce_score_entry(handle, scores[handle])


def build_envelope(scores: dict[str, Any], model: str) -> dict[str, Any]:
    """Wrap a ``handle -> entry`` map in the versioned output schema."""
    return {
        "version": SCHEMA_VERSION,
        "generated_by": model,
        "rubric": RUBRIC,
        "scores": scores,
    }


def _build_client(base_url: str, api_key: str) -> Any:
    """Construct an OpenAI-compatible client (imported lazily for --dry-run)."""
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def _grade_batch(
    client: Any, model: str, batch: list[dict[str, Any]]
) -> dict[str, Any]:
    """Send one batch of public KOL views to the LLM and parse its JSON reply."""
    payload = json.dumps(batch, ensure_ascii=False)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Grade these creators:\n{payload}"},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM did not return a JSON object")
    # Some gateways wrap the map under a "scores" key; unwrap if so.
    if "scores" in parsed and isinstance(parsed["scores"], dict):
        parsed = parsed["scores"]
    return parsed


def generate_scores(model: str, batch_size: int) -> dict[str, Any]:
    """Grade every KOL via the configured LLM; return a validated score map."""
    base_url = os.getenv("SCORES_LLM_BASE_URL")
    api_key = os.getenv("SCORES_LLM_API_KEY")
    if not base_url or not api_key:
        raise SystemExit(
            "SCORES_LLM_BASE_URL and SCORES_LLM_API_KEY must be set "
            "(or use --dry-run to validate without calling the LLM)."
        )
    kols = _load_kols()
    handles = _handles(kols)
    by_handle = {
        str((k.get("metadata") or {}).get("handle")): _public_view(
            k.get("metadata") or {}
        )
        for k in kols
        if (k.get("metadata") or {}).get("handle")
    }
    client = _build_client(base_url, api_key)

    scores: dict[str, Any] = {}
    for start in range(0, len(handles), batch_size):
        chunk = handles[start : start + batch_size]
        batch = [by_handle[h] for h in chunk]
        raw = _grade_batch(client, model, batch)
        for handle in chunk:
            if handle not in raw:
                raise ValueError(f"LLM omitted handle {handle!r} from its batch reply")
            scores[handle] = _coerce_score_entry(handle, raw[handle])
        print(
            f"  graded {min(start + batch_size, len(handles))}/{len(handles)}",
            file=sys.stderr,
        )

    validate_scores(scores, handles)
    return scores


def _dry_run() -> int:
    """Validate inputs + schema using synthetic scores; never calls the LLM."""
    kols = _load_kols()
    handles = _handles(kols)
    placeholder = dict.fromkeys(SCORE_FIELDS, 5.0) | {
        ONE_LINER_FIELD: "Placeholder recommendation blurb for schema validation."
    }
    scores = {h: dict(placeholder) for h in handles}
    validate_scores(scores, handles)
    envelope = build_envelope(scores, DEFAULT_MODEL)
    # Round-trip the envelope to prove it serializes cleanly.
    json.dumps(envelope, ensure_ascii=False)
    fields = [*SCORE_FIELDS, ONE_LINER_FIELD]
    print(
        f"[dry-run] OK — {len(handles)} handles, schema v{SCHEMA_VERSION}, "
        f"fields={fields}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and emitted schema without calling the LLM.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help=f"Output path (default: {OUT_PATH}).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SCORES_LLM_MODEL", DEFAULT_MODEL),
        help="Model id to grade with and record in generated_by.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"KOLs per LLM request (default: {DEFAULT_BATCH_SIZE}).",
    )
    args = parser.parse_args()

    if args.dry_run:
        return _dry_run()

    scores = generate_scores(args.model, args.batch_size)
    envelope = build_envelope(scores, args.model)
    args.out.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(scores)} scores -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
