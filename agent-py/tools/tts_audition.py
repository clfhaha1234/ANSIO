"""ANSIO TTS audition — synthesize the same bilingual (中英混说) consultant
script across a shortlist of MiniMax voices so the team can listen and pick the
"professional growth-engineer" voice in seconds.

This is NOT a unit test. It hits MiniMax's T2A v2 HTTP endpoint directly (the
same backend the LiveKit `minimax` plugin uses) with `MINIMAX_API_KEY`, writes
one mp3 per candidate voice to `tts_samples/`, and prints a table of voice id +
characteristics. With no key it degrades gracefully: it prints the candidate
shortlist + the selection recommendation and exits 0 (never fatal), so the doc
artifact is still useful offline.

Run (ENV_FILE convention matches the rest of the repo's harnesses):

    ENV_FILE=.env uv run python tools/tts_audition.py

Pick a subset / override the script:

    ENV_FILE=.env TTS_VOICES="English_Persuasive_Man,English_WiseScholar" \
        uv run python tools/tts_audition.py
    ENV_FILE=.env TTS_SCRIPT="Hi 你好, 这是测试。" uv run python tools/tts_audition.py

Why direct HTTP (not the plugin): the plugin streams into a LiveKit audio sink;
for an offline "save N samples to disk" audition the documented batch endpoint
`POST {base}/v1/t2a_v2` with `output_format=hex` is simpler and gives us the
raw bytes to write. Endpoint/host/model literals are taken from the installed
plugin (livekit/plugins/minimax/tts.py) and docs/research/03-*.md, not memory.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_FILE", ".env.local"))

# --- config (no secrets; only the variable NAME is referenced) ---------------
# The key value itself is never printed or logged — only its presence.
API_KEY_ENV = "MINIMAX_API_KEY"
API_KEY = os.getenv(API_KEY_ENV)

# Host + endpoint mirror the installed LiveKit minimax plugin defaults
# (api-uw = US-West low-latency edge). T2A v2 batch synthesis.
MINIMAX_BASE_URL = os.getenv("MINIMAX_TTS_BASE_URL", "https://api-uw.minimax.io")
T2A_ENDPOINT = f"{MINIMAX_BASE_URL}/v1/t2a_v2"

# speech-2.6-turbo: low-latency realtime tier that also supports the "fluent"
# emotion (only 2.6-* per the plugin note); good audition default. agent.py
# currently runs speech-2.8-turbo — both are turbo (low-latency) tiers; we
# audition on 2.6-turbo so `emotion="fluent"` is available, and recommend the
# final model in the report.
MODEL = os.getenv("TTS_AUDITION_MODEL", "speech-2.6-turbo")

# Default bilingual consultant script (中英混说, professional growth-engineer
# tone). Override with TTS_SCRIPT.
DEFAULT_SCRIPT = (
    "Hi, I am ANSIO, your growth engineer. "
    "我帮你找被低估的达人——比如这位 Travis Media, CPM 才 97. "
    "Let me pull the numbers for you."
)
SCRIPT = os.getenv("TTS_SCRIPT", DEFAULT_SCRIPT)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tts_samples")


@dataclass(frozen=True)
class Candidate:
    """A voice candidate with the rationale shown to the team."""

    voice_id: str
    gender: str
    note: str


# Shortlist curated for a bilingual professional-consultant persona. All ids are
# from the installed plugin's TTSVoice literal (static-analysis-safe) EXCEPT the
# one free-string id noted below, which the plugin accepts as `str` (the plugin
# types `voice: TTSVoice | str`). The repo's agent.py already ships such a
# free-string id (English_CalmWoman).
CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "English_Persuasive_Man",
        "male",
        "Confident, persuasive sales/advisor energy — closes on numbers. "
        "Top pick for a 'growth engineer who pitches undervalued creators'.",
    ),
    Candidate(
        "English_Insightful_Speaker",
        "male",
        "Measured, analytical, TED-talk cadence — reads data points credibly.",
    ),
    Candidate(
        "English_WiseScholar",
        "male",
        "Calm authority, trustworthy expert tone — good if we want gravitas "
        "over hustle.",
    ),
    Candidate(
        "English_Explanatory_Man",
        "male",
        "Clear teacher voice — explains CPM/benchmark math without sounding "
        "salesy.",
    ),
    Candidate(
        # In the plugin's TTSVoice literal (static-analysis-safe), so it is
        # guaranteed to resolve on the account. Warm, bright female voice — the
        # consultative female counterpart to the current English_CalmWoman for
        # an A/B. This is the SAFE default female option.
        "English_radiant_girl",
        "female",
        "Warm, bright female voice (in plugin literal — guaranteed to resolve). "
        "Consultative female A/B vs the current English_CalmWoman.",
    ),
    Candidate(
        # Free-string catalog id (NOT in plugin literal; plugin types
        # voice: TTSVoice | str so it is still accepted). More overtly
        # "confident female consultant" timbre IF it resolves on your account —
        # if synthesis returns a voice-not-found status, fall back to
        # English_radiant_girl above.
        "English_ConfidentWoman",
        "female",
        "Warm-but-sharp female consultant. NOTE: free-string id (not in plugin "
        "literal) — verify it resolves; else use English_radiant_girl.",
    ),
)


def _selected_candidates() -> list[Candidate]:
    """Honor TTS_VOICES override (comma list of voice ids), else full shortlist."""
    override = os.getenv("TTS_VOICES", "").strip()
    if not override:
        return list(CANDIDATES)
    wanted = [v.strip() for v in override.split(",") if v.strip()]
    by_id = {c.voice_id: c for c in CANDIDATES}
    out: list[Candidate] = []
    for v in wanted:
        out.append(by_id.get(v, Candidate(v, "?", "custom voice id (override)")))
    return out


def _print_shortlist(cands: list[Candidate]) -> None:
    print("Candidate voices (bilingual professional-consultant shortlist):")
    print(f"  {'voice_id':<28} {'gender':<7} characteristics")
    print(f"  {'-' * 28} {'-' * 7} {'-' * 50}")
    for c in cands:
        print(f"  {c.voice_id:<28} {c.gender:<7} {c.note}")


def _recommendation() -> None:
    print()
    print("Recommendation (professional consultant, 中英双语):")
    print("  1st: English_Persuasive_Man — growth-engineer persona, sells the")
    print("       undervalued-creator insight with conviction.")
    print("  2nd: English_Insightful_Speaker — if Persuasive feels too salesy;")
    print("       analytical, credible on the numbers.")
    print("  Config: model=speech-2.8-turbo (or 2.6-turbo for emotion=fluent),")
    print("          language_boost='auto' (中英 code-switch), emotion='neutral'")
    print("          or 'fluent' (2.6 only), speed=1.0.")


def _synthesize(voice_id: str) -> bytes | None:
    """Call T2A v2 once; return audio bytes or None on any failure (degrade)."""
    import requests  # local import so the --no-key path needs no deps

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "text": SCRIPT,
        "stream": False,
        "language_boost": "auto",
        "output_format": "hex",
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
            "emotion": "neutral",
        },
        "audio_setting": {
            "sample_rate": 24000,
            "format": "mp3",
            "bitrate": 128000,
        },
    }
    try:
        resp = requests.post(T2A_ENDPOINT, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        base_resp = body.get("base_resp", {})
        status_code = base_resp.get("status_code", 0)
        if status_code != 0:
            # Do NOT print the key; only the API's own status message.
            print(
                f"  ! {voice_id}: API status {status_code} "
                f"({base_resp.get('status_msg', 'unknown')})"
            )
            return None
        audio_hex = (body.get("data") or {}).get("audio")
        if not audio_hex:
            print(f"  ! {voice_id}: no audio in response")
            return None
        return bytes.fromhex(audio_hex)
    except Exception as e:  # noqa: BLE001 — never fatal per CLAUDE.md
        print(f"  ! {voice_id}: synthesis failed ({type(e).__name__}: {e})")
        return None


def main() -> int:
    cands = _selected_candidates()

    print("=" * 70)
    print("ANSIO TTS audition — MiniMax bilingual consultant voices")
    print("=" * 70)
    print(f"Script: {SCRIPT!r}")
    print(f"Model:  {MODEL}   Endpoint: {T2A_ENDPOINT}")
    print()
    _print_shortlist(cands)

    if not API_KEY:
        print()
        print(f"[degraded] {API_KEY_ENV} not set — no audio synthesized.")
        print("           Showing shortlist + recommendation only.")
        _recommendation()
        print()
        print("To synthesize: ENV_FILE=.env uv run python tools/tts_audition.py")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    print()
    print(f"Synthesizing {len(cands)} samples -> {OUT_DIR}/")
    results: list[tuple[str, str | None]] = []
    for c in cands:
        audio = _synthesize(c.voice_id)
        if audio is None:
            results.append((c.voice_id, None))
            continue
        path = os.path.join(OUT_DIR, f"{c.voice_id}.mp3")
        try:
            with open(path, "wb") as f:
                f.write(audio)
            print(f"  ok  {c.voice_id:<28} -> {path} ({len(audio):,} bytes)")
            results.append((c.voice_id, path))
        except OSError as e:
            print(f"  ! {c.voice_id}: write failed ({e})")
            results.append((c.voice_id, None))

    ok = [r for r in results if r[1]]
    print()
    print(f"Done: {len(ok)}/{len(results)} samples written.")
    _recommendation()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
