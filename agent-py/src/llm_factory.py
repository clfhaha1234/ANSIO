"""LLM factory — env-driven three-config switch for the ANSIO voice agent.

One knob, `LLM_PROVIDER`, picks the conversational brain at construction time:

- ``minimax``   — MiniMax-M2 via its OpenAI-compatible endpoint
                  (`MINIMAX_API_KEY`, `MINIMAX_BASE_URL`, `MINIMAX_MODEL`).
                  Also the model that drives the demo TTS, so it always has a
                  political seat in the narrative even when not the brain.
- ``claude``    — sub2api local relay: an Anthropic-account-backed OpenAI-
                  compatible gateway on `ANTHROPIC_BASE_URL` (default
                  http://localhost:9090), bearer = `ANTHROPIC_AUTH_TOKEN`.
                  We reach it through the LiveKit `openai` plugin's `base_url`
                  path — measured to speak OpenAI `/v1/chat/completions` with
                  streaming + tool-calls.
- ``inference`` — the already-verified LiveKit Inference fallback
                  (gpt-4.1-mini). No provider key required.

KISS + lazy: this module imports nothing from livekit at import time, and
`build_llm()` only imports the plugin it actually needs. That keeps the agent's
pure logic importable offline for unit tests (PRD digest T10) and avoids paying
for a provider you didn't select.

Secret discipline: only variable NAMES appear here; values live in the
gitignored `.env`.
"""

from __future__ import annotations

import os

# Defaults are safe placeholders only; real values come from `.env`.
_MINIMAX_BASE_URL_DEFAULT = "https://api.minimax.io/v1"
_MINIMAX_MODEL_DEFAULT = "MiniMax-M2"
_CLAUDE_BASE_URL_DEFAULT = "http://localhost:9090"
# Low-latency Claude tier for voice; overridable via CLAUDE_MODEL.
_CLAUDE_MODEL_DEFAULT = "claude-haiku-4-5-20251001"
_INFERENCE_MODEL_DEFAULT = "openai/gpt-4.1-mini"


def resolve_provider(override: str | None = None) -> str:
    """Return the normalized provider name (minimax | claude | inference)."""
    provider = (override or os.getenv("LLM_PROVIDER", "minimax")).strip().lower()
    if provider not in {"minimax", "claude", "inference"}:
        provider = "minimax"
    return provider


def _claude_base_url() -> str:
    """sub2api relay base, normalized to an OpenAI-compatible `/v1` root.

    The relay is mounted at ANTHROPIC_BASE_URL (e.g. http://localhost:9090) and
    exposes `/v1/chat/completions`. The LiveKit openai plugin appends the route
    itself, so we hand it the `/v1` root.
    """
    base = os.getenv("ANTHROPIC_BASE_URL", _CLAUDE_BASE_URL_DEFAULT).rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def build_llm(provider: str | None = None):
    """Construct and return a LiveKit LLM for the selected provider.

    Imports are deferred so selecting one provider never drags in another, and so
    this module stays importable without livekit installed (offline unit tests).
    """
    provider = resolve_provider(provider)

    if provider == "claude":
        # sub2api local relay, OpenAI-compatible. Auth via ANTHROPIC_AUTH_TOKEN.
        from livekit.plugins import openai

        return openai.LLM(
            model=os.getenv("CLAUDE_MODEL", _CLAUDE_MODEL_DEFAULT),
            base_url=_claude_base_url(),
            api_key=os.getenv("ANTHROPIC_AUTH_TOKEN"),
        )

    if provider == "minimax":
        # MiniMax-M2 via OpenAI-compatible endpoint. Auth via MINIMAX_API_KEY.
        from livekit.plugins import openai

        return openai.LLM(
            model=os.getenv("MINIMAX_MODEL", _MINIMAX_MODEL_DEFAULT),
            base_url=os.getenv("MINIMAX_BASE_URL", _MINIMAX_BASE_URL_DEFAULT),
            api_key=os.getenv("MINIMAX_API_KEY"),
        )

    # inference: already-verified LiveKit Inference fallback (gpt-4.1-mini).
    from livekit.agents import inference

    return inference.LLM(
        model=os.getenv("FALLBACK_LLM_MODEL", _INFERENCE_MODEL_DEFAULT)
    )


def model_label(provider: str | None = None) -> str:
    """Human-readable model id for logs/headers (no secrets)."""
    provider = resolve_provider(provider)
    if provider == "claude":
        return os.getenv("CLAUDE_MODEL", _CLAUDE_MODEL_DEFAULT)
    if provider == "minimax":
        return os.getenv("MINIMAX_MODEL", _MINIMAX_MODEL_DEFAULT)
    return os.getenv("FALLBACK_LLM_MODEL", _INFERENCE_MODEL_DEFAULT)
