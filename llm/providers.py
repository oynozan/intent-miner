"""Provider-agnostic structured-JSON completion, with primary -> fallback.

OpenAI (gpt-5-nano) is primary; Anthropic is the optional fallback, tried only when
the primary errors or has no key. Both calls are the same shape -- a system prompt, a
user message, and a JSON schema the output must satisfy -- so the schema is written
once and formatted per provider here.

Two provider facts drive the per-provider code:

* **OpenAI gpt-5-nano is a reasoning model.** The token cap is ``max_completion_tokens``
  (``max_tokens`` is rejected), depth is ``reasoning_effort`` (low|medium|high), and
  temperature/top_p must be omitted -- a strict reasoning endpoint rejects them even at
  their default value. Structured output is ``response_format`` with a strict
  ``json_schema`` (all properties required, ``additionalProperties: false`` -- the
  schemas in llm/client.py already comply, because Anthropic strict mode wants the same).

* **Anthropic** uses ``output_config.format`` and adaptive thinking. Same schema dict,
  different envelope.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config import settings

log = logging.getLogger(__name__)

# Provider names in a stable list. Dispatch resolves the caller from module globals at
# call time (see _dispatch) rather than freezing function references in a dict, so a
# test can monkeypatch _openai_json / _anthropic_json without also rewiring a table.
PROVIDER_NAMES = ("openai", "anthropic")

_openai_client = None
_anthropic_client = None


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=settings().openai_api_key or None)
    return _openai_client


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic(api_key=settings().anthropic_api_key or None)
    return _anthropic_client


def _openai_json(
    *, system: str, user: str, schema: dict, schema_name: str, model: str, effort: str, max_tokens: int
) -> dict[str, Any]:
    resp = _openai().chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,   # reasoning models reject max_tokens
        reasoning_effort=effort,            # low | medium | high
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        # No temperature / top_p on purpose: reasoning models reject non-default
        # sampling, and a strict endpoint can reject the field even set to its default.
    )
    choice = resp.choices[0]
    if getattr(choice.message, "refusal", None):
        raise RuntimeError(f"openai refused: {choice.message.refusal}")
    if choice.finish_reason == "length":
        # Reasoning tokens are billed to the same budget, so this means the tree/scores
        # were cut off mid-JSON. json.loads would fail anyway; fail with the real cause.
        raise RuntimeError(f"openai truncated -- raise max_completion_tokens (was {max_tokens})")
    return json.loads(choice.message.content)


def _anthropic_json(
    *, system: str, user: str, schema: dict, schema_name: str, model: str, effort: str, max_tokens: int
) -> dict[str, Any]:
    resp = _anthropic().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"anthropic refused: {resp.stop_details}")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def _dispatch(provider: str, **kwargs: Any) -> dict[str, Any]:
    # Resolve from module globals at call time so monkeypatching the caller works.
    return {"openai": _openai_json, "anthropic": _anthropic_json}[provider](**kwargs)


def _configured(provider: str) -> bool:
    s = settings()
    if provider == "openai":
        return bool(s.openai_api_key)
    if provider == "anthropic":
        return bool(s.anthropic_api_key)
    return False


def provider_order() -> list[str]:
    """Primary first, then the remaining known providers as fallbacks."""
    primary = settings().llm_provider
    if primary not in PROVIDER_NAMES:
        primary = "openai"
    return [primary] + [p for p in PROVIDER_NAMES if p != primary]


def complete_json(
    *, system: str, user: str, schema: dict, schema_name: str, params: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Call the primary provider; fall back to the next configured one on any error.

    ``params`` maps provider name -> that provider's call kwargs (model, effort,
    max_tokens), so a provider only exists in the order if the caller supplied kwargs
    for it. A provider with no API key is skipped, not attempted.

    A refusal is treated as a normal failure and falls through to the fallback: for
    this benign task a refusal is almost certainly spurious, and resilience is the
    whole reason a fallback exists. If every provider refuses, the last refusal is
    surfaced in the final error.
    """
    attempted: list[str] = []
    last_exc: Exception | None = None
    order = provider_order()

    for provider in order:
        if provider not in params:
            continue
        if not _configured(provider):
            log.info("llm: skipping %s (no api key)", provider)
            continue
        attempted.append(provider)
        try:
            result = _dispatch(
                provider, system=system, user=user, schema=schema, schema_name=schema_name, **params[provider]
            )
            if provider != order[0]:
                log.warning("llm: served by fallback provider %s", provider)
            return result
        except Exception as exc:  # noqa: BLE001 -- any failure should try the fallback
            last_exc = exc
            log.warning("llm: provider %s failed (%s); trying next", provider, exc)

    if not attempted:
        raise RuntimeError(
            "no LLM provider configured -- set OPENAI_API_KEY (primary) or ANTHROPIC_API_KEY (fallback)"
        )
    raise RuntimeError(f"all LLM providers failed {attempted}: {last_exc}")
