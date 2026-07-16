"""Embeddings, with primary -> fallback like the LLM calls.

OpenAI (text-embedding-3-small) is primary; Voyage (voyage-4-lite) is the fallback.
gpt-5-nano cannot embed, so "OpenAI for everything" means the embedding model here,
reduced to 1024 dims via OpenAI's ``dimensions`` param -- which keeps the vector(1024)
schema unchanged.

``output_dimension`` / ``dimensions`` is passed explicitly on every call, both
providers. OpenAI reduces to it; Voyage's HF weights default to 2048 while its API
defaults to 1024. An implicit default lets dev and prod produce vectors that do not
share a space -- silent, just meaningless cosines.

``input_type`` (query vs document) is a Voyage concept; OpenAI text-embedding-3 has no
such distinction and ignores it. The prefilter normalizes vectors regardless of
provider, so the two are interchangeable at the cosine level even though only Voyage's
arm of the calibration sweep can vary ``input_type``.
"""

from __future__ import annotations

import logging
from typing import Literal, Sequence

from core.config import settings

log = logging.getLogger(__name__)

InputType = Literal["query", "document"] | None

_openai_client = None
_voyage_client = None


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=settings().openai_api_key or None)
    return _openai_client


def _voyage():
    global _voyage_client
    if _voyage_client is None:
        import voyageai

        _voyage_client = voyageai.Client(api_key=settings().voyage_api_key or None)
    return _voyage_client


def _openai_embed(texts: list[str], input_type: InputType, batch_size: int) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        result = _openai().embeddings.create(
            model=settings().openai_embed_model,
            input=chunk,
            dimensions=settings().embed_dimension,  # never implicit -- see module docstring
        )
        # Sort by index: the API returns objects carrying their input position, and
        # relying on incidental ordering would silently misalign vectors to rows.
        for item in sorted(result.data, key=lambda d: d.index):
            out.append(item.embedding)
    return out


def _voyage_embed(texts: list[str], input_type: InputType, batch_size: int) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        result = _voyage().embed(
            chunk,
            model=settings().embed_model,
            input_type=input_type,
            output_dimension=settings().embed_dimension,
        )
        out.extend(result.embeddings)
    return out


# Names in a stable list. Dispatch resolves the embedder from module globals at call
# time (see _run) so a test can monkeypatch _openai_embed / _voyage_embed directly.
EMBED_PROVIDERS = ("voyage", "openai")


def _run(provider: str, texts: list[str], input_type: InputType, batch_size: int) -> list[list[float]]:
    return {"openai": _openai_embed, "voyage": _voyage_embed}[provider](texts, input_type, batch_size)


def _configured(provider: str) -> bool:
    s = settings()
    if provider == "openai":
        return bool(s.openai_api_key)
    if provider == "voyage":
        return bool(s.voyage_api_key)
    return False


def _order() -> list[str]:
    primary = settings().embed_provider
    if primary not in EMBED_PROVIDERS:
        primary = "openai"
    return [primary] + [p for p in EMBED_PROVIDERS if p != primary]


def embed(texts: Sequence[str], input_type: InputType = None, batch_size: int = 128) -> list[list[float]]:
    """Embed texts, preserving input order, via the primary provider then the fallback.

    Order preservation is load-bearing: the prefilter zips these against candidate rows
    positionally, so a reordering silently attaches every score to the wrong post.
    """
    if not texts:
        return []
    texts = list(texts)
    order = _order()

    attempted: list[str] = []
    last_exc: Exception | None = None
    for provider in order:
        if not _configured(provider):
            log.info("embed: skipping %s (no api key)", provider)
            continue
        attempted.append(provider)
        try:
            out = _run(provider, texts, input_type, batch_size)
        except Exception as exc:  # noqa: BLE001 -- any failure should try the fallback
            last_exc = exc
            log.warning("embed: provider %s failed (%s); trying next", provider, exc)
            continue
        if provider != order[0]:
            log.warning("embed: served by fallback provider %s", provider)
        if len(out) != len(texts):
            raise RuntimeError(f"embedding count mismatch: got {len(out)} for {len(texts)} inputs")
        return out

    if not attempted:
        raise RuntimeError(
            "no embedding provider configured -- set OPENAI_API_KEY (primary) or VOYAGE_API_KEY (fallback)"
        )
    raise RuntimeError(f"all embedding providers failed {attempted}: {last_exc}")
