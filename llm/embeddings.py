"""Voyage embeddings.

voyage-4-lite @ 1024 dims. Two things about this are not defaults and must not be
allowed to become implicit:

* **output_dimension is passed explicitly, always.** The Voyage 4 family supports
  256/512/1024/2048. The API defaults to 1024 but the HuggingFace weights default to
  2048 -- so an implicit default means a local dev path and the API path can silently
  produce vectors that do not share a space. The failure is not an error; it is
  quietly meaningless cosine values.
* **1536 does not exist here.** The original spec's ``vector(1536)`` was an OpenAI
  ada-002/3-small leftover. There is no 1536 option in this family.

``input_type`` is a genuine open question, not a settled default. Voyage's FAQ says
"do not omit input_type", but that guidance is scoped to retrieval/RAG, and their own
model cards use an identical prompt on both sides for STS-style symmetric tasks. Our
task -- a pain description against a forum post -- is arguably symmetric (two pain
descriptions) rather than query/document. It is configurable so the calibration sweep
can settle it with measured recall instead of an assumption.
"""

from __future__ import annotations

import logging
from typing import Literal, Sequence

from core.config import settings

log = logging.getLogger(__name__)

InputType = Literal["query", "document"] | None

_client = None


def client():
    global _client
    if _client is None:
        import voyageai

        _client = voyageai.Client(api_key=settings().voyage_api_key or None)
    return _client


def embed(texts: Sequence[str], input_type: InputType = None, batch_size: int = 128) -> list[list[float]]:
    """Embed texts, preserving input order.

    Order preservation is load-bearing: the prefilter zips these against candidate rows
    positionally, so a reordering would silently attach every score to the wrong post.
    """
    if not texts:
        return []

    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        result = client().embed(
            chunk,
            model=settings().embed_model,
            input_type=input_type,
            output_dimension=settings().embed_dimension,  # never implicit -- see module docstring
        )
        out.extend(result.embeddings)

    if len(out) != len(texts):
        raise RuntimeError(f"embedding count mismatch: got {len(out)} for {len(texts)} inputs")
    return out
