"""The prefilter: the cost gate.

This stage decides the LLM bill. It runs before any scoring model sees anything, and
its kill rate is the single biggest lever on run cost.

**Why numpy and not pgvector.** 20k candidates x 1024 dims against ~300 leaves is one
matmul: 12.29 GFLOP, ~48ms measured. The same 6M pairs computed pair-by-pair takes
~31.6s -- 387x slower -- because GEMM amortizes memory traffic through cache blocking
and per-pair distance cannot. pgvector has no batch/GEMM path, so routing this through
the database means paying the round trip *and* the slower algorithm. The embeddings are
already in-process at the moment they are generated; the DB write is for later
re-calibration, not for this computation.

**Why no HNSW index.** pgvector only uses an approximate index when the query is
ORDER BY <distance> LIMIT n. This is a threshold gate over the whole run with no
top-k, so an index would never be consulted -- and approximate recall is disqualifying
for a recall-critical gate regardless.

**Why a percentile, not a cosine.** Absolute cosine values shift with model,
input_type, dimension, and dtype, and embedding anisotropy makes them systematically
miscalibrated while leaving rank order intact. A hardcoded 0.72 silently means
something different after any of those change; a percentile does not. Trust the
ranking, not the number.

**Tune for recall.** A false negative is gone forever -- nothing downstream can
recover a candidate this stage drops. A false positive costs one cheap scoring slot.
The asymmetry is enormous and the threshold should reflect it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Pair:
    candidate_index: int
    node_index: int
    cosine: float


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product is a cosine.

    Zero vectors would divide by zero; they become zero rows, which score 0 against
    everything and get filtered rather than producing NaN that silently poisons the
    percentile.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected 2-D, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_matrix(candidates: np.ndarray, leaves: np.ndarray) -> np.ndarray:
    """(n_candidates, n_leaves) cosine similarities. One GEMM."""
    return normalize(candidates) @ normalize(leaves).T


def threshold_for_percentile(best_per_candidate: np.ndarray, keep_percentile: float) -> float:
    """The cosine cut that keeps the top ``keep_percentile`` of CANDIDATES.

    The percentile is taken over each candidate's *best* leaf score, not over all
    candidate x leaf pairs. That distinction is the whole gate, and getting it wrong
    is silent:

    A global pair percentile does not survive contact with top-k-per-candidate. Every
    candidate's best 3 of ~300 leaves sit in its own top 1%, so they clear a global
    85th-percentile cut almost regardless of whether the candidate is any good.
    Measured on 20k x 300: a global pair cut at keep=0.15 returned 60,000 pairs --
    6,000 scoring calls where the cost model budgets 45. The gate did not gate.

    Ranking candidates by their best match asks the right question: "is this post's
    strongest pain match good enough to pay a model to read it?"
    """
    if not 0 < keep_percentile <= 1:
        raise ValueError(f"keep_percentile must be in (0, 1], got {keep_percentile}")
    if best_per_candidate.size == 0:
        return 1.0
    return float(np.quantile(best_per_candidate, 1.0 - keep_percentile))


def select(
    candidate_vectors: np.ndarray,
    leaf_vectors: np.ndarray,
    keep_percentile: float,
    max_leaves_per_candidate: int = 3,
) -> list[Pair]:
    """Drop candidates whose best leaf match is weak; keep survivors' top leaves.

    The unit of survival is the **candidate**, because the unit of cost downstream is
    the post: score_batch reads a post once per pair, and the run's bill is driven by
    how many posts get read at all.

    ``max_leaves_per_candidate`` then caps fan-out among survivors. One post genuinely
    can answer several leaves, but without a cap a generic post that matches everything
    weakly would burn a scoring slot per leaf.
    """
    if candidate_vectors.size == 0 or leaf_vectors.size == 0:
        return []

    sims = cosine_matrix(candidate_vectors, leaf_vectors)
    best_per_candidate = sims.max(axis=1)
    cutoff = threshold_for_percentile(best_per_candidate, keep_percentile)

    survivors = np.nonzero(best_per_candidate >= cutoff)[0]
    k = min(max_leaves_per_candidate, sims.shape[1])

    pairs: list[Pair] = []
    for ci in survivors:
        row = sims[ci]
        # argpartition beats a full sort: we want the top-k, not the order of the rest.
        top = np.argpartition(row, -k)[-k:] if k < row.shape[0] else np.arange(row.shape[0])
        for ni in top:
            # A surviving candidate's *secondary* leaves must clear the bar on their own
            # merit, or every survivor drags in k-1 weak pairs for free.
            if row[ni] >= cutoff:
                pairs.append(Pair(candidate_index=int(ci), node_index=int(ni), cosine=float(row[ni])))

    log.info(
        "prefilter: %d candidates x %d leaves -> %d survivors, %d pairs "
        "(cutoff %.4f, keep %.0f%%, kill %.1f%%)",
        sims.shape[0], sims.shape[1], len(survivors), len(pairs), cutoff,
        keep_percentile * 100, 100 * (1 - len(survivors) / sims.shape[0]),
    )
    return pairs


def drop_by_negative_terms(body: str, negative_terms: Sequence[str]) -> bool:
    """True if the body should be dropped for this leaf.

    Runs before the cosine gate. Negative terms are the precision lever for ambiguous
    pain phrases -- "python" the snake, "mask" the skincare product. A wrong negative
    term silently deletes real leads, which is why the prompt tells the model to leave
    the list empty rather than invent weak filters.

    Word-boundary matched, not substring: a substring check on "ai" would drop every
    post containing "email", "detail", or "certain".
    """
    if not negative_terms or not body:
        return False
    import re

    haystack = body.lower()
    for term in negative_terms:
        term = term.strip().lower()
        if not term:
            continue
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", haystack):
            return True
    return False
