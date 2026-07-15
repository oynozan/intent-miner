"""Prefilter tests.

The gate's job is to decide the LLM bill. The tests that matter are the ones that
check it actually *kills* at the rate it claims -- a gate that silently passes
everything is invisible until the invoice arrives.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.prefilter import (
    cosine_matrix,
    drop_by_negative_terms,
    normalize,
    select,
    threshold_for_percentile,
)


def test_normalize_makes_rows_unit_length() -> None:
    m = normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)


def test_normalize_survives_zero_vectors() -> None:
    """A zero vector must become a zero row, not NaN. NaN would poison the percentile
    for the whole run, silently."""
    m = normalize(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert not np.isnan(m).any()
    assert np.allclose(m[0], [0.0, 0.0])


def test_cosine_matrix_matches_hand_computed() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 1.0]])
    sims = cosine_matrix(a, b)
    assert sims[0, 0] == pytest.approx(1.0)
    assert sims[0, 1] == pytest.approx(0.7071, abs=1e-3)
    assert sims[1, 0] == pytest.approx(0.0, abs=1e-6)


def test_kill_rate_matches_keep_percentile() -> None:
    """The regression that motivated ranking candidates rather than pairs.

    A global *pair* percentile combined with top-k-per-candidate returned 60,000 pairs
    from 20k candidates at keep=0.15 -- 6,000 scoring calls against a budget of 45.
    Each candidate's best 3 of 300 leaves always sit in its own top 1%, so they cleared
    a global cut regardless of quality.

    Ranking candidates by their best leaf match makes the kill rate mean what it says.
    """
    rng = np.random.default_rng(0)
    candidates = rng.standard_normal((2_000, 64), dtype=np.float32)
    leaves = rng.standard_normal((50, 64), dtype=np.float32)

    pairs = select(candidates, leaves, keep_percentile=0.15, max_leaves_per_candidate=3)
    survivors = {p.candidate_index for p in pairs}

    assert 0.13 * 2_000 <= len(survivors) <= 0.17 * 2_000, (
        f"expected ~15% of candidates to survive, got {len(survivors)}/2000"
    )
    assert len(pairs) < 2_000, "must produce far fewer pairs than candidates x k"


def test_strong_match_survives_and_noise_does_not() -> None:
    """The gate must keep the signal, not merely keep the right *count*."""
    leaf = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    candidates = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],    # identical -> must survive
            [0.9, 0.1, 0.0, 0.0],    # close     -> must survive
            [0.0, 1.0, 0.0, 0.0],    # orthogonal
            [-1.0, 0.0, 0.0, 0.0],   # opposite
        ],
        dtype=np.float32,
    )
    pairs = select(candidates, leaf, keep_percentile=0.5)
    survivors = {p.candidate_index for p in pairs}
    assert 0 in survivors and 1 in survivors
    assert 3 not in survivors


def test_max_leaves_per_candidate_caps_fanout() -> None:
    """A post that weakly matches everything must not burn a scoring slot per leaf."""
    rng = np.random.default_rng(1)
    candidates = rng.standard_normal((100, 32), dtype=np.float32)
    leaves = rng.standard_normal((40, 32), dtype=np.float32)
    pairs = select(candidates, leaves, keep_percentile=1.0, max_leaves_per_candidate=2)
    counts: dict[int, int] = {}
    for p in pairs:
        counts[p.candidate_index] = counts.get(p.candidate_index, 0) + 1
    assert max(counts.values()) <= 2


def test_keep_percentile_one_keeps_every_candidate() -> None:
    """keep=1.0 means no candidate is dropped.

    It does not mean every *pair* is kept: the cutoff becomes the weakest surviving
    candidate's best score, and secondary leaves still have to clear it. So survivors
    are all 10, while pairs land under the 10 x k ceiling. That asymmetry is the point
    of the gate -- a candidate earns its place on its best match, each extra leaf on
    its own.
    """
    rng = np.random.default_rng(2)
    candidates = rng.standard_normal((10, 8), dtype=np.float32)
    leaves = rng.standard_normal((5, 8), dtype=np.float32)
    pairs = select(candidates, leaves, keep_percentile=1.0, max_leaves_per_candidate=5)

    assert len({p.candidate_index for p in pairs}) == 10, "keep=1.0 must drop no candidate"
    assert 0 < len(pairs) <= 50


def test_empty_inputs_return_no_pairs() -> None:
    assert select(np.zeros((0, 8), dtype=np.float32), np.ones((3, 8), dtype=np.float32), 0.15) == []
    assert select(np.ones((3, 8), dtype=np.float32), np.zeros((0, 8), dtype=np.float32), 0.15) == []


def test_invalid_percentile_rejected() -> None:
    with pytest.raises(ValueError):
        threshold_for_percentile(np.array([0.5]), 0.0)
    with pytest.raises(ValueError):
        threshold_for_percentile(np.array([0.5]), 1.5)


# --- negative terms: the precision lever ---

def test_negative_term_matches_whole_word_only() -> None:
    """Substring matching on short terms is a silent lead-shredder: 'ai' would drop
    every post containing email, detail, certain, available..."""
    assert not drop_by_negative_terms("I need help with my email and details", ["ai"])
    assert drop_by_negative_terms("this is about ai models", ["ai"])


def test_negative_term_disambiguates_pain_phrase() -> None:
    body = "How do I stop my python from escaping its enclosure at the reptile shop"
    assert drop_by_negative_terms(body, ["reptile", "enclosure"])
    assert not drop_by_negative_terms("How do I parse JSON in python", ["reptile", "enclosure"])


def test_no_negative_terms_drops_nothing() -> None:
    assert not drop_by_negative_terms("anything at all", [])
    assert not drop_by_negative_terms("anything at all", None)  # type: ignore[arg-type]


def test_empty_body_is_not_dropped_by_negatives() -> None:
    assert not drop_by_negative_terms("", ["python"])
