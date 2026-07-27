"""Two levers tuned from measured run data: the engagement cap, and LinkedIn's share.

Both exist because of numbers from real runs rather than taste, so the tests assert the
*property* that motivated them -- fit beats popularity, and LinkedIn is bounded -- rather
than the constants, which are meant to be tunable.
"""

from __future__ import annotations

import pytest

from core.config import settings
from pipeline import actors


def _pair(engagement: int | None = None, posted_at=None) -> dict:
    return {"engagement": engagement, "posted_at": posted_at}


def _scored(pain: int, icp: int) -> dict:
    return {"is_seeking": True, "actionable": True, "pain_match": pain, "icp_match": icp}


# --- engagement cap ----------------------------------------------------------------

def test_a_viral_weak_match_cannot_outrank_a_quiet_strong_one() -> None:
    """The property the cap exists for. Uncapped, log1p(5000)=8.5 multiplied a 40/40
    match to well past a 95/95 match with no upvotes."""
    viral_weak = actors._final_score(_scored(40, 40), _pair(engagement=5_000))
    quiet_strong = actors._final_score(_scored(95, 95), _pair(engagement=0))
    assert quiet_strong > viral_weak


def test_engagement_still_separates_comparable_leads() -> None:
    """Capping must not flatten the signal -- between two equal matches, the one people
    actually engaged with should still win."""
    busy = actors._final_score(_scored(80, 80), _pair(engagement=50))
    quiet = actors._final_score(_scored(80, 80), _pair(engagement=0))
    assert busy > quiet


def test_the_cap_actually_binds() -> None:
    """Beyond the cap, more upvotes buy nothing at all."""
    big = actors._final_score(_scored(80, 80), _pair(engagement=10_000))
    bigger = actors._final_score(_scored(80, 80), _pair(engagement=1_000_000))
    assert big == bigger


def test_cap_is_configurable_and_applied(monkeypatch) -> None:
    """The constant is a tuning knob, not a law. Assert it is read, not what it is."""
    from core import config

    monkeypatch.setattr(actors, "settings",
                        lambda: config.Settings(engagement_cap=0.0))
    with_engagement = actors._final_score(_scored(80, 80), _pair(engagement=10_000))
    without = actors._final_score(_scored(80, 80), _pair(engagement=0))
    assert with_engagement == without, "a zero cap must remove engagement entirely"


def test_missing_engagement_is_neutral_not_a_penalty() -> None:
    """SERP-only rows (quora, and reddit when the jar fails) carry no engagement. They
    must not be pushed down for it."""
    assert actors._final_score(_scored(80, 80), _pair(engagement=None)) == \
        actors._final_score(_scored(80, 80), _pair(engagement=0))


def test_gates_still_beat_every_factor() -> None:
    """No amount of engagement rescues a non-lead."""
    for bad in ({"is_seeking": False, "actionable": True},
                {"is_seeking": True, "actionable": False}):
        s = _scored(100, 100) | bad
        assert actors._final_score(s, _pair(engagement=10_000)) == 0.0


# --- linkedin budget ---------------------------------------------------------------

def _leaf(n_per_platform: int) -> dict:
    return {"queries": {p: [f"{p} q{i}" for i in range(n_per_platform)]
                        for p in actors.PLATFORMS}}


def test_linkedin_queries_are_capped_and_others_are_not() -> None:
    """LinkedIn: ~8-10% of leads for ~30-45% of discovery spend across two keywords.
    Capped, not dropped -- it still produces leads."""
    rows = actors._compile_queries("run1", "leaf1", _leaf(4))
    by_platform: dict[str, int] = {}
    for r in rows:
        by_platform[r["platform"]] = by_platform.get(r["platform"], 0) + 1

    assert by_platform["linkedin"] == settings().linkedin_queries_per_leaf
    assert by_platform["reddit"] == 4, "reddit is where the leads are; do not cap it"
    assert by_platform["quora"] == 4


def test_linkedin_is_not_removed_entirely() -> None:
    """Two keywords is not enough evidence to delete a platform."""
    rows = actors._compile_queries("run1", "leaf1", _leaf(4))
    assert any(r["platform"] == "linkedin" for r in rows)


def test_capping_survives_a_leaf_with_fewer_queries_than_the_cap() -> None:
    rows = actors._compile_queries("run1", "leaf1", {"queries": {"linkedin": []}})
    assert not [r for r in rows if r["platform"] == "linkedin"]


def test_depth_stays_at_ten() -> None:
    """11+ bills 2 Serper credits per query. The budget work is pointless if the cap
    lands and the depth silently doubles."""
    rows = actors._compile_queries("run1", "leaf1", _leaf(2))
    assert rows and all(r["depth"] == 10 for r in rows)
