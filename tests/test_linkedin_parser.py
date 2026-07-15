"""LinkedIn parser tests, against a real fetched post.

``linkedin_post.html`` is a live capture of a real public LinkedIn post, fetched
anonymously with no proxy and no browser. Its existence is the evidence for deleting
the Playwright lane from the spec: if this fixture parses, the browser was never needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.urls import NotCanonicalizable, canonicalize
from scrape.linkedin import LinkedInPost, is_authwalled, parse

FIXTURES = Path(__file__).parent / "fixtures"
URL = (
    "https://www.linkedin.com/posts/tarapowers_looking-for-a-tool-that-builds-relationships"
    "-activity-7463608443963318272-2kSA"
)


@pytest.fixture(scope="module")
def post() -> LinkedInPost:
    html = (FIXTURES / "linkedin_post.html").read_text(encoding="utf-8")
    return parse(html, URL)


def test_extracts_body_from_ldjson_articlebody(post: LinkedInPost) -> None:
    """The whole architecture rests on this: an anonymous GET yields the post body."""
    assert post.source == "articleBody"
    assert post.body is not None
    assert post.body.startswith("Looking for a tool that builds relationships")
    assert len(post.body) > 200


def test_full_body_preferred_over_truncated_og(post: LinkedInPost) -> None:
    """og:description is present on this page too. articleBody must win -- og is clipped."""
    assert post.is_truncated is False


def test_extracts_date_and_engagement(post: LinkedInPost) -> None:
    assert post.posted_at is not None
    assert post.posted_at.year == 2026
    assert post.engagement >= 0


def test_og_fallback_strips_comment_count_tail() -> None:
    """og:description ends '... | 274 comments on LinkedIn'. That tail must not reach
    the scorer as if it were part of the author's post."""
    html = (
        '<html><meta property="og:description" '
        'content="I spent six months looking for a tool that does X | 274 comments on LinkedIn" />'
        "</html>"
    )
    post = parse(html, URL)
    assert post.source == "og:description"
    assert post.body == "I spent six months looking for a tool that does X"
    assert post.is_truncated is True, "og-sourced bodies must be flagged as clipped"


def test_video_post_falls_back_to_videoobject_description() -> None:
    html = """<html><script type="application/ld+json">
    {"@type":"VideoObject","description":"How I finally cut my video background without a green screen",
     "uploadDate":"2026-03-01T10:00:00Z"}
    </script></html>"""
    post = parse(html, URL)
    assert post.source == "VideoObject.description"
    assert post.body is not None and "green screen" in post.body
    assert post.is_truncated is False


def test_empty_page_yields_no_body_rather_than_empty_string() -> None:
    """None means 'nothing extracted'. An empty string would score as a real, blank post."""
    post = parse("<html><body></body></html>", URL)
    assert post.body is None
    assert post.source is None


# --- Auth wall / 999 detection: these must never be scored as legitimate ---

def test_999_is_authwalled() -> None:
    """LinkedIn's own throttle status. A verdict on the caller -- trip the breaker,
    do not retry."""
    assert is_authwalled(999, URL, "")


@pytest.mark.parametrize(
    "final_url",
    [
        "https://www.linkedin.com/authwall?trk=...",
        "https://www.linkedin.com/signup/cold-join?session_redirect=...",
        "https://www.linkedin.com/uas/login?session_redirect=...",
    ],
)
def test_redirect_to_authwall_detected(final_url: str) -> None:
    """These arrive as HTTP 200 at a different URL, so status alone cannot catch them."""
    assert is_authwalled(200, final_url, "<html>...</html>")


def test_real_post_not_flagged_as_authwalled() -> None:
    html = (FIXTURES / "linkedin_post.html").read_text(encoding="utf-8")
    assert not is_authwalled(200, URL, html)


# --- Canonicalization: the silent-miss guard ---

def test_post_url_canonicalizes_to_slug() -> None:
    assert canonicalize(URL + "?utm_source=x&trk=y") == (
        "https://linkedin.com/posts/tarapowers_looking-for-a-tool-that-builds-relationships"
        "-activity-7463608443963318272-2kSA"
    )


def test_feed_update_url_refused_rather_than_fetched() -> None:
    """/feed/update/ 307s to signup. Fetching it returns a page with no post and no
    error -- the exact shape of a bug that never surfaces. Refuse at canonicalization."""
    with pytest.raises(NotCanonicalizable, match="no /posts/ slug"):
        canonicalize("https://www.linkedin.com/feed/update/urn:li:activity:7463608443963318272")


def test_company_posts_url_refused() -> None:
    with pytest.raises(NotCanonicalizable):
        canonicalize("https://www.linkedin.com/company/acme/posts/")
