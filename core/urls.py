"""URL canonicalization and platform routing.

Canonicalization decides the dedupe key, so getting it wrong shows up as inflated
candidate counts and duplicate scoring spend rather than as an error.

The LinkedIn rule here is load-bearing rather than cosmetic: only ``/posts/{slug}``
is anonymously readable. ``/feed/update/urn:li:activity:{id}`` 307s to the signup
wall, ``/company/{x}/posts/`` 302s to login, and ``/in/{x}/recent-activity/`` 301s
with the feed stripped. A SERP result that arrives in any of those shapes must be
rewritten to ``/posts/`` or it becomes a silent fetch miss -- HTTP 200, no body, and
nothing that looks like a failure.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse, urlunparse

PLATFORM_QUORA = "quora"
PLATFORM_LINKEDIN = "linkedin"
PLATFORM_REDDIT = "reddit"

# Reddit permalinks carry a human-readable slug that varies for the same thread
# (/comments/{id}/some_title/ vs /comments/{id}/other_title/). The id is canonical.
_REDDIT_COMMENTS = re.compile(r"/comments/([a-z0-9]+)", re.I)
_LINKEDIN_POST_SLUG = re.compile(r"/posts/([^/?#]+)")
_LINKEDIN_ACTIVITY_ID = re.compile(r"(?:urn:li:activity:|activity-)(\d+)")


class NotCanonicalizable(ValueError):
    """The URL belongs to a known platform but cannot be reduced to a fetchable form."""


def platform_of(url: str) -> str | None:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if host.endswith("quora.com"):
        return PLATFORM_QUORA
    if host.endswith("linkedin.com"):
        return PLATFORM_LINKEDIN
    if host.endswith("reddit.com"):
        return PLATFORM_REDDIT
    return None


def canonicalize(url: str) -> str:
    """Reduce a URL to its dedupe/fetch form.

    Drops every query param and fragment. No platform here needs one, and tracking
    params (utm_*, ?share_via=, ?rdt=) are the main source of same-page-different-URL
    duplicates in SERP results.
    """
    parsed = urlparse(url.strip())
    scheme = "https"
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"

    platform = platform_of(url)
    if platform == PLATFORM_LINKEDIN:
        path = _canonical_linkedin_path(path)
        # LinkedIn is the one host where dropping `www.` changes what you get served.
        # Measured back to back on the same public post, same client, same second:
        #     https://www.linkedin.com/posts/{slug}  -> 200, 335,526 bytes, articleBody
        #     https://linkedin.com/posts/{slug}      -> 200,  20,488 bytes, no ld+json
        # There is no redirect -- the bare host answers with a stub. Both are 200, so
        # nothing downstream looks like a failure: the stub parses to an empty body and
        # `looks_throttled` (no post ld+json node), so every LinkedIn candidate gets
        # retried three times and then trips the throttle circuit breaker. That is the
        # real "why no LinkedIn links", and it was misread as IP throttling.
        host = "www.linkedin.com"
    elif platform == PLATFORM_REDDIT:
        path = _canonical_reddit_path(path)

    return urlunparse((scheme, host, path, "", "", ""))


def _canonical_linkedin_path(path: str) -> str:
    slug = _LINKEDIN_POST_SLUG.search(path)
    if slug:
        return f"/posts/{slug.group(1)}"

    # /feed/update/urn:li:activity:12345 -> we hold an id but not the slug, and the
    # slug is not derivable from it. Refuse rather than fetch a URL that will 307 to
    # the signup wall and be scored as an empty post.
    activity = _LINKEDIN_ACTIVITY_ID.search(path)
    if activity:
        raise NotCanonicalizable(
            f"LinkedIn activity URL {path!r} has no /posts/ slug. "
            "Only /posts/{slug} is anonymously readable; /feed/update/ redirects to signup."
        )
    raise NotCanonicalizable(f"LinkedIn URL {path!r} is not a public post URL")


def _canonical_reddit_path(path: str) -> str:
    match = _REDDIT_COMMENTS.search(path)
    if match:
        return f"/comments/{match.group(1).lower()}"
    return path.lower()


def url_hash(url: str) -> str:
    """sha256 of the canonical URL. The dedupe key -- hash the canonical form, never the raw."""
    return hashlib.sha256(canonicalize(url).encode("utf-8")).hexdigest()
