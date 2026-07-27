"""LinkedIn public post fetch + parse.

The original spec routed LinkedIn through Playwright + a residential proxy. It does
not need either. Verified against a live public post:

    curl_cffi (Chrome TLS):  200, 101,827 bytes, articleBody present
    httpx     (OpenSSL TLS): 200, 105,014 bytes, articleBody present

Unlike Quora -- where httpx is challenged 0/6 and curl_cffi succeeds 3/3 -- LinkedIn
does not fingerprint the TLS handshake on this route, so the cheap client is fine.
The two platforms genuinely differ; don't unify them onto one client "for consistency".

This works for a structural reason rather than an oversight: LinkedIn publishes
``/posts/{slug}`` as an SEO landing surface for Googlebot. The adjacent routes prove
the split -- ``/feed/update/urn:li:activity:{id}`` 307s to signup, ``/company/{x}/posts/``
302s to login, ``/in/{x}/recent-activity/`` 301s with the feed stripped. Only
``/posts/{slug}`` is open, which is why core/urls.py refuses to fetch anything else.

A useful self-correcting property: a post is only anonymously readable if its author
enabled public visibility, and non-public posts never reach the ``/posts/`` SEO
surface at all. So anything SERP-discoverable is public by construction. Non-public
posts cost recall silently at *discovery*; they are not failed fetches, and there is
no retry that recovers them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Mapping

log = logging.getLogger(__name__)

_LDJSON = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I)
_OG_DESCRIPTION = re.compile(
    r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']', re.S | re.I
)
# og:description is a truncated meta description; LinkedIn appends this tail.
_OG_TAIL = re.compile(r"\s*\|\s*\d+\s+comments?\s+on\s+LinkedIn\s*$", re.I)

_AUTHWALL_MARKERS = ("/authwall", "/signup/cold-join", "/uas/login", "/checkpoint/challenge")

# The ld+json @type carrying articleBody. LinkedIn now emits `DiscussionForumPosting`
# on public post pages; it used to be `SocialMediaPosting`. Both are accepted because
# the rename is theirs to reverse, and matching only one is a silent failure: the node
# is missed, `had_posting_ldjson` stays False, parse falls through to the truncated
# og:description, and posted_at/engagement come back None/0 on a page that had both.
_POSTING_TYPES = ("SocialMediaPosting", "DiscussionForumPosting")


@dataclass
class LinkedInPost:
    url: str
    body: str | None = None
    author: str | None = None
    posted_at: datetime | None = None
    engagement: int = 0
    # Which tier of the extraction chain produced `body`. Worth persisting: og:description
    # is truncated, so a run where og-sourced bodies spike is a run whose scores are
    # quietly built on clipped text.
    source: str | None = None
    # True if the page carried a SocialMediaPosting/VideoObject ld+json node -- i.e. it
    # really is a post page. A public post page always has one. A throttled/gated 200
    # (LinkedIn's response to a suspect IP under volume) does NOT. So `not
    # had_posting_ldjson` with an empty body is the signature of a throttle, which the
    # fetch actor should retry -- distinct from a genuinely text-less post (posting node
    # present, empty articleBody), which is terminal.
    had_posting_ldjson: bool = False

    @property
    def is_truncated(self) -> bool:
        """og:description is a clipped meta description, never the full post."""
        return self.source == "og:description"

    @property
    def looks_throttled(self) -> bool:
        """Empty body AND no post ld+json node -> a gated/throttled page, not the post."""
        return not self.body and not self.had_posting_ldjson

    @property
    def embed_text(self) -> str:
        return self.body or ""


class AuthWalled(RuntimeError):
    """LinkedIn served the auth wall instead of the post.

    Distinct from an empty body: an auth-walled fetch tells you nothing about the post,
    while an empty body means the post genuinely had no text. Scoring either as
    legitimate is how a run silently fills with zeros.
    """


def fetch(url: str, timeout: float = 25.0, proxy: str | None = None) -> tuple[int, str, Mapping[str, str], str]:
    """Fetch a public post. Plain httpx -- LinkedIn does not need TLS impersonation here.

    Returns (status, html, headers, final_url). The final URL matters: an auth-wall
    redirect is a 200 at a different URL, not an error status.
    """
    import httpx

    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        proxy=proxy,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    ) as client:
        response = client.get(url)
        return response.status_code, response.text, response.headers, str(response.url)


def is_authwalled(status_code: int, final_url: str, html: str) -> bool:
    """Detect the auth wall, including the 999 LinkedIn returns to suspect clients.

    999 is LinkedIn's own throttle status and concentrates on cloud ASNs. It must trip
    a circuit breaker rather than be retried -- it is a verdict on the caller, not the
    request.
    """
    if status_code == 999:
        return True
    if any(marker in final_url for marker in _AUTHWALL_MARKERS):
        return True
    if status_code == 200 and len(html) < 10_000 and "ld+json" not in html:
        return True
    return False


def parse(html: str, url: str) -> LinkedInPost:
    """Extract the post via the three-tier chain, best source first.

    1. ld+json _POSTING_TYPES.articleBody -- full text, the only complete source, and
       the ONLY tier that also yields datePublished + commentCount
    2. ld+json VideoObject.description    -- video posts carry text here instead
    3. og:description                     -- TRUNCATED. Last resort, flagged.

    Tier 1 is not just about body length: tiers 2 and 3 return posted_at=None and
    engagement=0, so a run that silently falls to tier 3 has no recency decay and no
    engagement signal for any LinkedIn row.

    Tier 3 is never silently acceptable: it ends mid-sentence with "... | 274 comments
    on LinkedIn", so it feeds clipped text to the scorer while looking like a success.
    """
    post = LinkedInPost(url=url)

    for node in _iter_ldjson_nodes(html):
        typename = node.get("@type")
        if typename in _POSTING_TYPES:
            post.had_posting_ldjson = True
            body = _clean(node.get("articleBody"))
            if body:
                post.body = body
                post.source = "articleBody"
            post.posted_at = _as_datetime(node.get("datePublished"))
            post.engagement = _as_int(node.get("commentCount")) + _as_int(node.get("interactionCount"))
            post.author = _author_of(node)
            if post.body:
                return post
        elif typename == "VideoObject" and not post.body:
            post.had_posting_ldjson = True
            body = _clean(node.get("description"))
            if body:
                post.body = body
                post.source = "VideoObject.description"
                post.posted_at = post.posted_at or _as_datetime(node.get("uploadDate"))

    if post.body:
        return post

    og = _OG_DESCRIPTION.search(html)
    if og:
        body = _clean(_OG_TAIL.sub("", og.group(1)))
        if body:
            post.body = body
            post.source = "og:description"
            log.info("linkedin %s: fell back to truncated og:description", url)
    return post


def _iter_ldjson_nodes(html: str) -> Iterator[dict[str, Any]]:
    for block in _LDJSON.findall(html):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        for node in data if isinstance(data, list) else [data]:
            if isinstance(node, dict):
                yield node
                for value in node.get("@graph", []) or []:
                    if isinstance(value, dict):
                        yield value


def _author_of(node: dict[str, Any]) -> str | None:
    author = node.get("author")
    if isinstance(author, dict):
        return author.get("name")
    if isinstance(author, str):
        return author
    return None


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
