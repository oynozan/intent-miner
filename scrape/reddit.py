"""Reddit fetch + parse, via the public ``.json`` view of a post.

Reddit was written off as unfetchable on curl_cffi evidence alone -- the interstitial on
``www``, 403 on ``old`` and 403 on ``.json`` -- and ``_from_serp`` was built on that
conclusion. The conclusion was wrong, and it cost the platform that produces most of the
leads its body text, its date and its engagement.

**The gate is state, not fingerprint.** Measured, same Chrome TLS via curl_cffi
throughout:

    no cookies                       403 on every .json      (0/6)
    cookies minted by a real browser 200 on every .json      (20/20, 0.83s each)

So the browser is needed per *session*, not per URL. After a mint the cheap fetcher reads
a few KB of JSON per post instead of a 1MB DOM.

**A jar is spent by volume, and that number is small.** The 20/20 above led to "mint once
per TTL", which the first real run disproved: a jar that had worked earlier 429'd on 30
consecutive URLs, and a fresh one served the same URLs 6/6 immediately. Measured properly,
a session is good for roughly 30 fetches and then 429s on everything -- so a run of a few
hundred candidates re-mints ten-odd times, and the *replacement* path is the one that has
to work, not the cold-start path. ``jar(stale=...)`` and the retry in ``_fetch_reddit``
exist for that, and the earlier 2h TTL is now only an idle bound.

**Why a real browser and nothing lighter.** Both cheaper options were tried and both
yield a single ``edgebucket`` cookie, which does not unlock anything:

    curl_cffi GET of the homepage      1 cookie
    obscura --stealth --dump cookies   1 cookie
    headless chromium                  12 cookies -> works

The jar Reddit actually wants includes ``loid``, ``csv`` and ``pxrc`` (that last one
looks like PerimeterX), and they are written by JavaScript that neither a bare HTTP
client nor obscura's V8 executes fully. Hence Playwright -- used only here, roughly once
per TTL, never in the per-URL path.

**The payload is the point.** One ``.json`` gives ``selftext`` (the real post body, not a
truncated SERP snippet), ``created_utc`` (real recency, replacing the 0.5 unknown-date
default), ``ups`` and ``num_comments``. The last two are deliberately kept apart: upvotes
are reach and feed ``engagement``, comments are saturation and feed ``answers_total``,
which the scorer reads as ``existing_answer_count`` when judging ``actionable``. Summing
them would make one post simultaneously more attractive and more crowded.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Redis key for the shared jar. Global rather than per-run: the jar is not run-specific
# and a run should not pay the mint cost just for existing.
_JAR_KEY = "reddit:cookiejar"
# Only an idle-staleness bound. What actually retires a jar is volume: measured, a fresh
# session serves ~30 .json fetches and then 429s on everything, so a full run re-mints
# roughly once per 30 candidates regardless of this TTL.
_JAR_TTL_SECONDS = 2 * 3600

# Minting is not always possible, and the failure is per-host rather than per-request:
# Reddit answers a datacenter IP with 403 and a ~190KB interstitial, so the browser runs,
# returns a lone `edgebucket`, and no amount of retrying changes it. Without a memory of
# that, every Reddit candidate launches Chromium twice (once cold, once for the stale-jar
# replacement) and every launch is doomed -- hundreds of them per run, each costing an
# interpreter start and a browser boot, to arrive at the SERP fallback anyway.
#
# So a failed mint is remembered for this long and the browser is skipped entirely until
# it expires. Short enough that a host which regains access recovers on its own.
_MINT_FAIL_KEY = "reddit:cookiejar:unavailable"
_MINT_FAIL_TTL_SECONDS = 15 * 60

# A real jar carries ~12 cookies including loid / csv / pxrc. A blocked or JS-less page
# yields exactly one (`edgebucket`), which is indistinguishable from success to a caller
# that only checks for an exception -- and it is worse than no jar, because it caches for
# the full TTL and gates every fetch made with it. Counting rather than name-matching so
# a renamed cookie does not silently disable the whole path.
_MIN_USEFUL_COOKIES = 3

_MINT_LOCK = threading.Lock()

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")

# /comments/<id>/ and /r/<sub>/comments/<id>/<slug>/ both appear in SERP results.
_POST_ID = re.compile(r"/comments/([a-z0-9]+)", re.I)


@dataclass
class RedditPost:
    url: str
    title: str | None = None
    selftext: str | None = None
    ups: int | None = None
    num_comments: int | None = None
    created_utc: float | None = None

    @property
    def body(self) -> str:
        """Title plus post body.

        Mirrors ``_from_serp``'s title+snippet join so the two paths produce the same
        shape. Link posts carry an empty ``selftext`` and degrade to the title alone,
        which is still no worse than the SERP row it replaces.
        """
        return "\n\n".join(p for p in (self.title, self.selftext) if p and p.strip())

    @property
    def posted_at(self) -> datetime | None:
        if not self.created_utc:
            return None
        return datetime.fromtimestamp(self.created_utc, tz=timezone.utc)


def post_id(url: str) -> str | None:
    m = _POST_ID.search(url)
    return m.group(1) if m else None


def json_url(url: str) -> str | None:
    pid = post_id(url)
    return f"https://www.reddit.com/comments/{pid}/.json" if pid else None


def mint_jar(timeout: float = 120.0) -> dict[str, str]:
    """Drive a real browser once to obtain a jar Reddit will honour.

    Runs in a child process, which is not incidental. Playwright's sync API drives its
    driver over an asyncio subprocess transport, and on Windows that only works on the
    main thread; called from a dramatiq worker thread it fails with
    ``[Errno 9] Bad file descriptor`` -- observed 58 times in one run, so every re-mint
    silently degraded to the SERP fallback. A child process always has a main thread.

    Costs one interpreter start on a path that runs about once per TTL.
    """
    proc = subprocess.run([sys.executable, "-m", "scrape.reddit"],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"reddit jar mint failed: {proc.stderr.strip()[-400:]}")
    return json.loads(proc.stdout)


def _proxy() -> dict[str, str] | None:
    """Playwright proxy config from ``REDDIT_MINT_PROXY``, or None.

    Reddit refuses datacenter IPs at the edge. Measured with the *same* Playwright
    build, same Chrome UA, same flags, minutes apart:

        residential IP   200, 12 cookies (loid / csv / pxrc present)
        datacenter VPS   403, 1 cookie  (edgebucket only), ~190KB interstitial

    So the client was never the problem and no stealth tooling addresses it -- obscura
    was measured at 1 cookie even from a residential IP, worse than plain Chromium.
    Only the address matters, and only for this one call: a mint happens about once per
    30 fetches, never per URL, so the cheapest residential proxy covers a whole run and
    the per-URL fetches keep going out direct.

    Accepts one URL, credentials inline: ``http://user:pass@host:port``.
    """
    raw = os.environ.get("REDDIT_MINT_PROXY", "").strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    if not parsed.hostname:
        log.warning("REDDIT_MINT_PROXY is set but unparseable; minting direct")
        return None

    config = {"server": f"{parsed.scheme or 'http'}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        config["username"] = parsed.username
        config["password"] = parsed.password or ""
    return config


def _mint_here() -> dict[str, str]:
    """The actual browser work. Only ever called in the child, on its main thread."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True, args=["--disable-blink-features=AutomationControlled"],
            proxy=_proxy())
        try:
            ctx = browser.new_context(user_agent=UA, viewport={"width": 1440, "height": 900},
                                      locale="en-US")
            # navigator.webdriver is the cheapest headless tell; drop it before any page
            # script runs, since the cookies we are here for are set by page script.
            ctx.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
            page = ctx.new_page()
            page.goto("https://www.reddit.com/", wait_until="domcontentloaded",
                      timeout=60_000)
            page.wait_for_timeout(6_000)   # the jar is written by JS after load
            return {c["name"]: c["value"] for c in ctx.cookies()}
        finally:
            browser.close()


def jar(stale: dict[str, str] | None = None) -> dict[str, str]:
    """The shared jar, minted on demand and cached in Redis.

    Pass ``stale`` -- the jar that just came back gated -- to demand a replacement. That
    is deliberately a jar rather than a boolean flag: a jar dies of volume, so when it
    dies every in-flight fetch gates at once and asks for a new one within a second or
    two of the others. Comparing against the caller's jar answers all of them from the
    first replacement, while a plain "force" would mint once per gated fetch and a
    time-based grace window would do the opposite -- suppress the *next* real re-mint,
    which arrives ~30 fetches later and can easily be inside the window.

    The lock serialises minting within the process; the re-check inside it is what
    collapses a cold-start stampede to one browser launch. Across processes a double-mint
    is harmless -- the later write wins -- and a distributed lock would add a failure mode
    (holder dies, everyone waits) to save a few seconds on a rare path.
    """
    from core.limits import _redis      # the broker's Redis, already configured

    def cached() -> dict[str, str] | None:
        raw = _redis.get(_JAR_KEY)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            log.warning("reddit jar in redis was unreadable; re-minting")
            return None

    hit = cached()
    if hit is not None and hit != stale:
        return hit

    with _MINT_LOCK:
        hit = cached()                    # someone may have minted while we waited
        if hit is not None and hit != stale:
            return hit
        if _redis.get(_MINT_FAIL_KEY):
            raise RuntimeError("reddit jar mint failed recently on this host; not retrying yet")
        try:
            fresh = mint_jar()
        except Exception:
            _redis.setex(_MINT_FAIL_KEY, _MINT_FAIL_TTL_SECONDS, "1")
            raise
        if len(fresh) < _MIN_USEFUL_COOKIES:
            # Succeeded mechanically, produced nothing usable -- the blocked-host case.
            # Caching this would gate every fetch for the whole TTL while looking healthy.
            _redis.setex(_MINT_FAIL_KEY, _MINT_FAIL_TTL_SECONDS, "1")
            raise RuntimeError(
                f"reddit jar mint returned {len(fresh)} cookie(s) ({', '.join(sorted(fresh)) or 'none'}); "
                "this host is most likely blocked by Reddit -- falling back to the SERP row"
            )
        _redis.setex(_JAR_KEY, _JAR_TTL_SECONDS, json.dumps(fresh))
        log.info("minted reddit cookie jar (%d cookies)", len(fresh))
        return fresh


def fetch(url: str, cookies: dict[str, str], timeout: float = 30.0) -> tuple[int, str]:
    """Fetch a post's ``.json``. Same Chrome TLS impersonation as ``scrape/quora.py``."""
    from curl_cffi import requests as cffi

    target = json_url(url) or url
    response = cffi.get(target, impersonate="chrome", timeout=timeout,
                        headers={"User-Agent": UA}, cookies=cookies)
    return response.status_code, response.text


def is_gated(status: int, text: str) -> bool:
    """True when the response is not a post, so it never reaches the parser.

    Shape, not status codes. The Quora incident is the reason: ``is_challenge``
    enumerated 403 and let 429 through, so 29 error pages were stored with
    ``fetch_status='ok'`` and ``body='Error 429 (Too Many Requests)'``, and only the
    prefilter's semantic kill kept them out of the results -- luck, not a guard.

    Reddit's refusal is a ~190KB HTML page, which is exactly the kind of thing that
    parses into a plausible-looking row if the check trusts the status code alone.
    """
    if status != 200:
        return True
    return not text.lstrip()[:1] in ("[", "{")


def parse(text: str, url: str) -> RedditPost:
    """Pull the post out of the listing envelope.

    ``.json`` returns ``[post_listing, comment_listing]``; the post is the first child of
    the first listing. Anything unexpected yields an empty RedditPost rather than raising
    -- ``is_gated`` is the guard, and a shape change should degrade, not crash a worker.
    """
    page = RedditPost(url=url)
    try:
        payload: Any = json.loads(text)
        data = payload[0]["data"]["children"][0]["data"]
    except (ValueError, KeyError, IndexError, TypeError):
        log.warning("reddit payload did not match the expected listing shape: %s", url)
        return page

    page.title = data.get("title")
    page.selftext = data.get("selftext") or None
    page.ups = data.get("ups")
    page.num_comments = data.get("num_comments")
    page.created_utc = data.get("created_utc")
    return page


if __name__ == "__main__":
    # The child half of mint_jar: mint on this process's main thread, print the jar.
    # Also the manual check -- `python -m scrape.reddit` should print a dozen cookies
    # including loid/csv/pxrc. One cookie means the browser layer is not running.
    print(json.dumps(_mint_here()))
