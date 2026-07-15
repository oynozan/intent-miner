"""Per-domain rate limiting.

The spec's version was a no-op::

    def acquire(domain):
        limiter = WindowRateLimiter(...)
        with limiter.acquire(raise_on_failure=False) as ok:
            if not ok:
                raise Retry(delay=15_000)
            yield

A ``yield`` with no ``@contextmanager`` makes that a *generator function*. Calling it
builds a generator object and executes none of the body, so the limiter is never
touched and the ``raise Retry`` is unreachable. ``with acquire(...)`` raises
TypeError outright. Both failure modes are silent in the sense that neither one
rate-limits anything.

Two other fixes from the spec version: the limiter is built once at module scope
(it is stateless and keyed in Redis, so rebuilding per call is pure overhead), and
the window is 10s rather than 60s. A 60s window keeps 60 Redis keys per domain and
every acquire sums across all of them.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from dramatiq import Retry
from dramatiq.rate_limits import WindowRateLimiter
from dramatiq.rate_limits.backends import RedisBackend
from redis import Redis

from core.config import settings

log = logging.getLogger(__name__)

_redis = Redis.from_url(settings().redis_url)
BACKEND = RedisBackend(client=_redis)

WINDOW_SECONDS = 10

# Requests per WINDOW_SECONDS, i.e. these are 1/6th of a per-minute budget.
#
# Quora is deliberately absent. Measurement showed pacing does not help there
# (~67% success bursting vs ~75% polite) because its 403s are short-lived per-IP
# penalty windows rather than a sustained-rate ceiling. Rate limiting Quora buys
# latency and no admission. Rotate the IP on retry instead -- see scrape/quora.py.
LIMITS: dict[str, int] = {
    "linkedin.com": 2,
    "serper.dev": 50,
}


@contextmanager
def rate_limited(domain: str) -> Iterator[None]:
    """Hold a slot in ``domain``'s window, or raise Retry to requeue the message.

    Retry is a dramatiq control-flow exception, not a failure: the middleware
    requeues the message after ``delay`` ms without consuming a retry budget.
    """
    limit = LIMITS.get(domain)
    if limit is None:
        yield
        return

    limiter = WindowRateLimiter(BACKEND, f"rl:{domain}", limit=limit, window=WINDOW_SECONDS)
    with limiter.acquire(raise_on_failure=False) as acquired:
        if not acquired:
            log.debug("rate limit hit for %s, requeueing", domain)
            raise Retry(delay=15_000)
        yield
