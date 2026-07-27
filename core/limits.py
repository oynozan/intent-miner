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
import random
import time
from contextlib import ExitStack, contextmanager
from typing import Iterator

from dramatiq import Retry
from dramatiq.rate_limits import WindowRateLimiter
from dramatiq.rate_limits.backends import RedisBackend
from redis import Redis

from core.config import settings

log = logging.getLogger(__name__)

_redis = Redis.from_url(settings().redis_url)
BACKEND = RedisBackend(client=_redis)

# Per domain: a tuple of (requests, window_seconds) ceilings, ALL of which must be held
# at once. Multiple windows because a vendor's ceiling is not always one number -- Quora
# enforces a per-second rate AND a longer quota on top of it, and a limiter that models
# only the first passes a short probe while bleeding a third of a real run.
#
# The window shape is per-domain rather than one global constant because the *shape* of
# the ceiling differs by vendor, and getting the shape wrong makes the limiter a no-op
# even when the total looks conservative.
#
# Serper is the cautionary case. It was configured as 50 per 10s -- the same average as
# 5/s, and its 429 body says "up to 5 requests per second" verbatim. But
# WindowRateLimiter is a *window* limiter, not a smoother: it admits all 50 the instant
# the window opens. The discover queue runs 2 processes x 16 threads, so all 32 threads
# acquired a slot immediately and hit Serper's per-second wall together. Measured: ~85%
# of queries 429'd, fell through to SerpApi, and burned dramatiq retries.
#
# 4 rather than 5 because a Redis-second boundary can straddle Serper's own: 5 admitted
# late in second N plus 5 early in N+1 is 10 inside one real second from their side.
#
# Quora WAS deliberately absent, on the measured grounds that pacing did not help
# (~67% success bursting vs ~75% polite) because its rejections were short-lived per-IP
# *challenge* windows rather than a sustained-rate ceiling. That premise no longer
# holds: Quora now answers 429, not 403. Measured on one run of 199 candidates at 6
# concurrent fetch threads:
#
#     546 rejections: 541x 429, 5x 403, and ZERO successful fetches
#
# while sequential fetches of the same URLs in the same session returned 200 with the
# full 220-260KB payload. Bursting is the whole difference, which is what a rate ceiling
# means and what the old 403-era measurement could not have seen. 1/s because that is
# the rate the sequential probes actually sustained; the true ceiling is unmeasured, so
# this is the conservative end.
#
# 1/s alone was NOT enough, and the shortfall is invisible to a short probe:
#
#     burst     (24 requests over 24s, 6 threads):  24/24 = 100%, zero 429s
#     sustained (~300 requests over ~7min):         194 ok, 109x 429 -> 72% after retries
#
# Same limiter, same concurrency, same URLs. Quora holds a longer quota *on top of* the
# per-second rate, so a per-second value that passes a 24s test still bled ~28% of a
# full run. The sustained run is the measurement that matters: it landed 194 successes
# in ~7 minutes, i.e. Quora actually admitted ~28/min while we asked for 60/min.
#
# Hence the second window. 25/min rather than 28 for headroom, since 28 is the observed
# admit rate and not a documented ceiling.
LIMITS: dict[str, tuple[tuple[int, int], ...]] = {
    "linkedin.com": ((2, 10),),
    "serper.dev": ((4, 1),),
    # ponytail: 60 Redis keys per acquire for the minute window (WindowRateLimiter keys
    # per-second and sums the window). Fine at ~0.4 acquires/sec; if Quora volume grows
    # by an order of magnitude, move to a token bucket rather than widening this.
    #
    # Quora is currently unused by the pipeline -- fetch_candidate routes Quora to the
    # SERP-only path -- but scrape/quora.py still holds the fetcher, and enriching the
    # ~13% that survive the prefilter would need this pacing back. Kept deliberately.
    "quora.com": ((1, 1), (25, 60)),
    # Reddit's .json, once a cookie jar is held. 20/20 succeeded at ~0.83s each
    # sequentially (~72/min) with no sign of a ceiling, so the ceiling is unmeasured and
    # these are the conservative end rather than a known limit. Two windows for the same
    # reason as Quora: a per-second rate alone cannot express a longer quota, and Quora's
    # 24s probe passing at 100% while a 7-minute run bled 28% is what that mistake looks
    # like. Widen only against a measurement, never against a hunch.
    "reddit.com": ((2, 1), (50, 60)),
}


# How long to wait for a slot before giving up and requeueing the whole message.
# Bounded under the actors' time_limit (120s) with room for the request itself, so
# waiting can never be what kills a message. Sized for the widest window in LIMITS: a
# thread arriving at a saturated 60s window may have to wait most of a window for it to
# roll, so a 30s cap would requeue -- and requeueing costs a retry (see below).
ACQUIRE_TIMEOUT_SECONDS = 75


@contextmanager
def rate_limited(domain: str) -> Iterator[None]:
    """Wait for a slot in ``domain``'s window. Requeue only if the wait is hopeless.

    **Do not "simplify" this to a bare ``raise Retry`` on a missed slot.** That is what
    it used to do, on the stated grounds that "Retry requeues without consuming a retry
    budget". That is false, and verified false against the installed dramatiq 2.2.0 --
    ``Retries.after_process_message`` runs::

        message.options["retries"] += 1          # <-- unconditional
        ...
        if isinstance(exception, Retry) and exception.delay is not None:

    The increment happens *before* anything looks at the exception type, so a limiter
    rejection costs a retry exactly like a real failure does. With ``max_retries=3`` that
    is fatal: three missed slots and the message is dead having never made a request.

    Measured, when the Quora limiter was first wired up this way: 587 limiter requeues,
    195 of 199 candidates terminal at ``{'retries': 3, 'max_retries': 3}``, and only 8
    actual Quora responses. The pacing worked perfectly and the run still got nothing --
    the limiter had become the thing killing the messages.

    So: block until a slot frees, which is what pacing means anyway. A held thread is
    the correct cost -- on the fetch queue the work *is* the network call, so there is
    nothing else for that thread to do with the time.
    """
    specs = LIMITS.get(domain)
    if not specs:
        yield
        return

    # Shortest window first, deliberately. Acquiring several windows is not atomic, so
    # when a later one refuses we have already spent a slot in the earlier ones. Spending
    # a 1s slot is nearly free -- it refills next second -- while spending one of 25/min
    # is the scarce thing. Ordering the cheap window first makes the unavoidable waste
    # land on the cheap resource. (Over-spending is always the safe direction here: it
    # makes us slower than the ceiling, never faster.)
    ordered = sorted(specs, key=lambda s: s[1])
    shortest_window = ordered[0][1]
    deadline = time.monotonic() + ACQUIRE_TIMEOUT_SECONDS

    while True:
        with ExitStack() as stack:
            for limit, window in ordered:
                limiter = WindowRateLimiter(
                    BACKEND, f"rl:{domain}:{window}", limit=limit, window=window
                )
                if not stack.enter_context(limiter.acquire(raise_on_failure=False)):
                    break
            else:
                yield
                return
        if time.monotonic() >= deadline:
            # Genuinely saturated for the whole cap. Requeueing costs a retry, but at
            # this point the message is not getting through on this pass either.
            log.warning("rate limit for %s not clear after %ds; requeueing", domain, ACQUIRE_TIMEOUT_SECONDS)
            raise Retry(delay=shortest_window * 1000)
        # Jittered so N blocked threads do not all wake into the same window boundary.
        # Paced off the SHORTEST window: a wider one rolls continuously (it sums the
        # trailing N seconds), so sleeping a whole minute to recheck would waste most of
        # the capacity that frees up second by second.
        time.sleep(shortest_window * (0.25 + random.random() * 0.5))
