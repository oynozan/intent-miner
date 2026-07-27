"""Rate limiter: it must actually pace, and it must pace on the vendor's own shape.

The bug this locks down was not a missing limiter -- `rate_limited` existed and was
correct. It was (a) never called, and (b) configured as 50-per-10s for a vendor whose
ceiling is 5-per-second. A window limiter admits its whole allowance the instant the
window opens, so 50/10s let 32 concurrent threads through at once and every one of them
429'd. The average looked conservative; the shape was wrong.
"""

from __future__ import annotations

import os

import pytest
from dramatiq import Retry
from redis import Redis
from redis.exceptions import RedisError

from core import limits

# Host as well as port, because the suite has to run where the app runs. From the host
# compose publishes redis on 127.0.0.1:6380; from inside a container it is redis:6379,
# and a hardcoded localhost cannot be pointed at it by any amount of configuration.
REDIS_HOST = os.environ.get("TEST_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("TEST_REDIS_PORT", "6380"))


@pytest.fixture(autouse=True)
def _redis_up():
    try:
        Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=3).ping()
    except RedisError as exc:
        pytest.fail(f"redis unavailable on {REDIS_HOST}:{REDIS_PORT} ({exc})")


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch):
    """Point the limiter at a unique key so tests never share a window.

    A 5s window, not 1s: WindowRateLimiter buckets on ``int(time.time())``, so with
    window=1 a test that happens to straddle a second boundary lands in a fresh bucket
    and the assertion flaps. A 5s window sums the trailing seconds, so it holds
    regardless of where the test starts.
    """
    import uuid

    domain = f"test-{uuid.uuid4().hex[:8]}.example"
    monkeypatch.setitem(limits.LIMITS, domain, ((2, 5),))
    return domain


def test_serper_is_paced_per_second_not_per_window() -> None:
    """Serper's 429 says 'up to 5 requests per second'. The limit must be expressed in
    a 1s window -- a 10s window with a 10x allowance is not the same constraint."""
    ((limit, window),) = limits.LIMITS["serper.dev"]
    assert window == 1, "a multi-second window lets the whole allowance burst at once"
    assert limit <= 5, f"Serper allows 5/s; {limit} would 429"


@pytest.fixture
def no_wait(monkeypatch: pytest.MonkeyPatch):
    """Collapse the acquire wait so the give-up path is reachable in a fast test."""
    monkeypatch.setattr(limits, "ACQUIRE_TIMEOUT_SECONDS", 0)


def test_admits_up_to_the_limit_then_requeues(fresh: str, no_wait) -> None:
    admitted = 0
    for _ in range(2):
        with limits.rate_limited(fresh):
            admitted += 1
    assert admitted == 2

    # The third acquire in the same window must not run the body.
    with pytest.raises(Retry):
        with limits.rate_limited(fresh):
            pytest.fail("limiter admitted more than its allowance")


def test_requeue_delay_matches_the_window(fresh: str, no_wait) -> None:
    """A flat delay far larger than the window makes latency a function of the constant
    rather than the ceiling -- 15s of idling for a 1s window."""
    for _ in range(2):
        with limits.rate_limited(fresh):
            pass
    with pytest.raises(Retry) as exc:
        with limits.rate_limited(fresh):
            pass
    assert exc.value.delay == 5000, "delay must track the domain's window, not a constant"


def test_waits_for_a_slot_instead_of_spending_a_retry(monkeypatch) -> None:
    """The load-bearing behaviour. dramatiq increments message.options['retries'] BEFORE
    it inspects the exception type, so a bare `raise Retry` on a missed slot costs a
    retry exactly like a real failure. With max_retries=3, three missed slots killed a
    message that had never made a request -- measured at 195/199 Quora candidates.

    So a contended limiter must BLOCK and then proceed, not raise.
    """
    import uuid

    domain = f"test-{uuid.uuid4().hex[:8]}.example"
    monkeypatch.setitem(limits.LIMITS, domain, ((1, 1),))

    with limits.rate_limited(domain):        # consumes the only slot in this window
        pass

    slept: list[float] = []
    monkeypatch.setattr(limits.time, "sleep", lambda s: slept.append(s))

    # Second acquire is contended. It must wait (sleep) and then run the body -- the
    # real limiter's window rolls while it sleeps.
    ran = False
    try:
        with limits.rate_limited(domain):
            ran = True
    except Retry:
        pytest.fail("limiter raised Retry instead of waiting -- this burns a retry")
    assert ran
    assert slept, "a contended limiter must wait for a slot, not fail fast"


def test_retry_consumes_budget_in_installed_dramatiq() -> None:
    """Locks the upstream behaviour the fix depends on. If a future dramatiq stops
    counting Retry against max_retries, the wait-loop can be simplified again."""
    import inspect

    from dramatiq.middleware.retries import Retries

    src = inspect.getsource(Retries.after_process_message)
    increment = src.index('message.options["retries"] += 1')
    type_check = src.index("isinstance(exception, Retry)")
    assert increment < type_check, (
        "dramatiq now checks the exception type before incrementing retries; "
        "rate_limited's wait-loop may no longer be necessary"
    )


def test_serper_requeue_delay_is_one_second() -> None:
    """The real config, not a fixture: Serper's window is 1s, so a near-miss costs 1s."""
    ((_limit, window),) = limits.LIMITS["serper.dev"]
    assert window * 1000 == 1000


def test_unlimited_domain_is_a_passthrough() -> None:
    """A domain with no entry must not be paced at all."""
    assert "example.invalid" not in limits.LIMITS
    ran = False
    with limits.rate_limited("example.invalid"):
        ran = True
    assert ran


def test_quora_is_paced_now_that_it_returns_429() -> None:
    """Quora used to be deliberately unpaced, on the grounds that its 403 challenge
    windows ignored politeness. It now returns 429 -- a rate ceiling -- and at 6
    concurrent threads a full run scored 541 rejections and zero fetches."""
    assert (1, 1) in limits.LIMITS["quora.com"], "per-second pacing is still required"


def test_quora_also_has_a_sustained_window() -> None:
    """The per-second rate alone passed a 24s probe at 100% and still lost 28% of a
    7-minute run: Quora admitted ~28/min while we asked for 60/min. One window cannot
    express 'fast enough per second AND slow enough per minute'."""
    windows = {w for _, w in limits.LIMITS["quora.com"]}
    assert len(windows) > 1, "a single window cannot model a per-second + per-quota ceiling"
    per_min = [(lim, w) for lim, w in limits.LIMITS["quora.com"] if w >= 60]
    assert per_min, "expected a >=60s window"
    limit, window = per_min[0]
    rate_per_min = limit * 60 / window
    assert rate_per_min <= 28, f"{rate_per_min}/min exceeds the ~28/min Quora actually admitted"


def test_all_windows_must_be_held_together(monkeypatch) -> None:
    """Two ceilings means both apply. The tight one must still bind after the loose one
    has been satisfied -- otherwise the second window is decorative."""
    import uuid

    domain = f"test-{uuid.uuid4().hex[:8]}.example"
    # Loose per-second (5/s) but tight per-5s (2 per 5s): the second must do the work.
    monkeypatch.setitem(limits.LIMITS, domain, ((5, 1), (2, 5)))
    monkeypatch.setattr(limits, "ACQUIRE_TIMEOUT_SECONDS", 0)

    admitted = 0
    for _ in range(2):
        with limits.rate_limited(domain):
            admitted += 1
    assert admitted == 2

    with pytest.raises(Retry):
        with limits.rate_limited(domain):
            pytest.fail("the wider window was not enforced")


def test_windows_get_separate_redis_keys(monkeypatch) -> None:
    """Both windows for one domain must not collide on a single key, or they would
    increment each other's counters and neither ceiling would mean anything."""
    import uuid

    domain = f"test-{uuid.uuid4().hex[:8]}.example"
    monkeypatch.setitem(limits.LIMITS, domain, ((1, 1), (5, 60)))

    keys: list[str] = []
    real = limits.WindowRateLimiter

    def spy(backend, key, **kw):
        keys.append(key)
        return real(backend, key, **kw)

    monkeypatch.setattr(limits, "WindowRateLimiter", spy)
    with limits.rate_limited(domain):
        pass
    assert len(keys) == 2 and len(set(keys)) == 2, f"windows shared a key: {keys}"


def test_shortest_window_is_acquired_first(monkeypatch) -> None:
    """Acquiring several windows is not atomic, so a later refusal strands slots already
    taken. Taking the cheap (fast-refilling) window first makes that waste land on the
    resource that costs least -- never on the scarce long-window quota."""
    import uuid

    domain = f"test-{uuid.uuid4().hex[:8]}.example"
    monkeypatch.setitem(limits.LIMITS, domain, ((25, 60), (1, 1)))  # deliberately unsorted

    order: list[int] = []
    real = limits.WindowRateLimiter

    def spy(backend, key, *, limit, window):
        order.append(window)
        return real(backend, key, limit=limit, window=window)

    monkeypatch.setattr(limits, "WindowRateLimiter", spy)
    with limits.rate_limited(domain):
        pass
    assert order == sorted(order), f"windows acquired widest-first: {order}"


def test_quora_is_served_from_the_serp_not_fetched() -> None:
    """Quora's gate turned out to be a quota no amount of pacing or IP rotation clears:
    0/25 datacenter proxies and 2/15 residential, one request per fresh IP. Since the
    pipeline only ever kept the *question* -- which the SERP title already carries -- the
    fetch was paying a hostile gate for data we had. Quora now shares reddit's path.

    The limiter config for quora.com deliberately stays: scrape/quora.py still fetches,
    and re-enriching prefilter survivors would need it back.
    """
    import inspect

    from pipeline import actors

    assert not hasattr(actors, "_fetch_quora"), "the fetch path is supposed to be gone"
    src = inspect.getsource(actors.fetch_candidate.fn)
    assert '"quora"' in src and "_from_serp" in src, "quora must route to the SERP-only path"
    # Reddit was split back out to its own enricher; quora must not have ridden along.
    assert "_fetch_reddit" in src
    quora_branch = src[src.index('"quora"'):]
    assert quora_branch.index("_from_serp") < (
        quora_branch.index("_fetch_reddit") if "_fetch_reddit" in quora_branch else len(quora_branch)
    ), "the quora branch must reach _from_serp, not the reddit enricher"


def test_discovery_actually_holds_the_limiter() -> None:
    """The original bug: rate_limited was defined and never called from anywhere, so
    the whole module was dead code. Assert the discovery path references it."""
    import inspect

    from pipeline import actors

    # .fn unwraps the dramatiq Actor back to the underlying function.
    src = inspect.getsource(actors.run_query.fn)
    assert 'rate_limited("serper.dev")' in src, "run_query must hold the SERP limiter"
