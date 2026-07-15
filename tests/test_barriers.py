"""Barrier tests.

These run against a real Redis, not fakeredis, because the properties under test are
Lua atomicity and SADD/SCARD semantics -- exactly what an emulation layer is most
likely to get subtly right when the real thing is wrong, or vice versa.

The tests named ``test_retry_does_not_*`` and ``test_redelivery_does_not_*`` are
regression locks on the two bugs that made dramatiq's own Barrier unusable here.
If someone later "simplifies" barriers.py back to a DECR, these fail. Do not delete
them.
"""

from __future__ import annotations

import os
import uuid

import pytest
from redis import Redis
from redis.exceptions import RedisError

from core.barriers import Barrier, BarrierNotCreated


REDIS_HOST = os.environ.get("TEST_REDIS_HOST", "localhost")
# 6380, not 6379: compose offsets the host port so this stack can run alongside the
# sibling bg-remover stack, which binds 6379.
REDIS_PORT = int(os.environ.get("TEST_REDIS_PORT", "6380"))


@pytest.fixture(scope="session")
def redis_client() -> Redis:
    client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=15, decode_responses=True, socket_connect_timeout=3)
    try:
        client.ping()
    except RedisError as exc:
        pytest.fail(f"redis unavailable on {REDIS_HOST}:{REDIS_PORT} ({exc}) -- run `docker compose up -d redis`")
    return client


@pytest.fixture
def barrier(redis_client: Redis) -> Barrier:
    b = Barrier(redis_client, run_id=f"test-{uuid.uuid4()}", stage="discover", ttl=60)
    yield b
    redis_client.delete(b.arrived_key, b.total_key)


def test_fires_exactly_once_on_last_arrival(barrier: Barrier) -> None:
    barrier.create(["a", "b", "c"])
    assert barrier.arrive("a") is False
    assert barrier.arrive("b") is False
    assert barrier.arrive("c") is True


def test_arrival_order_does_not_matter(barrier: Barrier) -> None:
    barrier.create(["a", "b", "c"])
    assert barrier.arrive("c") is False
    assert barrier.arrive("a") is False
    assert barrier.arrive("b") is True


def test_retry_does_not_fire_barrier_early(barrier: Barrier) -> None:
    """The bug that survives the obvious fix for dead retries.

    dramatiq's Barrier is a blind DECR. Re-raising the exception (to make
    max_retries live) while keeping `finally: close_barrier()` decrements once per
    ATTEMPT. Party "a" failing and retrying three times would fire a 3-party barrier
    while "b" and "c" had not run at all.

    Party identity makes repeat arrivals free.
    """
    barrier.create(["a", "b", "c"])

    # Party "a" retries four times (max_retries=3 means 4 total attempts).
    for _ in range(4):
        assert barrier.arrive("a") is False, "a's retries must never complete the barrier"

    arrived, total = barrier.progress()
    assert (arrived, total) == (1, 3), "four attempts by one party must count as one arrival"

    assert barrier.arrive("b") is False
    assert barrier.arrive("c") is True


def test_redelivery_does_not_fire_barrier_early(barrier: Barrier) -> None:
    """Dramatiq is at-least-once: a worker dying after work but before ack redelivers.

    A counting barrier decrements twice for one party. An identity barrier does not.
    """
    barrier.create(["a", "b"])
    assert barrier.arrive("a") is False
    assert barrier.arrive("a") is False, "redelivery of the same party must not advance the count"
    assert barrier.progress() == (1, 2)
    assert barrier.arrive("b") is True


def test_extra_arrival_after_completion_does_not_refire(barrier: Barrier) -> None:
    """An already-complete barrier must not fire the next stage a second time."""
    barrier.create(["a", "b"])
    barrier.arrive("a")
    assert barrier.arrive("b") is True
    assert barrier.arrive("b") is False, "re-arrival after completion must not refire"
    assert barrier.arrive("a") is False


def test_arrive_before_create_raises_instead_of_firing(barrier: Barrier) -> None:
    """dramatiq's Barrier reads a missing key as 0 and returns True -- a silent early fire.

    A child racing ahead of create() must be a loud error, not a fired next stage.
    """
    with pytest.raises(BarrierNotCreated):
        barrier.arrive("a")


def test_arrive_after_ttl_expiry_raises(redis_client: Redis) -> None:
    """An expired barrier must not read as complete."""
    b = Barrier(redis_client, run_id=f"test-{uuid.uuid4()}", stage="fetch", ttl=60)
    b.create(["a", "b"])
    redis_client.delete(b.total_key)  # simulate TTL expiry of the total
    with pytest.raises(BarrierNotCreated):
        b.arrive("a")


def test_zero_parties_raises_rather_than_deadlocking(barrier: Barrier) -> None:
    """An empty fan-out (LLM returned no queries, or everything deduped away).

    dramatiq asserts on parties=0, but `python -O` strips asserts and the barrier then
    fires on the first waiter. The caller must guard the empty case explicitly.
    """
    with pytest.raises(ValueError, match="zero parties"):
        barrier.create([])


def test_duplicate_party_ids_rejected(barrier: Barrier) -> None:
    """Duplicate ids would make the barrier uncompletable -- SCARD could never reach total."""
    with pytest.raises(ValueError, match="duplicate"):
        barrier.create(["a", "b", "a"])


def test_create_is_idempotent_on_parent_retry(barrier: Barrier) -> None:
    """A retried fan-out parent must not reset the total under already-arrived parties."""
    assert barrier.create(["a", "b"]) is True
    barrier.arrive("a")
    assert barrier.create(["a", "b"]) is False, "second create must be a no-op"
    assert barrier.progress() == (1, 2), "arrivals must survive a parent retry"
    assert barrier.arrive("b") is True


def test_concurrent_final_arrivals_fire_once(redis_client: Redis) -> None:
    """SADD+SCARD as two round trips lets two concurrent final parties both fire.

    With total=2, if both SADDs land before either SCARD, both SCARDs return 2. The
    single EVAL is what prevents it. This drives real concurrency at it rather than
    trusting the read of the Lua.
    """
    from concurrent.futures import ThreadPoolExecutor

    fire_counts = []
    for _ in range(25):
        b = Barrier(redis_client, run_id=f"test-{uuid.uuid4()}", stage="score", ttl=60)
        b.create(["a", "b"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(b.arrive, ["a", "b"]))
        fire_counts.append(sum(results))
        redis_client.delete(b.arrived_key, b.total_key)

    assert set(fire_counts) == {1}, f"barrier must fire exactly once per run, got {fire_counts}"
