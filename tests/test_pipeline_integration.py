"""End-to-end barrier behaviour through real dramatiq actors and a real broker.

tests/test_barriers.py tests the barrier primitive directly. This tests the thing that
actually breaks in production: the *actor lifecycle* around it -- retries, terminal
failures, and fan-out ordering, driven through a real worker.

Uses dramatiq's StubBroker so the middleware stack (Retries, Callbacks) is the genuine
one, while ``join()`` gives deterministic completion instead of sleeps.
"""

from __future__ import annotations

import os
import uuid

import pytest
from redis import Redis
from redis.exceptions import RedisError

import dramatiq
from dramatiq import Worker
from dramatiq.brokers.stub import StubBroker

from core.barriers import Barrier

REDIS_HOST = os.environ.get("TEST_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("TEST_REDIS_PORT", "6380"))


@pytest.fixture(scope="session")
def redis_client() -> Redis:
    client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=14, decode_responses=True)
    try:
        client.ping()
    except RedisError as exc:
        pytest.fail(f"redis unavailable on {REDIS_HOST}:{REDIS_PORT} ({exc})")
    return client


@pytest.fixture
def broker() -> StubBroker:
    b = StubBroker()
    b.emit_after("process_boot")
    dramatiq.set_broker(b)
    yield b
    b.flush_all()
    b.close()


@pytest.fixture
def worker(broker: StubBroker) -> Worker:
    w = Worker(broker, worker_timeout=100)
    w.start()
    yield w
    w.stop()


@pytest.fixture
def run_id(redis_client: Redis) -> str:
    rid = f"itest-{uuid.uuid4()}"
    yield rid
    for key in redis_client.scan_iter(f"barrier:{rid}:*"):
        redis_client.delete(key)


def test_fanout_fires_next_stage_exactly_once(
    broker: StubBroker, worker: Worker, redis_client: Redis, run_id: str
) -> None:
    """Ten parties, one downstream fire. The basic contract."""
    fired: list[str] = []

    @dramatiq.actor(max_retries=0)
    def next_stage(rid: str) -> None:
        fired.append(rid)

    @dramatiq.actor(max_retries=0)
    def child(rid: str, party: str) -> None:
        if Barrier(redis_client, rid, "discover").arrive(party):
            next_stage.send(rid)

    parties = [f"q{i}" for i in range(10)]
    Barrier(redis_client, run_id, "discover").create(parties)   # create BEFORE send
    for p in parties:
        child.send(run_id, p)

    broker.join(child.queue_name)
    broker.join(next_stage.queue_name)
    worker.join()

    assert fired == [run_id], f"next stage must fire exactly once, fired {len(fired)}x"


def test_retrying_child_does_not_fire_barrier_early(
    broker: StubBroker, worker: Worker, redis_client: Redis, run_id: str
) -> None:
    """The bug the whole barrier rewrite exists for, exercised through real retries.

    Party 'a' fails twice and succeeds on its third attempt. A counting barrier would
    have fired after a's three arrivals while 'b' and 'c' had not run. Crucially, the
    fire must happen only after the LAST party arrives -- not before.
    """
    fired: list[str] = []
    attempts: dict[str, int] = {}
    order: list[str] = []

    @dramatiq.actor(max_retries=0)
    def next_stage(rid: str) -> None:
        fired.append(rid)
        order.append("FIRE")

    @dramatiq.actor(max_retries=5, min_backoff=1, max_backoff=10)
    def flaky_child(rid: str, party: str) -> None:
        attempts[party] = attempts.get(party, 0) + 1
        if party == "a" and attempts[party] < 3:
            raise RuntimeError(f"transient failure #{attempts[party]}")
        order.append(party)
        # Success path only. No finally: arriving per-attempt is the trap.
        if Barrier(redis_client, rid, "discover").arrive(party):
            next_stage.send(rid)

    parties = ["a", "b", "c"]
    Barrier(redis_client, run_id, "discover").create(parties)
    for p in parties:
        flaky_child.send(run_id, p)

    # Party 'a' fails twice before succeeding; fail_fast would re-raise those.
    broker.join(flaky_child.queue_name, fail_fast=False)
    broker.join(next_stage.queue_name, fail_fast=False)
    worker.join()

    assert attempts["a"] == 3, "party a should have retried twice then succeeded"
    assert fired == [run_id], f"barrier must fire exactly once despite retries, got {len(fired)}"
    assert order[-1] == "FIRE", f"fire must come after the last party arrived, got {order}"


def test_terminally_failed_child_still_releases_barrier(
    broker: StubBroker, worker: Worker, redis_client: Redis, run_id: str
) -> None:
    """Partial failure degrades a run; it must not hang it.

    on_retry_exhausted owns the terminal path. Exactly one of {success, exhausted}
    arrives per party -- never both, never neither.
    """
    fired: list[str] = []
    failed: list[str] = []

    @dramatiq.actor(max_retries=0)
    def next_stage(rid: str) -> None:
        fired.append(rid)

    @dramatiq.actor(max_retries=0)
    def child_failed(message: dict, retry_info: dict) -> None:
        rid, party = message["args"][0], message["args"][1]
        failed.append(party)
        if Barrier(redis_client, rid, "discover").arrive(party):
            next_stage.send(rid)

    @dramatiq.actor(max_retries=1, min_backoff=1, max_backoff=5, on_retry_exhausted="child_failed")
    def child(rid: str, party: str) -> None:
        if party == "doomed":
            raise RuntimeError("permanent failure")
        if Barrier(redis_client, rid, "discover").arrive(party):
            next_stage.send(rid)

    parties = ["ok1", "ok2", "doomed"]
    Barrier(redis_client, run_id, "discover").create(parties)
    for p in parties:
        child.send(run_id, p)

    # fail_fast defaults to True as of dramatiq 2.0.0, and would re-raise the doomed
    # child's exception here. Dead-lettering is the path under test, not an accident.
    broker.join(child.queue_name, fail_fast=False)
    broker.join(child_failed.queue_name, fail_fast=False)
    broker.join(next_stage.queue_name, fail_fast=False)
    worker.join()

    assert failed == ["doomed"], "the doomed party must reach the terminal handler"
    assert fired == [run_id], "a permanently failed child must still release the barrier"
    arrived, total = Barrier(redis_client, run_id, "discover").progress()
    assert (arrived, total) == (3, 3)


def test_empty_fanout_skips_to_next_stage_instead_of_hanging(
    broker: StubBroker, worker: Worker, redis_client: Redis, run_id: str
) -> None:
    """Zero parties cannot be completed by any arrival. The guard must skip the stage.

    Without it the run waits forever on a barrier nobody can close -- and dramatiq's
    own Barrier makes this worse: parties=0 raises AssertionError, which `python -O`
    strips, after which the barrier fires on the first waiter.
    """
    from pipeline.stages import fan_out

    fired: list[str] = []

    def on_empty() -> None:
        fired.append(run_id)

    fan_out(run_id, "discover", [], send=lambda p: pytest.fail("must not send"), on_empty=on_empty)
    assert fired == [run_id], "empty fan-out must advance the pipeline, not hang"
