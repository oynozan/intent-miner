"""Fan-in barriers for the pipeline.

This deliberately does NOT use ``dramatiq.rate_limits.Barrier``. That implementation
is a blind ``DECR`` with no party identity, which breaks under two conditions this
pipeline actually hits:

* **Retries.** A retried actor decrements once per attempt. Dramatiq cannot tell
  three attempts of one party from three separate parties, so the barrier fires
  early while other parties have not run.
* **Redelivery.** Dramatiq is at-least-once. A worker that dies after decrementing
  but before acking gets its message redelivered, and the barrier decrements twice.

Both are fixed by keying on party identity instead of counting: a set of arrived
party ids, so a repeat arrival is a no-op. ``SADD`` and ``SCARD`` must be one
atomic EVAL -- as two round trips, two concurrent final parties can both observe
the full count and both fire.

Guarantee: ``arrive()`` returns True **at most once** per barrier. Not exactly
once -- a worker dying between the EVAL and enqueuing the next stage converts a
double-fire into a never-fire. That is the right trade (a double-fire corrupts the
run; a never-fire stalls one run visibly), but it means the caller must enqueue the
continuation inside the message's ack boundary, and downstream actors must be
idempotent anyway.
"""

from __future__ import annotations

import logging
from typing import Iterable

from redis import Redis

log = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 6 * 3600

# Returns 1 if this call created the barrier, 0 if it already existed.
# SET NX makes a parent retry a no-op rather than resetting the total.
_CREATE_LUA = """
local created = redis.call('SET', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[2])
if created then return 1 end
return 0
"""

# Returns:
#    1 -> this arrival completed the barrier (fire the next stage)
#    0 -> arrival recorded, barrier not yet complete (or duplicate arrival)
#   -1 -> barrier does not exist (never created, or TTL expired)
#
# The -1 case is why the total lives in Redis rather than being passed by the
# caller. dramatiq's Barrier reads a missing key as 0 and returns True, so a child
# racing ahead of create() fires the next stage instantly. Here a missing total is
# an explicit error instead of a silent early fire.
_ARRIVE_LUA = """
local total = redis.call('GET', KEYS[2])
if not total then return -1 end
local added = redis.call('SADD', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
if added == 0 then return 0 end
if redis.call('SCARD', KEYS[1]) == tonumber(total) then return 1 end
return 0
"""


class BarrierNotCreated(RuntimeError):
    """Raised when a party arrives at a barrier that does not exist.

    Means either the fan-out ordering is wrong (children sent before create) or the
    barrier's TTL expired mid-stage. Never swallow this -- it is the failure mode
    that silently corrupts a run.
    """


class Barrier:
    """A party-identity fan-in barrier.

    Usage is strictly ordered::

        b = Barrier(redis, run_id, "discover")
        b.create(query_ids)        # 1. persist the total FIRST
        for qid in query_ids:      # 2. only then send the children
            run_query.send(run_id, qid)

    Creating after sending lets a fast child arrive at a barrier that does not
    exist yet.
    """

    def __init__(self, redis: Redis, run_id: str, stage: str, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self.redis = redis
        self.run_id = run_id
        self.stage = stage
        self.ttl = ttl
        self._create = redis.register_script(_CREATE_LUA)
        self._arrive = redis.register_script(_ARRIVE_LUA)

    @property
    def arrived_key(self) -> str:
        return f"barrier:{self.run_id}:{self.stage}:arrived"

    @property
    def total_key(self) -> str:
        return f"barrier:{self.run_id}:{self.stage}:total"

    def create(self, parties: Iterable[str]) -> bool:
        """Persist the expected party set. Returns False if already created.

        Raises ValueError on an empty party set: a zero-party barrier can never be
        completed by an arrival, so the caller must skip straight to the next stage.
        dramatiq's Barrier asserts on parties=0, but `python -O` strips asserts and
        the barrier then fires on the first waiter -- so this is an explicit raise.
        """
        party_list = list(parties)
        if not party_list:
            raise ValueError(
                f"Barrier {self.run_id}:{self.stage} has zero parties. "
                "Guard the empty fan-out at the call site and advance to the next stage directly."
            )
        if len(set(party_list)) != len(party_list):
            raise ValueError(f"Barrier {self.run_id}:{self.stage} got duplicate party ids")

        created = self._create(keys=[self.total_key], args=[len(party_list), self.ttl])
        if not created:
            log.info("barrier %s:%s already created (parent retry?)", self.run_id, self.stage)
        return bool(created)

    def arrive(self, party_id: str) -> bool:
        """Record one party's completion. True only for the arrival that completes it.

        Idempotent: a redelivered or retried party arriving twice returns False the
        second time and does not advance the count.
        """
        result = self._arrive(keys=[self.arrived_key, self.total_key], args=[party_id, self.ttl])
        if result == -1:
            raise BarrierNotCreated(
                f"Party {party_id!r} arrived at barrier {self.run_id}:{self.stage}, "
                "which does not exist. Either children were sent before create(), or the TTL expired."
            )
        return result == 1

    def progress(self) -> tuple[int, int]:
        """(arrived, total) -- for the partial-barrier policy and stats. Not atomic."""
        with self.redis.pipeline() as pipe:
            pipe.scard(self.arrived_key)
            pipe.get(self.total_key)
            arrived, total = pipe.execute()
        return int(arrived or 0), int(total or 0)
