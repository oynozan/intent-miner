"""Fan-out / fan-in helpers shared by every stage.

Every stage in this pipeline has the same shape, and the same three ways to go wrong.
They are solved once, here, rather than re-derived (and re-broken) per stage.

**1. Ordering.** Persist the party set, *then* send the children. A child that arrives
at a barrier which does not exist yet would otherwise fire the next stage instantly.

**2. The empty fan-out.** Zero parties cannot be completed by any arrival, so the run
would hang forever behind a barrier nobody can close. Skip straight to the next stage.

**3. Exactly-one arrival per party.** Each actor must arrive on the success path *or*
the terminal-failure path, never both and never neither. Two rules make that hold:

  - Do NOT wrap the body in try/except. Swallowing the exception is what makes
    ``max_retries`` dead code: dramatiq's Retries middleware opens with
    ``if exception is None: return``, so nothing retries if nothing propagates.
  - Do NOT arrive in a ``finally``. That is the trap the obvious fix walks into --
    it arrives once per *attempt*, and the barrier cannot tell four attempts of one
    party from four parties. (Our Lua barrier is keyed on party id, so it would
    actually survive this; the pattern is still wrong because a party that eventually
    succeeds after retries would arrive during its failed attempts. Don't rely on the
    barrier to paper over a lifecycle bug.)

Use ``on_retry_exhausted`` for the failure path. Not ``on_failure``: that fires per
attempt, because the Callbacks middleware branches only on ``if exception is None``
with no check for whether the message is terminally dead. Note the option is singular
-- the changelog prose says ``on_retries_exhausted``; the code says otherwise, and the
code wins.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

from core.barriers import Barrier
from core.broker import redis_client
from core.config import settings

log = logging.getLogger(__name__)


def barrier(run_id: str, stage: str) -> Barrier:
    return Barrier(redis_client, run_id, stage)


def fan_out(
    run_id: str,
    stage: str,
    party_ids: Sequence[str],
    send: Callable[[str], None],
    on_empty: Callable[[], None],
) -> None:
    """Open a barrier for ``party_ids`` and dispatch them. Handles the empty case.

    ``send`` is called once per party id. ``on_empty`` runs instead when there is
    nothing to do, and must advance the pipeline itself.
    """
    if not party_ids:
        log.warning("run %s stage %s: empty fan-out, skipping to next stage", run_id, stage)
        on_empty()
        return

    b = barrier(run_id, stage)
    created = b.create(party_ids)
    if not created:
        # A retried fan-out parent. create() is SET NX so the total is intact and
        # arrivals so far are preserved -- but the children are about to be re-sent,
        # and they will arrive again. The barrier tolerates that (SADD is idempotent
        # on party id), which is exactly why it is keyed on identity.
        log.info("run %s stage %s: barrier exists, re-sending children (parent retry)", run_id, stage)

    # Strictly after create(): a fast child must never beat the barrier into existence.
    for party_id in party_ids:
        send(party_id)


def arrive(run_id: str, stage: str, party_id: str, then: Callable[[], None]) -> None:
    """Record one party's completion; run ``then`` iff this arrival completed the stage.

    ``then`` must be an enqueue, not real work: it runs inside the arriving message's
    processing window, and anything slow here widens the gap between the barrier
    firing and the continuation being durably queued. If the worker dies in that gap,
    an at-most-once barrier turns into a never-fire.
    """
    if barrier(run_id, stage).arrive(party_id):
        log.info("run %s stage %s: barrier complete", run_id, stage)
        then()


def should_proceed(run_id: str, stage: str) -> bool:
    """Partial-barrier policy: has the stage completed *enough* to move on?

    One provider hanging must not stall an entire run behind a barrier that will never
    complete. This is the escape hatch a watchdog consults; the normal path is still
    the barrier firing on its own.
    """
    arrived, total = barrier(run_id, stage).progress()
    if not total:
        return False
    return (arrived / total) >= settings().barrier_min_completion
