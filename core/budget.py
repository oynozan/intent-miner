"""Cross-run spend counter.

A per-run query cap bounds one run. It does nothing about the failure that actually
costs money: routine dev iteration, a stuck retry loop, or a scheduled trigger quietly
running the same pipeline forty times. Those each stay under the per-run cap and add up.

So the ceiling is monthly and lives in Redis, outside any run's lifetime.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from redis import Redis

log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """The monthly credit ceiling is spent. Refuse rather than bill."""


class SpendCounter:
    def __init__(self, redis: Redis, name: str, monthly_limit: int) -> None:
        self.redis = redis
        self.name = name
        self.monthly_limit = monthly_limit

    def _key(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        return f"spend:{self.name}:{now:%Y-%m}"

    def spend(self, credits: int) -> int:
        """Charge credits and return the new month-to-date total.

        Increments first, then checks. Charging before checking means a burst of
        concurrent callers cannot slip past the ceiling by all reading a stale value --
        the atomic INCR is the serialization point. The cost is that the ceiling can be
        overshot by at most one in-flight batch, which is the right way to be wrong.
        """
        key = self._key()
        with self.redis.pipeline() as pipe:
            pipe.incrby(key, credits)
            pipe.expire(key, 70 * 24 * 3600)  # outlive the month, then self-clean
            total, _ = pipe.execute()

        if total > self.monthly_limit:
            log.error("monthly budget for %s exhausted: %d/%d", self.name, total, self.monthly_limit)
            raise BudgetExceeded(f"{self.name}: {total}/{self.monthly_limit} credits this month")
        return int(total)

    def used(self) -> int:
        return int(self.redis.get(self._key()) or 0)

    def remaining(self) -> int:
        return max(0, self.monthly_limit - self.used())
