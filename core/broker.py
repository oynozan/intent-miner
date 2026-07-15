"""Dramatiq broker.

Module-level construction is mandatory, not stylistic. Workers on Windows spawn
rather than fork, so each process re-imports this module instead of inheriting the
parent's objects. Anything built under ``if __name__ == "__main__"`` simply does not
exist in a worker, and the failure is a confusing "no broker declared" rather than
anything that points here.
"""

from __future__ import annotations

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from redis import Redis

from core.config import settings

redis_client = Redis.from_url(settings().redis_url)

broker = RedisBroker(url=settings().redis_url)
dramatiq.set_broker(broker)
