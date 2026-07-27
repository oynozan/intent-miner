"""Postgres connection pool.

Built at module import so it survives spawn-based workers (Windows) that re-import
rather than inherit. ``open=False`` + explicit ``open()`` avoids the pool trying to
connect at import time, which would make the module unimportable when Postgres is
briefly down.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from core.config import settings

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings().database_url,
            min_size=1,
            max_size=16,
            open=False,
            # connect_timeout is not a nicety. A host that resolves to several addresses
            # (`localhost` -> ::1 then 127.0.0.1 is the common one) makes libpq try each
            # in turn, and with no timeout a dead family hangs the pool's worker thread
            # for longer than the pool's own 30s wait -- so callers see PoolTimeout,
            # "couldn't get a connection", while a direct connect to the same URL
            # succeeds instantly. Bounding it per-address turns that into a fast failover.
            kwargs={"row_factory": dict_row, "connect_timeout": 5},
        )
        _pool.open()
    return _pool


@contextmanager
def connection() -> Iterator[Connection]:
    with pool().connection() as conn:
        yield conn
