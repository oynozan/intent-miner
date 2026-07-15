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
            kwargs={"row_factory": dict_row},
        )
        _pool.open()
    return _pool


@contextmanager
def connection() -> Iterator[Connection]:
    with pool().connection() as conn:
        yield conn
