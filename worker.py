"""Dramatiq worker entrypoint: `dramatiq worker --queues <queue>`.

Importing pipeline.actors is what registers every actor and, via core.broker, sets the
global broker. Workers on Windows spawn rather than fork, so this import runs in each
worker process -- which is exactly why the broker must be built at module scope.
"""

from pipeline import actors  # noqa: F401

__all__ = ["actors"]
