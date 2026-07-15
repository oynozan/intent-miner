"""Configuration.

Constructed at module import, not inside ``if __name__ == '__main__'``. Dramatiq
workers on Windows use spawn rather than fork, so every worker process re-imports
this module rather than inheriting it. Anything built under a main-guard simply
does not exist in a worker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env(key: str, default: str | None = None) -> str:
    value = os.environ.get(key, default)
    if value is None:
        raise RuntimeError(f"required environment variable {key} is not set")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "postgresql://intent:intent@localhost:5432/intent_miner"))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6380/0"))

    s3_endpoint: str = field(default_factory=lambda: _env("S3_ENDPOINT", "http://localhost:9002"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "intent-raw"))
    s3_access_key: str = field(default_factory=lambda: _env("S3_ACCESS_KEY", "minioadmin"))
    s3_secret_key: str = field(default_factory=lambda: _env("S3_SECRET_KEY", "minioadmin"))

    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    voyage_api_key: str = field(default_factory=lambda: os.environ.get("VOYAGE_API_KEY", ""))
    serper_api_key: str = field(default_factory=lambda: os.environ.get("SERPER_API_KEY", ""))

    # Opus for the tree: it is the one call where a big model changes the product,
    # because a bad pain translation cannot be recovered downstream. Haiku for
    # scoring: 45ish calls per run, and the plan is to bake it off against Sonnet 5
    # on labelled data before trusting it.
    expand_model: str = field(default_factory=lambda: os.environ.get("EXPAND_MODEL", "claude-opus-4-8"))
    score_model: str = field(default_factory=lambda: os.environ.get("SCORE_MODEL", "claude-haiku-4-5"))

    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", "voyage-4-lite"))
    # Must be passed explicitly on every call: the HuggingFace weights default to
    # 2048 while the API defaults to 1024, so an implicit default means dev and prod
    # can silently produce vectors that do not share a space.
    embed_dimension: int = field(default_factory=lambda: int(os.environ.get("EMBED_DIMENSION", "1024")))

    # Hard per-run ceiling on discovery spend. A runaway tree or a retry storm is
    # bounded here rather than on the invoice.
    max_queries_per_run: int = field(default_factory=lambda: int(os.environ.get("MAX_QUERIES_PER_RUN", "400")))
    # Proceed to the next stage at this completion fraction. One hung provider must
    # not stall an entire run behind a barrier that never completes.
    barrier_min_completion: float = field(default_factory=lambda: float(os.environ.get("BARRIER_MIN_COMPLETION", "0.95")))
    # A three-year-old pain post is a dead lead. This is the only freshness lever
    # available on LinkedIn, and the only one Quora structurally cannot give us.
    freshness_months: int = field(default_factory=lambda: int(os.environ.get("FRESHNESS_MONTHS", "18")))

    recency_half_life_days: float = field(default_factory=lambda: float(os.environ.get("RECENCY_HALF_LIFE_DAYS", "180")))


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
