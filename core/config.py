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

    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    voyage_api_key: str = field(default_factory=lambda: os.environ.get("VOYAGE_API_KEY", ""))
    serper_api_key: str = field(default_factory=lambda: os.environ.get("SERPER_API_KEY", ""))
    serpapi_api_key: str = field(default_factory=lambda: os.environ.get("SERPAPI_API_KEY", ""))

    # Discovery provider order. "serper" is primary; SerpApi is the fallback, tried
    # only when Serper errors or has no key. Set DISCOVERY_PROVIDER=serpapi to flip.
    discovery_provider: str = field(default_factory=lambda: os.environ.get("DISCOVERY_PROVIDER", "serper"))

    # LLM provider order. "openai" is primary; the other named provider is the
    # fallback, tried only when the primary errors or has no key. Set LLM_PROVIDER=
    # anthropic to flip the order.
    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "openai"))

    # --- OpenAI (primary) -----------------------------------------------------
    # gpt-5-nano for both tasks, per request ("for everything"). Separable so a
    # bigger OpenAI model can be dropped into expand -- the one call where model
    # quality changes the product -- without touching code.
    #
    # gpt-5-nano is a reasoning model: the token cap is max_completion_tokens (which
    # also covers reasoning tokens), depth is reasoning_effort (low|medium|high), and
    # temperature/top_p are rejected. Expand gets a generous cap because reasoning
    # tokens eat into it before the tree is emitted.
    openai_expand_model: str = field(default_factory=lambda: os.environ.get("OPENAI_EXPAND_MODEL", "gpt-5-nano"))
    openai_score_model: str = field(default_factory=lambda: os.environ.get("OPENAI_SCORE_MODEL", "gpt-5-nano"))
    openai_expand_effort: str = field(default_factory=lambda: os.environ.get("OPENAI_EXPAND_EFFORT", "high"))
    openai_score_effort: str = field(default_factory=lambda: os.environ.get("OPENAI_SCORE_EFFORT", "low"))
    # 24000 was measured as enough and then failed a run: "autoblogging plugin wordpress"
    # expanded fine once and, on a re-run of the same keyword, burned the whole budget on
    # reasoning and came back finish_reason=length -- three times, exhausting the actor's
    # retries and failing the run before discovery. Reasoning length varies per call, so
    # a cap that only just fits is a coin flip. Unused budget is not billed; a failed run
    # is. Headroom is the cheaper side of that trade.
    openai_expand_max_tokens: int = field(default_factory=lambda: int(os.environ.get("OPENAI_EXPAND_MAX_TOKENS", "48000")))
    openai_score_max_tokens: int = field(default_factory=lambda: int(os.environ.get("OPENAI_SCORE_MAX_TOKENS", "8000")))

    # --- Anthropic (fallback) -------------------------------------------------
    # Reached only when OpenAI is down or unset. Opus for the tree (the one call a
    # big model earns its cost on), Haiku for scoring.
    expand_model: str = field(default_factory=lambda: os.environ.get("EXPAND_MODEL", "claude-opus-4-8"))
    score_model: str = field(default_factory=lambda: os.environ.get("SCORE_MODEL", "claude-haiku-4-5"))

    # --- Embeddings -----------------------------------------------------------
    # Voyage primary, OpenAI fallback. voyage-4-lite (output_dimension=1024) and
    # text-embedding-3-small (dimensions=1024) both produce 1024-dim vectors, so the
    # vector(1024) schema fits either. NOTE: a run's leaf vectors and candidate
    # vectors must come from the SAME provider to be comparable -- both stages pick
    # the first configured provider, so they stay consistent unless the primary
    # fails intermittently between them (see llm/embeddings.py).
    embed_provider: str = field(default_factory=lambda: os.environ.get("EMBED_PROVIDER", "voyage"))
    openai_embed_model: str = field(default_factory=lambda: os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", "voyage-4-lite"))
    # Passed explicitly on every call, both providers. OpenAI reduces to this via
    # `dimensions`; Voyage's HF weights default to 2048 while its API defaults to
    # 1024, so an implicit default lets dev and prod produce vectors that do not
    # share a space -- a silent failure, just meaningless cosines.
    embed_dimension: int = field(default_factory=lambda: int(os.environ.get("EMBED_DIMENSION", "1024")))

    # Hard per-run ceiling on discovery spend. A runaway tree or a retry storm is
    # bounded here rather than on the invoice.
    max_queries_per_run: int = field(default_factory=lambda: int(os.environ.get("MAX_QUERIES_PER_RUN", "400")))
    # LinkedIn returns ~8-10% of leads for ~30-45% of discovery spend (measured across
    # two keywords; see _query_rows). The LLM writes several LinkedIn queries per leaf
    # and every one used to be issued. One knob for the one platform that needs it --
    # do not generalise to a per-platform map until a second platform earns one.
    linkedin_queries_per_leaf: int = field(default_factory=lambda: int(os.environ.get("LINKEDIN_QUERIES_PER_LEAF", "1")))
    # Proceed to the next stage at this completion fraction. One hung provider must
    # not stall an entire run behind a barrier that never completes.
    barrier_min_completion: float = field(default_factory=lambda: float(os.environ.get("BARRIER_MIN_COMPLETION", "0.95")))
    # A three-year-old pain post is a dead lead. This is the only freshness lever
    # available on LinkedIn, and the only one Quora structurally cannot give us.
    freshness_months: int = field(default_factory=lambda: int(os.environ.get("FRESHNESS_MONTHS", "18")))

    recency_half_life_days: float = field(default_factory=lambda: float(os.environ.get("RECENCY_HALF_LIFE_DAYS", "180")))

    # Ceiling on the engagement term in _final_score, which multiplies by
    # (1 + min(log1p(engagement), cap)). 1.0 means a post can at most double its score on
    # popularity -- reached around 2 upvotes, so engagement separates near-ties and stops
    # short of overturning a better pain/ICP match. Uncapped, 1,000 upvotes multiplied by
    # 7.9 and a viral off-target post outranked a quiet perfect one, which also
    # contradicted treating a crowded thread as un-actionable. Raise only if leads that
    # deserve the top are being held off it.
    engagement_cap: float = field(default_factory=lambda: float(os.environ.get("ENGAGEMENT_CAP", "1.0")))

    # LinkedIn throttles a single IP under crawl volume, serving login-gated 200s with no
    # post data. Two levers keep fetches under its radar: LinkedIn runs on a dedicated
    # low-concurrency queue (see docker-compose worker-fetch-linkedin), and each fetch
    # waits a random 0..jitter_ms first so even that low concurrency does not become a
    # tight burst. If throttling still trips this many times in a run, the circuit
    # breaker skips the rest rather than hammering a throttled IP for the whole run.
    linkedin_fetch_jitter_ms: int = field(default_factory=lambda: int(os.environ.get("LINKEDIN_FETCH_JITTER_MS", "500")))
    linkedin_throttle_breaker: int = field(default_factory=lambda: int(os.environ.get("LINKEDIN_THROTTLE_BREAKER", "20")))

    # --- x402 / A2MCP (paid agent surface) ------------------------------------
    # Seller side of the payment protocol: what /a2mcp/* charges, in what token, to
    # whom. Prices are plain decimal strings in whole USDT -- the same format OKX.AI's
    # service listing takes -- and are converted to base units when the challenge is
    # built. Keeping the listed price and the charged price in ONE place is the point:
    # a listing that says 0.05 and a challenge that says 0.5 is a silent 10x overcharge
    # that nothing downstream would flag.
    a2mcp_price_create_job: str = field(default_factory=lambda: os.environ.get("A2MCP_PRICE_CREATE_JOB", "0.05"))
    a2mcp_price_job_status: str = field(default_factory=lambda: os.environ.get("A2MCP_PRICE_JOB_STATUS", "0.001"))

    # Public https base of THIS service. It goes in the challenge's `resource` and must
    # match the endpoint registered on-chain -- a buyer that pays for one resource and
    # replays against another is exactly what `resource` exists to prevent.
    a2mcp_base_url: str = field(default_factory=lambda: os.environ.get("A2MCP_BASE_URL", ""))

    # Where the money lands. No default: an empty payTo would mint a challenge that
    # collects for nobody, so the challenge builder refuses rather than guessing.
    x402_pay_to: str = field(default_factory=lambda: os.environ.get("X402_PAY_TO", ""))
    # X Layer (196) by default -- it is OKX-native and charges zero gas, so a 0.001 USDT
    # call is not swamped by its own settlement cost. CAIP-2 is derived, not configured.
    x402_chain_id: int = field(default_factory=lambda: int(os.environ.get("X402_CHAIN_ID", "196")))
    x402_asset: str = field(default_factory=lambda: os.environ.get("X402_ASSET", ""))
    x402_asset_decimals: int = field(default_factory=lambda: int(os.environ.get("X402_ASSET_DECIMALS", "6")))
    # EIP-712 domain of the asset contract, echoed in `accepts[].extra`. The buyer's
    # signer reads these to build the typed-data domain; a wrong `name` produces a
    # signature the facilitator rejects with no useful message.
    x402_asset_name: str = field(default_factory=lambda: os.environ.get("X402_ASSET_NAME", "USDT"))
    x402_asset_version: str = field(default_factory=lambda: os.environ.get("X402_ASSET_VERSION", "2"))
    # How long a minted challenge stays signable.
    x402_timeout_seconds: int = field(default_factory=lambda: int(os.environ.get("X402_TIMEOUT_SECONDS", "300")))

    # Facilitator that verifies and settles a presented payment. Unset = the paid
    # surface fails closed (402 on every replay) rather than serving work for free.
    # See core/x402.py: this is the ONE integration point still to be confirmed against
    # OKX's facilitator before the service can take real money.
    x402_facilitator_verify_url: str = field(default_factory=lambda: os.environ.get("X402_FACILITATOR_VERIFY_URL", ""))
    x402_facilitator_settle_url: str = field(default_factory=lambda: os.environ.get("X402_FACILITATOR_SETTLE_URL", ""))
    x402_facilitator_api_key: str = field(default_factory=lambda: os.environ.get("X402_FACILITATOR_API_KEY", ""))


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
