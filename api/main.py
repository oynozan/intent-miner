from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.a2mcp import router as a2mcp_router
from api.routes import router
from core import x402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

DESCRIPTION = """
Describe a solution. Get ranked links to people describing the pain it solves.

**The one idea: do not search for your solution, search for the pain.** Nobody writes
"I need a video background removal API" -- they write "how do I cut myself out of a
video without a green screen." Every stage translates product language into complaint
language and back.

### How a run flows

`POST /runs` returns immediately with a `run_id`; the work happens asynchronously in a
worker pipeline. Poll `GET /runs/{run_id}` and watch `status` advance through:

`pending → expanding → discovering → fetching → prefiltering → scoring → done`

Then fetch `GET /runs/{run_id}/results`. Two intermediate views are also available:
`GET /runs/{run_id}/tree` shows the pain tree the run expanded to, and the run's
`stats` object reports per-stage metrics.

### What "results" means

Results are **grouped by leaf** (a specific pain), strongest leaf first -- which leaf
converts tells you where the demand is. A post is only a lead if the author is *seeking*
the solution (not selling it) and the thread is still *actionable* (recent, not answered
to death). Vendor ads and dead threads score exactly `0.0`, so filter with a
`min_score` slightly above 0.

### Scope

Quora, LinkedIn and Reddit are all live sources, and they are fetched differently on
purpose. **Reddit** is fetched from its `.json` view behind a browser-minted cookie jar,
so a Reddit result carries the real post body, `posted_at` and `engagement`. **Quora** is
**SERP-only** -- its gate became a quota that pacing and IP rotation do not clear, and the
search title already carries the question, which is what gets embedded anyway. A
SERP-only result has no `posted_at` and no `engagement`; Reddit falls back to that shape
only if the cookie jar fails. See the project README.
"""

TAGS_METADATA = [
    {"name": "runs", "description": "Start a run and read its status, pain tree, and ranked results."},
    {
        "name": "a2mcp",
        "description": (
            "The paid agent surface, priced per call in USDT and settled over x402. A call "
            "with no `PAYMENT-SIGNATURE` header returns **402** with the challenge in "
            "`PAYMENT-REQUIRED`; pay it and replay to get the result plus a "
            "`PAYMENT-RESPONSE` receipt. These two routes are the on-chain service contract "
            "-- the `/runs` routes are the free human API and are not price-stable."
        ),
    },
    {"name": "health", "description": "Liveness probe."},
]

app = FastAPI(
    title="Intent Miner",
    version="1.0.0",
    summary="Describe a solution; get ranked links to people describing the pain it solves.",
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    contact={"name": "Intent Miner"},
    license_info={"name": "Proprietary"},
)
app.include_router(router)
app.include_router(a2mcp_router)


@app.exception_handler(x402.PaymentRequired)
def _payment_required(request: Request, exc: x402.PaymentRequired) -> JSONResponse:
    """402 carrying the challenge in BOTH places a buyer looks.

    `PAYMENT-REQUIRED` is the x402 v2 signal and is what gets read first; the same JSON
    sits in the body because v1 clients look there, and because a human curling the
    endpoint should be able to see the price without base64-decoding a header.
    """
    return JSONResponse(
        status_code=402,
        content=exc.challenge | ({"error": exc.reason} if exc.reason else {}),
        headers={x402.PAYMENT_REQUIRED_HEADER: x402.challenge_header(exc.challenge)},
    )


@app.exception_handler(x402.Misconfigured)
def _misconfigured(request: Request, exc: x402.Misconfigured) -> JSONResponse:
    """500, never 402. Telling a buyer to pay a service that cannot collect or cannot
    verify is worse than telling them it is broken."""
    logging.getLogger(__name__).error("x402 seller misconfigured: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "payment surface is not configured"})


@app.get("/health", tags=["health"], summary="Liveness check")
def health() -> dict[str, str]:
    """Returns `{"status": "ok"}` when the API process is up. Does not check Postgres,
    Redis, or the workers."""
    return {"status": "ok"}
