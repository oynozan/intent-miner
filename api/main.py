from __future__ import annotations

import logging

from fastapi import FastAPI

from api.routes import router

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

Quora and LinkedIn are live sources. Reddit is deferred (no obtainable API credentials
today) -- see the project README.
"""

TAGS_METADATA = [
    {"name": "runs", "description": "Start a run and read its status, pain tree, and ranked results."},
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


@app.get("/health", tags=["health"], summary="Liveness check")
def health() -> dict[str, str]:
    """Returns `{"status": "ok"}` when the API process is up. Does not check Postgres,
    Redis, or the workers."""
    return {"status": "ok"}
