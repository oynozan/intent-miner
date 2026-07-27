"""The paid agent surface — two A2MCP services, priced per call in USDT.

Deliberately not the same routes as `/runs`. The free routes are the human API and may
change shape whenever the pipeline does; these two are a paid contract whose price and
endpoint URL are written on-chain at registration and cannot be edited without another
update transaction. Separate routes mean an internal refactor cannot silently change
what a buyer already paid for.

**There is no payment code in this file.** `core.x402` installs OKX's middleware in
front of these paths, so by the time a handler runs the money has already settled.
The one thing the handlers still do is refuse to run when the payment gate was never
installed -- otherwise an unconfigured deployment would serve both services for free.

The status service takes `job_id` as a query param rather than a path segment because
OKX.AI stores one fixed endpoint URL per service, and a buyer's
`payment quote <url> --param job_id=...` puts known params in the query string for GET.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core import x402
from pipeline import repo

router = APIRouter(prefix="/a2mcp", tags=["a2mcp"])


def _gate() -> None:
    """503 when the paid surface cannot collect.

    Not 402: telling a buyer to pay a service with no configured payee or credentials
    invites a payment nobody can settle. Not 200 either -- serving the work would make
    an unconfigured deployment a free one.
    """
    if not x402.configured():
        raise HTTPException(503, f"paid surface unavailable: missing {', '.join(x402.missing())}")


class CreateJob(BaseModel):
    """The JSON-body form of the input. Both fields are optional here because the query
    string is an equally valid carrier -- create_job merges the two and validates the
    result, so requiring anything at this layer would reject a legitimate query-only
    call before the handler ever sees it."""

    keyword: str | None = Field(default=None, max_length=500, description="The solution or product to mine pain for.")
    icp: str | None = Field(default=None, max_length=500, description="Optional ideal-customer hint.")


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    urls: list[str]
    url_count: int


@router.post(
    "/jobs",
    response_model=JobCreated,
    summary="Create a job from a keyword (paid)",
    responses={402: {"description": "Payment required; the challenge is in `PAYMENT-REQUIRED`."}},
)
def create_job(
    body: CreateJob | None = None,
    keyword: str | None = Query(default=None, description="The solution or product to mine pain for."),
    icp: str | None = Query(default=None, description="Optional ideal-customer hint."),
) -> JobCreated:
    """Queue an intent-mining job for one keyword. Payment has already settled.

    **Accepts the keyword as a query param OR a JSON body, deliberately.** The x402
    challenge has no way to tell a buyer which to use -- the SDK's v2 RouteConfig has no
    output_schema field, that lives only in its legacy v1 types -- so the buyer chooses,
    and a real one was measured choosing neither: the paid replay arrived with an empty
    body and this route answered 422. On a free endpoint that is a bad request; on a paid
    one it is a buyer who has parted with money and received an error, which is the one
    outcome worth writing extra code to avoid. Read both, and fail with 400 and a usable
    message only when the keyword is genuinely absent.

    The work itself stays asynchronous -- poll the status service. Returns 200 rather
    than 202 because a buyer who has just paid should not have to interpret a 2xx that
    reads as "maybe".
    """
    _gate()

    keyword = keyword or (body.keyword if body else None)
    icp = icp or (body.icp if body else None)
    if not keyword or len(keyword.strip()) < 2:
        raise HTTPException(400, "keyword is required, as a query param or a JSON body field")
    keyword = keyword.strip()[:500]

    # Imported here rather than at module scope: this pulls in dramatiq and the whole
    # actor chain, and an import error anywhere in the pipeline should not turn an
    # unpaid probe -- every buyer's first request -- into a 500 instead of a price quote.
    from pipeline.actors import start_run

    run_id = repo.create_run(keyword, icp)
    start_run.send(run_id)
    return JobCreated(job_id=run_id, status="pending")


@router.get(
    "/jobs/status",
    response_model=JobStatus,
    summary="Get job status and discovered links (paid)",
    responses={402: {"description": "Payment required; the challenge is in `PAYMENT-REQUIRED`."}},
)
def job_status(
    job_id: str = Query(description="The job_id returned when the job was created."),
    min_score: float = Query(default=0.01, ge=0.0, description="0.01 filters out ads and dead threads."),
) -> JobStatus:
    """The job's stage and its ranked links.

    Callable before the job finishes -- it returns whatever has been scored so far, so a
    buyer can watch it fill in. An unknown `job_id` still consumed a paid call: payment
    settles in the middleware, before this handler ever sees the id.
    """
    _gate()

    run = repo.get_run(job_id)
    if not run:
        return JobStatus(job_id=job_id, status="not_found", urls=[], url_count=0)

    urls = repo.result_urls(job_id, node_id=None, min_score=min_score)
    return JobStatus(job_id=job_id, status=run["status"], urls=urls, url_count=len(urls))
