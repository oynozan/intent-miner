"""The paid agent surface — two A2MCP services, priced per call in USDT.

These are deliberately *not* the same routes as `/runs`. The free routes are the human
API and may change shape whenever the pipeline does; these two are a paid contract whose
price and endpoint URL are written on-chain at registration and cannot be edited without
another update transaction. Keeping them separate means an internal refactor cannot
silently change what a buyer already paid for.

Both take `job_id` / params in ways that survive a *static* registered URL: the status
service reads `job_id` from the query string rather than a path segment, because OKX.AI
stores one fixed endpoint per service and `payment quote <url> --param job_id=...` puts
known params in the query string for GET.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field

from core import x402
from pipeline import repo

router = APIRouter(prefix="/a2mcp", tags=["a2mcp"])

CREATE_PATH = "/a2mcp/jobs"
STATUS_PATH = "/a2mcp/jobs/status"

CREATE_DESCRIPTION = "Start an intent-mining job from one keyword."
STATUS_DESCRIPTION = "Job status plus the ranked links discovered so far."

# Declares the shape of the PAID replay, not of this documentation. `payment quote`
# probes with GET unless `method` says otherwise -- omitting it on the POST service
# would return 405 and the buyer would see `endpoint_unreachable` instead of a price.
CREATE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "http",
    "method": "POST",
    "bodyType": "json",
    "body": {
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "The solution or product to mine pain for."},
            "icp": {"type": "string", "description": "Optional ideal-customer hint."},
        },
        "required": ["keyword"],
    },
}

STATUS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "http",
    "method": "GET",
    "queryParams": {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "The job_id returned when the job was created."},
            "min_score": {"type": "number", "description": "Minimum score; 0.01 filters out ads and dead threads."},
        },
        "required": ["job_id"],
    },
}


class CreateJob(BaseModel):
    keyword: str = Field(min_length=2, max_length=500, description="The solution or product to mine pain for.")
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
    status_code=200,
    summary="Create a job from a keyword (paid)",
    responses={402: {"description": "Payment required. The challenge is in `PAYMENT-REQUIRED`."}},
)
def create_job(body: CreateJob, request: Request, response: Response) -> JobCreated:
    """Charge, then queue an intent-mining job for one keyword.

    200 rather than 202: a buyer who has just paid gets a settled receipt, and a 2xx
    that is not 200 reads as "maybe" to a paying client. The work is still asynchronous
    -- poll the status service.
    """
    settled = x402.require_payment(
        request.headers.get(x402.PAYMENT_SIGNATURE_HEADER),
        path=CREATE_PATH,
        price=_price_create(),
        input_schema=CREATE_INPUT_SCHEMA,
        description=CREATE_DESCRIPTION,
    )

    # Imported after the gate, deliberately. This pulls in dramatiq and the whole actor
    # chain; doing it first would make every unpaid probe -- which is every buyer's FIRST
    # request -- pay for it, and would turn an import error anywhere in the pipeline into
    # a 500 on what should be a clean 402 price quote.
    from pipeline.actors import start_run

    run_id = repo.create_run(body.keyword, body.icp)
    start_run.send(run_id)
    response.headers[x402.PAYMENT_RESPONSE_HEADER] = x402.receipt_header(settled)
    return JobCreated(job_id=run_id, status="pending")


@router.get(
    "/jobs/status",
    response_model=JobStatus,
    summary="Get job status and discovered links (paid)",
    responses={402: {"description": "Payment required. The challenge is in `PAYMENT-REQUIRED`."}},
)
def job_status(
    request: Request,
    response: Response,
    job_id: str = Query(description="The job_id returned when the job was created."),
    min_score: float = Query(default=0.01, ge=0.0, description="0.01 filters out ads and dead threads."),
) -> JobStatus:
    """Charge, then return the job's stage and its ranked links.

    Callable before the job finishes -- it returns whatever has been scored so far, so a
    buyer can watch it fill in. An unknown `job_id` is still a paid call: the lookup
    happens after settlement, and refunding on a typo is not a thing this protocol does.
    """
    settled = x402.require_payment(
        request.headers.get(x402.PAYMENT_SIGNATURE_HEADER),
        path=STATUS_PATH,
        price=_price_status(),
        input_schema=STATUS_INPUT_SCHEMA,
        description=STATUS_DESCRIPTION,
    )

    run = repo.get_run(job_id)
    response.headers[x402.PAYMENT_RESPONSE_HEADER] = x402.receipt_header(settled)
    if not run:
        return JobStatus(job_id=job_id, status="not_found", urls=[], url_count=0)

    urls = repo.result_urls(job_id, node_id=None, min_score=min_score)
    return JobStatus(job_id=job_id, status=run["status"], urls=urls, url_count=len(urls))


def _price_create() -> str:
    from core.config import settings

    return settings().a2mcp_price_create_job


def _price_status() -> str:
    from core.config import settings

    return settings().a2mcp_price_job_status
