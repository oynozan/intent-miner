"""Seller side of x402 — the OKX facilitator, wired through the official SDK.

This file used to hand-roll the whole thing: mint the challenge, call `/verify`, call
`/settle`. That was written while the broker contract was unknown, inferred from
buyer-side documentation, and it was wrong in the way that matters — OKX's broker
authenticates with HMAC-SHA256 request signing, which no amount of reading the buyer
docs would have revealed. It would have failed closed forever while looking exactly
like a configuration problem.

So the protocol work belongs to ``okxweb3-app-x402``. What is left here is the part
that is genuinely ours: deciding what each route costs, and refusing to serve the paid
surface at all when it is not configured to collect.

**Fail closed, deliberately.** With no credentials the middleware is never installed
and every /a2mcp/* route 503s. The alternative failure mode is a service that hands out
paid work for free and looks perfectly healthy doing it, which is strictly worse than
being down. There is no bypass flag for the same reason.
"""

from __future__ import annotations

import logging

from core.config import settings

log = logging.getLogger(__name__)

# Route keys are "<METHOD> <path>" -- the SDK matches on this exact string, so they
# must track api/a2mcp.py's decorators. Kept here beside the prices they gate.
CREATE_ROUTE = "POST /a2mcp/jobs"
STATUS_ROUTE = "GET /a2mcp/jobs/status"


class Misconfigured(Exception):
    """The paid surface cannot collect. A 503 -- never a 402, which would tell a buyer
    to pay a service that has no way to take the money."""


def network() -> str:
    return f"eip155:{settings().x402_chain_id}"


def configured() -> bool:
    """True when there is enough to actually charge for a call."""
    cfg = settings()
    return bool(
        cfg.okx_api_key and cfg.okx_secret_key and cfg.okx_passphrase
        and cfg.x402_pay_to and cfg.a2mcp_base_url
    )


def missing() -> list[str]:
    """Which settings are absent, for a startup log that names the gap instead of
    leaving someone to infer it from a 503."""
    cfg = settings()
    return [
        name
        for name, value in (
            ("OKX_API_KEY", cfg.okx_api_key),
            ("OKX_SECRET_KEY", cfg.okx_secret_key),
            ("OKX_PASSPHRASE", cfg.okx_passphrase),
            ("X402_PAY_TO", cfg.x402_pay_to),
            ("A2MCP_BASE_URL", cfg.a2mcp_base_url),
        )
        if not value
    ]


def routes() -> dict:
    """The priced routes, in the SDK's shape.

    Price is a dollar-prefixed string because that is what PaymentOption takes; the
    bare decimal in config is the same number registered on-chain, so the two cannot
    drift as long as nothing else formats a price.
    """
    from x402.http import PaymentOption
    from x402.http.types import RouteConfig

    cfg = settings()
    if not configured():
        raise Misconfigured(f"paid surface is not configured: {', '.join(missing())}")

    def option(price: str) -> "PaymentOption":
        return PaymentOption(
            scheme="exact",
            price=f"${price}",
            network=network(),
            pay_to=cfg.x402_pay_to,
            max_timeout_seconds=cfg.x402_timeout_seconds,
        )

    # `resource` is pinned from A2MCP_BASE_URL rather than derived from the request.
    # Behind nginx the app is reached over plain HTTP, so a derived resource advertises
    # `http://host/...` while buyers pay at `https://host/...` -- and a buyer who pays
    # for one resource and replays against another is exactly what `resource` exists to
    # prevent. It also has to equal the endpoint registered on-chain, which is a fixed
    # string, not whatever scheme a proxy hop happened to use.
    base = cfg.a2mcp_base_url.rstrip("/")

    return {
        CREATE_ROUTE: RouteConfig(
            accepts=[option(cfg.a2mcp_price_create_job)],
            resource=f"{base}/a2mcp/jobs",
            description="Start an intent-mining job from one keyword.",
            mime_type="application/json",
        ),
        STATUS_ROUTE: RouteConfig(
            accepts=[option(cfg.a2mcp_price_job_status)],
            resource=f"{base}/a2mcp/jobs/status",
            description="Job status plus the ranked links discovered so far.",
            mime_type="application/json",
        ),
    }


def resource_server():
    """A server bound to OKX's facilitator, with the EVM `exact` scheme registered.

    `exact` on X Layer settles EIP-3009 stablecoins (USD₮0 / USDG) with no on-chain
    approve, which is what keeps a 0.001 call from costing more to collect than it
    earns. Registering only this scheme is deliberate: an unregistered scheme is one
    the server cannot be talked into accepting.
    """
    from x402.http import OKXAuthConfig, OKXFacilitatorClient, OKXFacilitatorConfig
    from x402.mechanisms.evm.exact.server import ExactEvmScheme
    from x402.server import x402ResourceServer

    cfg = settings()
    facilitator = OKXFacilitatorClient(
        OKXFacilitatorConfig(
            auth=OKXAuthConfig(
                api_key=cfg.okx_api_key,
                secret_key=cfg.okx_secret_key,
                passphrase=cfg.okx_passphrase,
            ),
            # Settle before the resource is returned. Async would hand over a job whose
            # payment had not landed, and a job is not recoverable once the pipeline has
            # spent LLM and SERP credits running it.
            sync_settle=cfg.okx_sync_settle,
        )
    )
    server = x402ResourceServer(facilitator)
    server.register(network(), ExactEvmScheme())
    return server


def install(app) -> bool:
    """Gate /a2mcp/* on payment. Returns whether the gate is actually active.

    Called at import of api/main.py. When unconfigured this installs nothing and logs
    what is missing -- the routes then 503 via the guard in api/a2mcp.py rather than
    silently serving work for free.
    """
    if not configured():
        log.warning("x402 paid surface DISABLED, missing: %s. /a2mcp/* will refuse calls.",
                    ", ".join(missing()))
        return False

    from x402.http.middleware.fastapi import PaymentMiddlewareASGI

    app.add_middleware(PaymentMiddlewareASGI, routes=routes(), server=resource_server())
    log.info("x402 paid surface active on %s, paying to %s", network(), settings().x402_pay_to)
    return True
