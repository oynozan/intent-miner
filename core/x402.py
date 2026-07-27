"""Seller side of x402 — mint the 402 challenge, verify the paid replay.

Everything the `onchainos payment` CLI does is the *buyer* half: probe a 402, sign it,
replay it. This module is the half that has to exist on our side for that to work at
all, and it splits cleanly in two:

**Minting the challenge is fully ours.** A GET/POST with no payment header returns 402
carrying `PAYMENT-REQUIRED` (base64 JSON, x402 v2). That is what `payment quote <url>`
reads to price the call, and what OKX.AI's listing crawls to show the service. It needs
no third party, so it is implemented and tested here.

**Settling is not ours and cannot be faked.** Verifying that a presented signature is
real, unspent, and worth the amount claimed requires the facilitator that holds the
on-chain view. There is no seller-side `verify` in the CLI (`onchainos payment` exposes
pay / quote / decode-receipt / pay-local / a2a-pay / charge / session / subscription —
all buyer-side), so the contract below follows the public x402 facilitator interface
(`POST /verify`, `POST /settle`, with `paymentPayload` + `paymentRequirements`).
**Point `X402_FACILITATOR_VERIFY_URL` at OKX's facilitator and confirm those field names
before taking real money.**

Until it is configured, `require_payment` refuses every paid call. That is deliberate:
the alternative failure mode is a service that hands out 0.05-USDT work for free and
looks perfectly healthy while doing it. A dev bypass flag is *not* provided on purpose —
it would be one env var away from doing exactly that in production.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from core.config import settings

log = logging.getLogger(__name__)

X402_VERSION = 2

# Externally defined by the protocol — never rename these.
PAYMENT_REQUIRED_HEADER = "PAYMENT-REQUIRED"
PAYMENT_SIGNATURE_HEADER = "PAYMENT-SIGNATURE"
PAYMENT_RESPONSE_HEADER = "PAYMENT-RESPONSE"

SCHEME = "exact"


class PaymentRequired(Exception):
    """No usable payment was presented. Carries the challenge to return as a 402."""

    def __init__(self, challenge: dict[str, Any], reason: str | None = None) -> None:
        self.challenge = challenge
        self.reason = reason
        super().__init__(reason or "payment required")


class Misconfigured(Exception):
    """The seller side is not set up. A 500 — never a 402, which would tell the buyer
    to pay for a service that cannot collect."""


def to_base_units(amount: str, decimals: int) -> str:
    """``"0.05"``, 6 -> ``"50000"``.

    Decimal rather than float throughout: 0.05 has no exact binary representation, and
    ``int(0.05 * 10**6)`` is 49999 on a bad day. Rejects more precision than the token
    can hold instead of silently truncating it -- a listed price the contract cannot
    express is a configuration bug, not something to round.
    """
    try:
        units = Decimal(amount) * (Decimal(10) ** decimals)
    except InvalidOperation as exc:
        raise Misconfigured(f"price {amount!r} is not a decimal number") from exc
    if units != units.to_integral_value():
        raise Misconfigured(f"price {amount!r} needs more than {decimals} decimals")
    if units < 0:
        raise Misconfigured(f"price {amount!r} is negative")
    return str(int(units))


def _b64(payload: dict[str, Any]) -> str:
    return base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


def _unb64(value: str) -> dict[str, Any]:
    """Decode a base64 / base64url header value. Buyers emit both."""
    padded = value.strip() + "=" * (-len(value.strip()) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded) if ("-" in padded or "_" in padded) else base64.b64decode(padded)
        decoded = json.loads(raw)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"payment header is not base64 JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("payment header did not decode to an object")
    return decoded


def requirements(path: str, price: str, input_schema: dict[str, Any], description: str) -> dict[str, Any]:
    """One `accepts[]` entry — the terms for a single call to ``path``."""
    cfg = settings()
    if not cfg.x402_pay_to:
        raise Misconfigured("X402_PAY_TO is not set; a challenge would collect for nobody")
    if not cfg.x402_asset:
        raise Misconfigured("X402_ASSET is not set; no token to charge in")
    if not cfg.a2mcp_base_url:
        raise Misconfigured("A2MCP_BASE_URL is not set; the challenge needs its public resource URL")

    resource = f"{cfg.a2mcp_base_url.rstrip('/')}{path}"
    return {
        "scheme": SCHEME,
        "network": f"eip155:{cfg.x402_chain_id}",
        "asset": cfg.x402_asset,
        "amount": to_base_units(price, cfg.x402_asset_decimals),
        # v1 buyers read maxAmountRequired; `exact` makes them the same number.
        "maxAmountRequired": to_base_units(price, cfg.x402_asset_decimals),
        "payTo": cfg.x402_pay_to,
        "resource": resource,
        "description": description,
        "mimeType": "application/json",
        "maxTimeoutSeconds": cfg.x402_timeout_seconds,
        # The buyer's signer builds its EIP-712 domain from these. A wrong `name` yields
        # a signature the facilitator rejects without saying why.
        "extra": {"name": cfg.x402_asset_name, "version": cfg.x402_asset_version},
        # Declares how the PAID replay must be shaped. `payment quote` probes with GET
        # unless this says otherwise, so a POST-only endpoint that omits it answers 405
        # and the buyer sees `endpoint_unreachable` instead of a price.
        "outputSchema": {"input": input_schema},
    }


def challenge(path: str, price: str, input_schema: dict[str, Any], description: str) -> dict[str, Any]:
    accepts = requirements(path, price, input_schema, description)
    return {
        "x402Version": X402_VERSION,
        "resource": {
            "url": accepts["resource"],
            "method": input_schema.get("method", "GET"),
            "description": description,
        },
        "accepts": [accepts],
    }


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    cfg = settings()
    headers = {"Authorization": f"Bearer {cfg.x402_facilitator_api_key}"} if cfg.x402_facilitator_api_key else {}
    response = httpx.post(url, json=payload, headers=headers, timeout=20.0)
    response.raise_for_status()
    return response.json()


def require_payment(
    header_value: str | None,
    *,
    path: str,
    price: str,
    input_schema: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    """Gate one call. Returns the settlement receipt, or raises ``PaymentRequired``.

    Verify and settle are two calls on purpose. Verify says the authorization is real
    and funded; settle moves it. Doing the work between them would mean settling after
    a run has already been queued -- and a failed settle at that point is work given
    away, not a refused request.
    """
    reqs = requirements(path, price, input_schema, description)
    full = {"x402Version": X402_VERSION, "resource": {"url": reqs["resource"]}, "accepts": [reqs]}

    if not header_value:
        raise PaymentRequired(challenge(path, price, input_schema, description))

    cfg = settings()
    if not cfg.x402_facilitator_verify_url or not cfg.x402_facilitator_settle_url:
        # Fail closed. Never serve the work: an unverifiable payment is an unpaid one.
        raise Misconfigured(
            "X402_FACILITATOR_VERIFY_URL / _SETTLE_URL are unset, so a presented payment "
            "cannot be verified. Refusing to serve paid work unverified."
        )

    try:
        payload = _unb64(header_value)
    except ValueError as exc:
        raise PaymentRequired(full, reason=str(exc)) from exc

    body = {"x402Version": X402_VERSION, "paymentPayload": payload, "paymentRequirements": reqs}

    try:
        verified = _post(cfg.x402_facilitator_verify_url, body)
    except httpx.HTTPError as exc:
        # The facilitator being unreachable is our outage, not the buyer's fault --
        # 402 would tell them to pay again for a call they already paid for.
        raise Misconfigured(f"facilitator verify failed: {exc}") from exc

    if not verified.get("isValid"):
        raise PaymentRequired(full, reason=verified.get("invalidReason") or "payment rejected")

    try:
        settled = _post(cfg.x402_facilitator_settle_url, body)
    except httpx.HTTPError as exc:
        raise Misconfigured(f"facilitator settle failed: {exc}") from exc

    if not settled.get("success"):
        raise PaymentRequired(full, reason=settled.get("errorReason") or "settlement failed")

    log.info("x402 settled %s for %s units: tx=%s", path, reqs["amount"], settled.get("transaction"))
    return settled


def challenge_header(challenge_body: dict[str, Any]) -> str:
    """The `PAYMENT-REQUIRED` value — base64 of the challenge the 402 body also carries."""
    return _b64(challenge_body)


def receipt_header(settled: dict[str, Any]) -> str:
    """The `PAYMENT-RESPONSE` value the buyer decodes with ``payment decode-receipt``."""
    return _b64(
        {
            "success": True,
            "status": "success",
            "transaction": settled.get("transaction"),
            "network": settled.get("network"),
            "payer": settled.get("payer"),
        }
    )
