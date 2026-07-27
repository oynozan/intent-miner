"""The paid surface: what a buyer is actually charged, and what we do when we cannot
verify a payment.

Minting the 402 challenge is the only part of x402 this service owns end to end --
settlement belongs to a facilitator -- so these tests pin the two failures that would
cost real money without looking like errors: charging an amount that is not the listed
price, and serving paid work that was never proven paid.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from core import x402
from core.config import settings


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully-configured seller, except the facilitator (set per-test)."""
    monkeypatch.setenv("A2MCP_BASE_URL", "https://miner.example.com")
    monkeypatch.setenv("A2MCP_PRICE_CREATE_JOB", "0.05")
    monkeypatch.setenv("A2MCP_PRICE_JOB_STATUS", "0.001")
    monkeypatch.setenv("X402_PAY_TO", "0x25a39c21b29b80df5b7fc59286aa7dc6f10f9c13")
    monkeypatch.setenv("X402_ASSET", "0x1e4a5963abfd975d8c9021ce480b42188849d41d")
    monkeypatch.setenv("X402_CHAIN_ID", "196")
    monkeypatch.setenv("X402_ASSET_DECIMALS", "6")
    monkeypatch.setenv("X402_FACILITATOR_VERIFY_URL", "")
    monkeypatch.setenv("X402_FACILITATOR_SETTLE_URL", "")
    settings.cache_clear()
    yield
    settings.cache_clear()


@pytest.fixture
def client(configured: None) -> TestClient:
    from api.main import app

    return TestClient(app)


def _challenge_from(response) -> dict:
    """Read the challenge back the way a buyer does -- out of the header, not the body."""
    return json.loads(base64.b64decode(response.headers[x402.PAYMENT_REQUIRED_HEADER]))


# --- amounts -----------------------------------------------------------------------

def test_price_converts_exactly_at_six_decimals() -> None:
    """The whole reason this uses Decimal. `int(0.05 * 10**6)` is 49999 -- a silent
    one-unit underprice on every single call."""
    assert x402.to_base_units("0.05", 6) == "50000"
    assert x402.to_base_units("0.001", 6) == "1000"
    assert x402.to_base_units("0", 6) == "0"


def test_a_price_the_token_cannot_express_is_rejected_not_rounded() -> None:
    """Truncating here would charge a different price than the one listed on-chain."""
    with pytest.raises(x402.Misconfigured):
        x402.to_base_units("0.0000001", 6)


def test_a_challenge_without_a_payee_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty payTo mints a challenge that collects for nobody -- the buyer pays and
    the money goes nowhere. Refuse to build it rather than emit it."""
    monkeypatch.setenv("A2MCP_BASE_URL", "https://miner.example.com")
    monkeypatch.setenv("X402_PAY_TO", "")
    monkeypatch.setenv("X402_ASSET", "0xabc")
    settings.cache_clear()
    with pytest.raises(x402.Misconfigured):
        x402.challenge("/a2mcp/jobs", "0.05", {"type": "http", "method": "POST"}, "d")
    settings.cache_clear()


# --- the challenge a buyer reads ---------------------------------------------------

def test_creating_a_job_unpaid_returns_402_priced_at_the_listed_amount(client: TestClient) -> None:
    response = client.post("/a2mcp/jobs", json={"keyword": "video background removal"})
    assert response.status_code == 402

    accepts = _challenge_from(response)["accepts"][0]
    assert accepts["amount"] == "50000"          # 0.05 USDT at 6 dp
    assert accepts["payTo"] == "0x25a39c21b29b80df5b7fc59286aa7dc6f10f9c13"
    assert accepts["network"] == "eip155:196"
    assert accepts["scheme"] == "exact"


def test_job_status_is_priced_fifty_times_cheaper_than_creating_one(client: TestClient) -> None:
    """The two services are deliberately not the same price -- polling must stay cheap
    enough that a buyer is not punished for watching a job they already paid for."""
    create = _challenge_from(client.post("/a2mcp/jobs", json={"keyword": "background removal"}))
    status = _challenge_from(client.get("/a2mcp/jobs/status", params={"job_id": "x"}))

    assert int(create["accepts"][0]["amount"]) == 50 * int(status["accepts"][0]["amount"])


def test_the_create_challenge_declares_post(client: TestClient) -> None:
    """`payment quote` probes with GET unless the challenge says otherwise. Omit this
    and a buyer's very first probe gets 405 and reads as an unreachable endpoint."""
    accepts = _challenge_from(client.post("/a2mcp/jobs", json={"keyword": "background removal"}))["accepts"][0]
    assert accepts["outputSchema"]["input"]["method"] == "POST"
    assert "keyword" in accepts["outputSchema"]["input"]["body"]["required"]


def test_the_status_challenge_declares_job_id_as_a_query_param(client: TestClient) -> None:
    """A registered endpoint URL is static, so the job id has to ride in the query
    string -- a path segment could not be expressed in the on-chain listing."""
    accepts = _challenge_from(client.get("/a2mcp/jobs/status", params={"job_id": "x"}))["accepts"][0]
    schema = accepts["outputSchema"]["input"]
    assert schema["method"] == "GET"
    assert "job_id" in schema["queryParams"]["required"]


def test_the_challenge_is_in_the_body_too(client: TestClient) -> None:
    """v1 buyers read the body; humans curling the endpoint should see a price without
    base64-decoding a header."""
    response = client.post("/a2mcp/jobs", json={"keyword": "background removal"})
    assert response.json()["x402Version"] == 2
    assert response.json()["accepts"] == _challenge_from(response)["accepts"]


# --- fail closed -------------------------------------------------------------------

def test_a_payment_we_cannot_verify_never_becomes_free_work(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this whole module is shaped around. With no facilitator configured a
    presented payment is unverifiable, so the call must fail -- NOT quietly succeed and
    queue a run that nobody paid for."""
    from pipeline import repo

    def _boom(*args, **kwargs):
        raise AssertionError("a run was created for an unverified payment")

    monkeypatch.setattr(repo, "create_run", _boom)

    response = client.post(
        "/a2mcp/jobs",
        json={"keyword": "background removal"},
        headers={x402.PAYMENT_SIGNATURE_HEADER: base64.b64encode(b'{"scheme":"exact"}').decode()},
    )
    assert response.status_code == 500


def test_an_unverifiable_payment_is_not_reported_as_402(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """402 means "pay me". Telling a buyer who already paid to pay again, because *we*
    have no facilitator, is the wrong side to fail on."""
    from pipeline import repo

    monkeypatch.setattr(repo, "create_run", lambda *a, **k: "never")

    response = client.post(
        "/a2mcp/jobs",
        json={"keyword": "background removal"},
        headers={x402.PAYMENT_SIGNATURE_HEADER: base64.b64encode(b"{}").decode()},
    )
    assert response.status_code != 402
