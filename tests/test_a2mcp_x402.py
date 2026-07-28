"""The paid surface: that it charges the listed price, and that it never works for free.

Protocol mechanics -- challenge format, signing, settlement -- belong to
``okxweb3-app-x402`` and are its tests to write, not ours. What is ours is the wiring
around it, and the two ways that wiring can lose money without erroring:

1. charging an amount that is not the price registered on-chain, and
2. serving the work when the payment gate was never installed.

Both are tested here. Neither shows up as a failure in production.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core import x402
from core.config import settings

CREDS = {
    "OKX_API_KEY": "test-key",
    "OKX_SECRET_KEY": "test-secret",
    "OKX_PASSPHRASE": "test-passphrase",
    "X402_PAY_TO": "0x25a39c21b29b80df5b7fc59286aa7dc6f10f9c13",
    "A2MCP_BASE_URL": "https://miner.example.com",
}


@pytest.fixture
def unconfigured(monkeypatch: pytest.MonkeyPatch):
    """No credentials -- the state a fresh deployment is in before .env is filled."""
    for key in CREDS:
        monkeypatch.delenv(key, raising=False)
    settings.cache_clear()
    yield
    settings.cache_clear()


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch):
    for key, value in CREDS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("A2MCP_PRICE_CREATE_JOB", "0.05")
    monkeypatch.setenv("A2MCP_PRICE_JOB_STATUS", "0.001")
    monkeypatch.setenv("X402_CHAIN_ID", "196")
    settings.cache_clear()
    yield
    settings.cache_clear()


# --- the price actually charged --------------------------------------------------

def test_routes_charge_the_listed_prices(configured) -> None:
    """The listing on-chain says 0.05 and 0.001. If these drift, the marketplace
    advertises one price while the endpoint quotes another and nothing errors."""
    routes = x402.routes()

    create = routes[x402.CREATE_ROUTE].accepts[0]
    status = routes[x402.STATUS_ROUTE].accepts[0]

    assert create.price == "$0.05"
    assert status.price == "$0.001"


def test_status_stays_far_cheaper_than_creating(configured) -> None:
    """Polling a job you already paid for must not cost like starting another one."""
    routes = x402.routes()
    create = float(routes[x402.CREATE_ROUTE].accepts[0].price.lstrip("$"))
    status = float(routes[x402.STATUS_ROUTE].accepts[0].price.lstrip("$"))
    assert create == pytest.approx(50 * status)


def test_both_routes_settle_on_x_layer_to_the_configured_payee(configured) -> None:
    routes = x402.routes()
    for route in (x402.CREATE_ROUTE, x402.STATUS_ROUTE):
        option = routes[route].accepts[0]
        assert option.network == "eip155:196"
        assert option.pay_to == CREDS["X402_PAY_TO"]
        assert option.scheme == "exact"


def test_resource_is_the_registered_https_url_not_the_proxy_hop(configured) -> None:
    """Behind nginx the app is reached over plain HTTP, so a resource derived from the
    request advertises http://host/... while buyers pay at https://host/... -- and it
    must also equal the endpoint registered on-chain, which is a fixed string."""
    routes = x402.routes()
    assert routes[x402.CREATE_ROUTE].resource == "https://miner.example.com/a2mcp/jobs"
    assert routes[x402.STATUS_ROUTE].resource == "https://miner.example.com/a2mcp/jobs/status"


def test_route_keys_match_the_real_paths(configured) -> None:
    """The SDK matches on these exact strings. A renamed path silently un-gates the
    route -- it does not 404, it becomes free."""
    from api.a2mcp import router

    paths = {f"{method} {r.path}" for r in router.routes for method in r.methods if method != "HEAD"}
    assert x402.CREATE_ROUTE in paths
    assert x402.STATUS_ROUTE in paths


# --- never free ------------------------------------------------------------------

def test_unconfigured_refuses_instead_of_building_routes(unconfigured) -> None:
    assert not x402.configured()
    with pytest.raises(x402.Misconfigured):
        x402.routes()


def test_missing_names_every_absent_setting(unconfigured) -> None:
    """A 503 that does not say what is missing costs someone an afternoon."""
    assert set(x402.missing()) == set(CREDS)


def test_a_payee_alone_is_not_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credentials without a payee, or a payee without credentials, are both unable to
    collect -- neither may be treated as configured."""
    monkeypatch.setenv("X402_PAY_TO", CREDS["X402_PAY_TO"])
    for key in ("OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"):
        monkeypatch.delenv(key, raising=False)
    settings.cache_clear()
    assert not x402.configured()
    settings.cache_clear()


def test_keyword_is_accepted_from_query_or_body(configured, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a real paid replay arrived with an empty body and this route answered
    422 -- a buyer who had parted with money getting an error back. The challenge cannot
    declare which carrier to use (the SDK's v2 RouteConfig has no output_schema), so both
    must work.
    """
    from api import a2mcp

    seen: list[tuple] = []
    monkeypatch.setattr(a2mcp.repo, "create_run", lambda kw, icp: seen.append((kw, icp)) or "run-1")
    monkeypatch.setattr(a2mcp.x402, "configured", lambda: True)

    from types import SimpleNamespace
    monkeypatch.setitem(
        __import__("sys").modules, "pipeline.actors",
        SimpleNamespace(start_run=SimpleNamespace(send=lambda _: None)),
    )

    assert a2mcp.create_job(body=None, keyword="from query", icp=None).job_id == "run-1"
    assert a2mcp.create_job(body=a2mcp.CreateJob(keyword="from body"), keyword=None, icp=None).job_id == "run-1"
    assert [kw for kw, _ in seen] == ["from query", "from body"]


def test_a_keyword_absent_from_both_is_a_400_not_a_422(configured, monkeypatch: pytest.MonkeyPatch) -> None:
    """400 with a usable message beats pydantic's 422 for a caller that just paid."""
    from fastapi import HTTPException

    from api import a2mcp

    monkeypatch.setattr(a2mcp.x402, "configured", lambda: True)
    with pytest.raises(HTTPException) as exc:
        a2mcp.create_job(body=None, keyword=None, icp=None)
    assert exc.value.status_code == 400
    assert "keyword" in exc.value.detail


def test_a_malformed_job_id_is_not_found_rather_than_a_server_error() -> None:
    """Regression: the listing was rejected for "a server error". `runs.id` is a uuid
    column, so a job_id that is not a uuid raised inside Postgres before the handler's
    not_found branch could run, and a buyer who had already paid got a 500. Reviewers
    probe with placeholder ids, so this is the first thing they hit.

    No database is involved on purpose: reaching one at all would mean the guard did not
    fire."""
    from pipeline import repo

    for junk in ("test", "123", "", "not-a-uuid", "402966f1-52f6-42b1-8999"):
        assert repo.get_run(junk) is None


def test_unconfigured_serves_no_paid_work(unconfigured, monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure this whole arrangement exists to prevent. With no payment gate
    installed, a create call must NOT reach the pipeline -- an unconfigured deployment
    would otherwise run jobs, and spend real LLM and SERP credits, for free."""
    from pipeline import repo

    monkeypatch.setattr(repo, "create_run", lambda *a, **k: pytest.fail("ran a job for free"))

    from api.main import app

    response = TestClient(app).post("/a2mcp/jobs", json={"keyword": "background removal"})
    assert response.status_code == 503


def test_unconfigured_is_not_reported_as_402(unconfigured) -> None:
    """402 means "pay me". Inviting payment to a service with no payee configured
    would take money nobody can settle."""
    from api.main import app

    response = TestClient(app).get("/a2mcp/jobs/status", params={"job_id": "x"})
    assert response.status_code == 503
    assert response.status_code != 402
