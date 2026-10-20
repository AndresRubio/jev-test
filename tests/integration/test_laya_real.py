"""Real-weights checks. Run `uv run local-jev pull` first, then `uv run pytest -m integration -v`."""

import asyncio
import dataclasses

import httpx2
import pytest
from mcp import Client

from local_jev.app import create_app
from local_jev.config import LayaSettings
from local_jev.engines.laya_mlx import LayaMLXEngine
from local_jev.guard import finalize
from local_jev.service import DecisionService
from tests.samples import JEV_DOC_EXAMPLE
from tests.test_app import free_port, serving

pytestmark = pytest.mark.integration

DEPT = {
    "dept": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, errors",
            "sales": "new purchases, upgrades, pricing",
        },
    }
}


@pytest.fixture(scope="module")
def engine():
    eng = LayaMLXEngine(LayaSettings())
    eng.prefetch(("english", "multilingual"))
    yield eng
    eng.unload()


def test_both_checkpoints_are_downloaded_and_verified(engine):
    assert {"english", "multilingual"} <= set(engine.status()["ready"])


def test_an_english_billing_request(engine):
    result = engine.decide("I was billed twice for March, please refund one.", DEPT, "auto")
    assert result.checkpoint == "english"
    assert result.answers["dept"]["choice"] == "billing"


def test_spanish_goes_to_the_multilingual_checkpoint(engine):
    result = engine.decide(
        "Me han cobrado dos veces este mes, quiero que me devuelvan uno.", DEPT, "auto"
    )
    assert result.checkpoint == "multilingual"
    assert result.answers["dept"]["choice"] == "billing"


def test_a_long_state_reports_truncation(engine):
    questions = {"q": {"type": "noul", "instructions": "Does the text mention a refund?"}}
    result = engine.decide("filler words " * 600 + "please refund me", questions, "english")
    response = finalize(result, questions, backend="laya-mlx", latency_ms=0.0)
    assert response["answers"]["q"]["warnings"][0].startswith("state_truncated:")


def test_more_than_20_options_are_shortlisted(engine):
    criteria = {f"topic_{i}": f"questions about unrelated topic number {i}" for i in range(29)}
    criteria["billing"] = "invoices, payments and refunds"
    questions = {
        "q": {"type": "choice", "instructions": "Which topic is this?", "criteria": criteria}
    }
    result = engine.decide("I was billed twice, please refund one invoice.", questions, "english")
    meta = result.shortlist["q"]
    assert meta["passthrough"] is False
    assert len(meta["labels"]) == 20
    assert meta["n"] == 30


@pytest.mark.anyio
async def test_rest_and_mcp_with_the_real_model(settings):
    port = free_port()
    settings = dataclasses.replace(settings, port=port, laya=LayaSettings(prefetch=("english",)))
    app = create_app(settings, service=DecisionService(settings, LayaMLXEngine(settings.laya)))
    with serving(app, port) as base:
        async with httpx2.AsyncClient(timeout=60) as http:
            for _ in range(120):
                status = (await http.get(f"{base}/v1/status")).json()
                if "english" in status["engine"]["ready"]:
                    break
                await asyncio.sleep(0.5)
            rest = await http.post(f"{base}/v1/systemone", json=JEV_DOC_EXAMPLE)
        async with Client(f"{base}/mcp") as client:
            result = await client.call_tool(
                "decide", {"state": "I was billed twice, please refund one.", "questions": DEPT}
            )
    assert rest.status_code == 200
    assert set(rest.json()["answers"]) == {"dept", "refund", "urgency"}
    assert result.is_error is False
    assert result.structured_content["answers"]["dept"]["choice"] == "billing"
