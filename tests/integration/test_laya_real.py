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
UPGRADE = "I would like to upgrade to the premium plan. What does it cost?"


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


@pytest.mark.xfail(
    strict=True,
    reason="the english checkpoint answers billing 0.6868 (sales 0.255, technical 0.0582)",
)
def test_an_upgrade_question_goes_to_sales(engine):
    result = engine.decide(UPGRADE, DEPT, "auto")
    assert result.answers["dept"]["choice"] == "sales"


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
    items = [
        "I was billed twice, please refund one.",
        "The app crashes with an error every time I open it.",
        UPGRADE,
    ]
    with serving(app, port) as base:
        async with httpx2.AsyncClient(timeout=60) as http:
            for _ in range(120):
                status = (await http.get(f"{base}/v1/status")).json()
                if "english" in status["engine"]["ready"]:
                    break
                await asyncio.sleep(0.5)
            rest = await http.post(f"{base}/v1/systemone", json=JEV_DOC_EXAMPLE)
        async with Client(f"{base}/mcp") as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            result = await client.call_tool(
                "decide", {"state": "I was billed twice, please refund one.", "questions": DEPT}
            )
            batch = await client.call_tool("decide_batch", {"questions": DEPT, "items": items})
            mcp_status = await client.call_tool("status", {})
    assert rest.status_code == 200
    body = rest.json()
    assert set(body) == {"model", "answers", "usage", "local"}
    answers = body["answers"]
    assert set(answers) == {"dept", "refund", "urgency"}
    assert {"type", "choice", "probabilities", "confidence"} <= set(answers["dept"])
    assert answers["dept"]["choice"] == "billing"
    assert {"type", "noul"} <= set(answers["refund"])
    assert answers["urgency"]["legend"] == {"0": "not urgent", "1": "soon", "2": "blocking"}
    assert all("answer_confidence" in answer for answer in answers.values())
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert body["local"]["warnings"] == []
    assert names == {"decide", "decide_batch", "status"}
    assert result.is_error is False
    assert result.structured_content["answers"]["dept"]["choice"] == "billing"
    assert batch.is_error is False
    assert batch.structured_content["count"] == 3
    choices = [match["answers"]["dept"]["choice"] for match in batch.structured_content["matches"]]
    assert choices[:2] == ["billing", "technical"]
    # The upgrade item is a known miss; test_an_upgrade_question_goes_to_sales pins its label.
    assert choices[2] in DEPT["dept"]["criteria"]
    assert mcp_status.is_error is False
    assert mcp_status.structured_content["backend"] == "laya-mlx"
    assert "english" in mcp_status.structured_content["engine"]["loaded"]
