import json

import httpx2
import pytest

from local_jev.engines.remote import ENDPOINT_HINT, RemoteJevEngine
from local_jev.errors import ModelUnavailable

SECRET = "sekrit-token"
URL = "http://127.0.0.1:8009/v1/systemone"
QUESTIONS = {"refund": {"type": "noul", "instructions": "Money back?"}}
ANSWER = {
    "model": "kev-4b",
    "answers": {"refund": {"type": "noul", "noul": 0.8}},
    "usage": {"input_tokens": 9, "output_tokens": 0},
}


def engine_with(handler, api_key=None):
    return RemoteJevEngine(URL, api_key=api_key, transport=httpx2.MockTransport(handler))


def test_posts_a_jev_payload_and_returns_the_answers():
    seen = {}

    def handler(request):
        seen.update(
            url=str(request.url),
            auth=request.headers.get("authorization"),
            body=json.loads(request.content),
        )
        return httpx2.Response(200, json=ANSWER)

    result = engine_with(handler, api_key="k").decide("refund please", QUESTIONS, "auto")
    assert seen == {
        "url": URL,
        "auth": "Bearer k",
        "body": {"model": "jev-latest", "state": "refund please", "questions": QUESTIONS},
    }
    assert result.model == "kev-4b"
    assert result.answers["refund"]["noul"] == 0.8
    assert result.usage["input_tokens"] == 9


def test_an_explicit_model_is_forwarded():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=ANSWER)

    engine_with(handler).decide("x", QUESTIONS, "kev-0.8b")
    assert seen["model"] == "kev-0.8b"


def test_validation_errors_become_value_errors():
    engine = engine_with(lambda request: httpx2.Response(422, json={"detail": "bad criteria"}))
    with pytest.raises(ValueError, match="bad criteria"):
        engine.decide("x", QUESTIONS, "auto")


@pytest.mark.parametrize(("status", "fragment"), [(401, "API key"), (500, "HTTP 500")])
def test_backend_failures_are_model_unavailable(status, fragment):
    engine = engine_with(lambda request: httpx2.Response(status, text="nope"))
    with pytest.raises(ModelUnavailable, match=fragment):
        engine.decide("x", QUESTIONS, "auto")


def test_an_unreachable_backend_is_model_unavailable():
    def handler(request):
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(ModelUnavailable, match="unreachable"):
        engine_with(handler).decide("x", QUESTIONS, "auto")


def test_an_unexpected_payload_is_model_unavailable():
    engine = engine_with(lambda request: httpx2.Response(200, json={"hello": "world"}))
    with pytest.raises(ModelUnavailable, match="unexpected payload"):
        engine.decide("x", QUESTIONS, "auto")


def test_status_and_token_estimate():
    engine = engine_with(lambda request: httpx2.Response(200, json=ANSWER))
    assert engine.status() == {"backend": "remote", "url": URL, "api_key": False}
    assert engine.count_tokens("x" * 40, "auto") == 10
    assert engine.is_loaded() is False


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param([1, 2], id="list"),
        pytest.param("lots", id="string"),
        pytest.param([], id="empty-list"),
    ],
)
def test_a_usage_that_is_not_an_object_is_model_unavailable(usage):
    engine = engine_with(lambda request: httpx2.Response(200, json={**ANSWER, "usage": usage}))
    with pytest.raises(ModelUnavailable, match="unexpected payload") as raised:
        engine.decide("x", QUESTIONS, "auto")
    assert raised.value.hint == ENDPOINT_HINT


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({key: value for key, value in ANSWER.items() if key != "usage"}, id="missing"),
        pytest.param({**ANSWER, "usage": None}, id="null"),
    ],
)
def test_a_missing_or_null_usage_becomes_an_empty_dict(payload):
    engine = engine_with(lambda request: httpx2.Response(200, json=payload))
    assert engine.decide("x", QUESTIONS, "auto").usage == {}


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(f"{SECRET}\n", id="newline"),
        pytest.param(f"{SECRET} ", id="trailing-space"),
        pytest.param(f"{SECRET}\r\nX-Injected: 1", id="header-injection"),
        pytest.param("sekrit-tökën", id="non-ascii"),
    ],
)
def test_a_key_that_cannot_go_in_a_header_is_refused_and_never_echoed(key):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx2.Response(200, json=ANSWER)

    with pytest.raises(ModelUnavailable, match="API key") as raised:
        engine_with(handler, api_key=key).decide("x", QUESTIONS, "auto")
    error = raised.value
    assert sent == []
    assert "sekrit" not in f"{error} {error.hint}"
    assert error.__cause__ is None
    assert error.__context__ is None


def test_status_and_repr_never_show_the_api_key():
    engine = engine_with(lambda request: httpx2.Response(200, json=ANSWER), api_key=SECRET)
    assert engine.status() == {"backend": "remote", "url": URL, "api_key": True}
    assert SECRET not in repr(engine)
