import dataclasses
import json
import socket
import threading
import time
from contextlib import asynccontextmanager, contextmanager

import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecurityMiddleware
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.testclient import TestClient

from local_jev.api_rest import MAX_BODY_BYTES, build_rest_routes
from local_jev.app import build_engine, create_app, mcp_transport_security
from local_jev.config import RemoteSettings
from local_jev.engines.remote import RemoteJevEngine
from local_jev.security import is_loopback_origin, loopback_hosts
from local_jev.service import DecisionService
from tests.fakes import FakeEngine
from tests.samples import JEV_DOC_EXAMPLE

DEPT = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]}}
JSON = {"content-type": "application/json"}


def client_for(settings, *, host=None, engine=None, **options):
    service = DecisionService(settings, engine or FakeEngine())
    base = f"http://{host or f'127.0.0.1:{settings.port}'}"
    return TestClient(create_app(settings, service=service), base_url=base, **options)


def test_health(settings):
    with client_for(settings) as client:
        assert client.get("/health").json() == {"ok": True}


def test_the_documented_jev_request_gets_a_jev_shaped_answer(settings):
    with client_for(settings) as client:
        response = client.post("/v1/systemone", json=JEV_DOC_EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "answers", "usage", "local"}
    assert set(body["answers"]) == {"dept", "refund", "urgency"}
    assert {"type", "choice", "probabilities", "confidence"} <= set(body["answers"]["dept"])
    assert {"type", "noul"} <= set(body["answers"]["refund"])
    assert body["answers"]["urgency"]["legend"] == {"0": "not urgent", "1": "soon", "2": "blocking"}
    assert body["usage"] == {"input_tokens": 30, "output_tokens": 0}


def test_validation_errors_use_the_error_body(settings):
    with client_for(settings) as client:
        response = client.post("/v1/systemone", json={"state": "x"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["hint"]


def test_invalid_json_is_422(settings):
    with client_for(settings) as client:
        response = client.post(
            "/v1/systemone", content=b"{not json", headers={"content-type": "application/json"}
        )
    assert response.status_code == 422


def test_an_oversized_body_is_413(settings):
    with client_for(settings) as client:
        response = client.post("/v1/systemone", content=b"x" * (5 * 1024 * 1024 + 1))
    assert response.status_code == 413


def test_status_and_batch_routes(settings, tmp_path):
    path = tmp_path / "items.txt"
    path.write_text("billing\ntech\n")
    with client_for(settings) as client:
        assert client.get("/v1/status").json()["backend"] == "fake"
        response = client.post("/v1/batch", json={"questions": DEPT, "path": str(path)})
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_requests_for_another_host_are_refused(settings):
    with client_for(settings, host="evil.example:7711") as client:
        assert client.get("/health").status_code == 421


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def serving(app, port):
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("uvicorn did not stop")


@pytest.mark.anyio
async def test_mcp_is_served_over_http_at_slash_mcp(settings):
    port = free_port()
    settings = dataclasses.replace(settings, port=port)
    app = create_app(settings, service=DecisionService(settings, FakeEngine()))
    with serving(app, port) as base:
        async with Client(f"{base}/mcp") as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            result = await client.call_tool("decide", {"state": "tech outage", "questions": DEPT})
    assert names == {"decide", "decide_batch", "status"}
    assert result.structured_content["answers"]["dept"]["choice"] == "tech"


# Bodies that crash json.loads, responses that carry lone surrogates, the API key from the
# environment, the lifespan, and the Host and Origin checks of the MCP SDK.


def test_a_body_with_a_huge_integer_is_422(settings):
    # json.loads raises a bare ValueError for an integer with more than 4,300 digits.
    body = b'{"state": {"n": ' + b"9" * 5000 + b'}, "questions": {}}'
    with client_for(settings, raise_server_exceptions=False) as client:
        response = client.post("/v1/systemone", content=body, headers=JSON)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["hint"]


def test_a_deeply_nested_body_is_422(settings):
    # json.loads raises RecursionError for about 20 KB of nested arrays.
    body = b"[" * 10_000 + b"]" * 10_000
    with client_for(settings, raise_server_exceptions=False) as client:
        response = client.post("/v1/systemone", content=body, headers=JSON)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["hint"]


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/batch"])
@pytest.mark.parametrize("body", [b"[]", b'"text"', b"1", b"null", b"true"])
def test_a_body_that_is_not_an_object_is_422(settings, path, body):
    with client_for(settings, raise_server_exceptions=False) as client:
        response = client.post(path, content=body, headers=JSON)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


NON_FINITE = ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"]


@pytest.mark.parametrize("literal", NON_FINITE)
def test_a_non_finite_number_in_a_decide_body_is_422(settings, literal):
    # json.loads reads NaN and the infinities, and turns a literal that overflows, 1e400, into inf.
    engine = FakeEngine()
    body = f'{{"state": {{"n": {literal}}}, "questions": {json.dumps(DEPT)}}}'.encode()
    with client_for(settings, engine=engine, raise_server_exceptions=False) as client:
        response = client.post("/v1/systemone", content=body, headers=JSON)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["message"].startswith("the body is not valid JSON: ")
    assert literal in error["message"]
    assert error["hint"] == "send a JSON object"
    assert engine.calls == []


@pytest.mark.parametrize("literal", NON_FINITE)
def test_a_non_finite_number_as_a_batch_item_id_is_422(settings, literal):
    # The batch would run and echo the id in matches, and the response could not be written.
    engine = FakeEngine()
    items = f'[{{"id": {literal}, "text": "tech"}}]'
    body = f'{{"questions": {json.dumps(DEPT)}, "items": {items}}}'.encode()
    with client_for(settings, engine=engine, raise_server_exceptions=False) as client:
        response = client.post("/v1/batch", content=body, headers=JSON)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["message"].startswith("the body is not valid JSON: ")
    assert literal in error["message"]
    assert error["hint"] == "send a JSON object"
    assert engine.calls == []


def test_a_long_out_of_range_number_is_not_echoed_in_full(settings):
    body = (
        b'{"state": {"n": 1' + b"0" * 400 + b'.5}, "questions": ' + json.dumps(DEPT).encode() + b"}"
    )
    with client_for(settings, raise_server_exceptions=False) as client:
        response = client.post("/v1/systemone", content=body, headers=JSON)
    assert response.status_code == 422
    assert len(response.json()["error"]["message"]) < 200


def test_an_oversized_body_without_a_content_length_is_413(settings):
    def chunks():
        for _ in range(6):
            yield b"x" * (1024 * 1024)

    with client_for(settings) as client:
        response = client.post("/v1/systemone", content=chunks())
    assert "content-length" not in response.request.headers  # sent as Transfer-Encoding: chunked
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "too_large"


def padded(request, size):
    """The request as JSON, then spaces, in exactly `size` bytes."""
    raw = json.dumps(request).encode()
    return raw + b" " * (size - len(raw))


def in_chunks(body, size=1024 * 1024):
    for start in range(0, len(body), size):
        yield body[start : start + size]


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(("extra", "status"), [(0, 200), (1, 413)])
def test_rest_serves_a_body_of_exactly_the_limit_and_refuses_one_byte_more(
    settings, chunked, extra, status
):
    body = padded(JEV_DOC_EXAMPLE, MAX_BODY_BYTES + extra)
    with client_for(settings, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/systemone", content=in_chunks(body) if chunked else body, headers=JSON
        )
    assert ("content-length" in response.request.headers) is not chunked
    assert response.status_code == status


@pytest.mark.anyio
async def test_a_chunked_body_over_the_limit_is_refused_before_it_is_read_in_full(settings):
    service = DecisionService(settings, FakeEngine())
    app = Starlette(routes=build_rest_routes(service))
    pulled = 0

    async def megabytes():
        nonlocal pulled
        while True:
            pulled += 1
            assert pulled <= 10, "the whole body was read"
            yield b"x" * (1024 * 1024)

    transport = httpx2.ASGITransport(app=app)
    client = httpx2.AsyncClient(transport=transport, base_url="http://127.0.0.1:7711")
    try:
        response = await client.post("/v1/systemone", content=megabytes(), headers=JSON)
    finally:
        await client.aclose()
        await service.stop()
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "too_large"
    assert (
        error["message"]
        == f"the body is over {MAX_BODY_BYTES} bytes; the limit is {MAX_BODY_BYTES}"
    )
    assert error["hint"] == "use decide_batch with a path"
    assert pulled == 6  # the sixth megabyte is the first to pass 5 MiB


def test_a_lone_surrogate_in_an_error_message_still_gives_a_json_error(settings, tmp_path):
    # The JSON escape \ud800 is a lone surrogate once parsed, and the "no such file" message repeats it.
    body = f'{{"questions": {json.dumps(DEPT)}, "path": "{tmp_path}/\\ud800.jsonl"}}'.encode()
    with client_for(settings, raise_server_exceptions=False) as client:
        response = client.post("/v1/batch", content=body, headers=JSON)
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert "no such file" in error["message"]
    assert "\ud800" in error["message"]  # parsed back from the escape in the response
    assert response.content.isascii()


def test_a_lone_surrogate_in_a_batch_item_never_gives_a_500(settings):
    body = f'{{"questions": {json.dumps(DEPT)}, "items": ["billing \\ud800", "tech"]}}'.encode()
    with client_for(settings, raise_server_exceptions=False) as client:
        response = client.post("/v1/batch", content=body, headers=JSON)
    assert response.status_code in (200, 422)
    assert response.headers["content-type"] == "application/json"
    assert response.content.isascii()
    response.json()


def test_responses_are_ascii_json_with_the_text_escaped(settings, tmp_path):
    body = json.dumps({"questions": DEPT, "path": f"{tmp_path}/café.jsonl"}).encode()
    with client_for(settings, raise_server_exceptions=False) as client:
        response = client.post("/v1/batch", content=body, headers=JSON)
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    assert response.content.isascii()
    assert "caf\\u00e9.jsonl" in response.text
    assert "café.jsonl" in response.json()["error"]["message"]


def remote_settings(settings, api_key_env="LOCAL_JEV_TEST_API_KEY"):
    remote = RemoteSettings(url="http://127.0.0.1:9/v1/systemone", api_key_env=api_key_env)
    return dataclasses.replace(settings, backend="remote", remote=remote)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("secret", "secret"),
        ("secret\n", "secret"),
        ("  secret \t\r\n", "secret"),
        ("", None),
        (" \n", None),
    ],
)
def test_build_engine_strips_the_remote_api_key(settings, monkeypatch, value, expected):
    monkeypatch.setenv("LOCAL_JEV_TEST_API_KEY", value)
    engine = build_engine(remote_settings(settings))
    assert isinstance(engine, RemoteJevEngine)
    assert engine._api_key == expected
    assert engine.status()["api_key"] is (expected is not None)


def test_build_engine_sends_no_key_when_the_variable_is_unset_or_unnamed(settings, monkeypatch):
    monkeypatch.delenv("LOCAL_JEV_TEST_API_KEY", raising=False)
    assert build_engine(remote_settings(settings))._api_key is None
    monkeypatch.setenv("LOCAL_JEV_TEST_API_KEY", "secret")
    assert build_engine(remote_settings(settings, api_key_env=""))._api_key is None


def test_build_engine_defaults_to_laya_mlx(settings, monkeypatch):
    seen = []

    class StubLaya:
        def __init__(self, laya_settings):
            seen.append(laya_settings)

    monkeypatch.setattr("local_jev.engines.laya_mlx.LayaMLXEngine", StubLaya)
    assert isinstance(build_engine(settings), StubLaya)
    assert seen == [settings.laya]


def test_create_app_builds_its_own_service_from_the_settings(settings, monkeypatch):
    seen = []

    def stub_engine(given):
        seen.append(given)
        return FakeEngine()

    monkeypatch.setattr("local_jev.app.build_engine", stub_engine)
    with TestClient(create_app(settings), base_url=f"http://127.0.0.1:{settings.port}") as client:
        assert client.get("/v1/status").json()["backend"] == "fake"
    assert seen == [settings]


def test_the_token_guards_rest_and_mcp_but_not_health(settings):
    settings = dataclasses.replace(settings, token="s3cret")
    with client_for(settings) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/status").status_code == 401
        assert client.post("/mcp", json={}).status_code == 401
        assert (
            client.get("/v1/status", headers={"authorization": "Bearer s3cret"}).status_code == 200
        )


def test_the_lifespan_starts_the_service_then_mcp_and_stops_them_in_reverse(settings, monkeypatch):
    events = []

    class RecordingService(DecisionService):
        async def start(self):
            events.append("service started")
            await super().start()

        async def stop(self):
            events.append("service stopped")
            await super().stop()

    real_run = StreamableHTTPSessionManager.run

    @asynccontextmanager
    async def recording_run(self):
        async with real_run(self):
            events.append("mcp running")
            yield
        events.append("mcp stopped")

    monkeypatch.setattr(StreamableHTTPSessionManager, "run", recording_run)
    service = RecordingService(settings, FakeEngine())
    with TestClient(create_app(settings, service=service), base_url="http://127.0.0.1:7711"):
        assert events == ["service started", "mcp running"]
    assert events == ["service started", "mcp running", "mcp stopped", "service stopped"]


MCP_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}


def mcp_initialize(base, **headers):
    """POST an MCP initialize request to /mcp, as a client would, with extra headers."""
    headers = {"accept": "application/json, text/event-stream", **headers}
    return httpx2.post(f"{base}/mcp", json=MCP_INITIALIZE, headers=headers, timeout=10)


@pytest.mark.parametrize(("extra", "status"), [(0, 200), (1, 413)])
def test_mcp_has_the_same_body_limit_as_rest(settings, extra, status):
    body = padded(MCP_INITIALIZE, MAX_BODY_BYTES + extra)
    headers = {**JSON, "accept": "application/json, text/event-stream"}
    with client_for(settings, raise_server_exceptions=False) as client:
        response = client.post("/mcp", content=body, headers=headers)
    assert response.status_code == status


def test_mcp_over_http_applies_the_loopback_origin_rule(settings):
    port = free_port()
    settings = dataclasses.replace(settings, port=port)
    app = create_app(settings, service=DecisionService(settings, FakeEngine()))
    expected = {
        "http://localhost": 200,
        "http://127.0.0.1": 200,
        "http://[::1]": 200,
        "http://localhost:3000": 200,
        f"http://127.0.0.1:{port}": 200,
        "http://evil.example": 403,
        "http://localhost.evil.example": 403,
        "https://localhost": 403,
        "null": 403,
        # The SDK's own check lets these through, so the middleware has to refuse them first.
        "http://localhost:abc": 403,
        "http://localhost:123456": 403,
        "http://localhost:80@evil.example": 403,
    }
    with serving(app, port) as base:
        statuses = {origin: mcp_initialize(base, origin=origin).status_code for origin in expected}
        assert mcp_initialize(base).status_code == 200  # no Origin header at all
    assert statuses == expected


async def sdk_lets_through(port, host, origin=None):
    """Whether the SDK's own Host and Origin check, as create_app sets it, lets a request through."""
    headers = [(b"host", host.encode())]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    request = Request({"type": "http", "method": "GET", "path": "/mcp", "headers": headers})
    check = TransportSecurityMiddleware(mcp_transport_security(port))
    return await check.validate_request(request) is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1:7711",
        "localhost:7711",
        "[::1]:7711",
        "127.0.0.1:9999",
        "127.0.0.1",
        "localhost",
        "evil.example",
        "evil.example:7711",
        "127.0.0.1.evil.example:7711",
        "localhost:7711.evil.example",
    ],
)
async def test_the_sdk_check_applies_the_hosts_of_the_middleware(settings, host):
    expected = host in loopback_hosts(settings.port)
    assert await sdk_lets_through(settings.port, host) is expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    "origin",
    [
        None,
        "http://localhost",
        "http://127.0.0.1",
        "http://[::1]",
        "http://localhost:3000",
        "http://127.0.0.1:7711",
        "http://[::1]:65535",
        "http://evil.example",
        "http://localhost.evil.example",
        "http://evil.example:7711",
        "https://localhost",
        "null",
    ],
)
async def test_the_sdk_check_applies_the_origins_of_the_middleware(settings, origin):
    expected = origin is None or is_loopback_origin(origin)
    host = f"127.0.0.1:{settings.port}"
    assert await sdk_lets_through(settings.port, host, origin) is expected
