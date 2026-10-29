"""The MCP exchange over Streamable HTTP, message by message, through the whole local-jev app.

tests/test_api_mcp.py talks to the MCPServer in memory and tests/test_app.py uses the SDK's
client over HTTP. This file sends the raw JSON-RPC messages a client sends, so each test shows
one rule of the protocol as the server applies it. docs/mcp-study-guide.md explains the flow.
"""

import json

import pytest
from starlette.testclient import TestClient

from local_jev.app import create_app
from local_jev.service import DecisionService
from tests.fakes import FakeEngine

PROTOCOL_VERSION = "2025-06-18"
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "wire-test", "version": "0"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
DEPT = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]}}


def rpc(request_id, method, params=None):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def payload(response):
    """The JSON-RPC message inside a response, whether it came as JSON or as one SSE event."""
    if response.headers["content-type"].startswith("text/event-stream"):
        lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
        assert len(lines) == 1, response.text
        return json.loads(lines[0].removeprefix("data: "))
    return response.json()


@pytest.fixture
def client(settings):
    service = DecisionService(settings, FakeEngine())
    app = create_app(settings, service=service)
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.port}") as client:
        yield client


@pytest.fixture
def session(client):
    """Headers for an initialized session: the Accept pair, the session id and the version."""
    response = client.post("/mcp", json=INITIALIZE, headers=HEADERS)
    headers = {
        **HEADERS,
        "mcp-session-id": response.headers["mcp-session-id"],
        "mcp-protocol-version": PROTOCOL_VERSION,
    }
    assert client.post("/mcp", json=INITIALIZED, headers=headers).status_code == 202
    return headers


def test_initialize_returns_the_server_identity_capabilities_and_instructions(client):
    response = client.post("/mcp", json=INITIALIZE, headers=HEADERS)
    assert response.status_code == 200
    # The session id is minted on initialize and travels in a header, not in the JSON-RPC body.
    assert len(response.headers["mcp-session-id"]) == 32
    result = payload(response)["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "local-jev"
    # Capabilities say which primitives the client may ask about; local-jev registers tools
    # only, so the resource and prompt lists are empty but the methods still answer.
    assert set(result["capabilities"]) == {"prompts", "resources", "tools"}
    assert result["instructions"].startswith("local-jev answers typed questions")


def test_a_notification_gets_202_and_no_body(client):
    response = client.post("/mcp", json=INITIALIZE, headers=HEADERS)
    headers = {**HEADERS, "mcp-session-id": response.headers["mcp-session-id"]}
    response = client.post("/mcp", json=INITIALIZED, headers=headers)
    assert response.status_code == 202
    assert response.text == ""


def test_tools_list_carries_the_description_and_both_schemas(client, session):
    result = payload(client.post("/mcp", json=rpc(2, "tools/list"), headers=session))["result"]
    tools = {tool["name"]: tool for tool in result["tools"]}
    assert set(tools) == {"decide", "decide_batch", "status"}
    decide = tools["decide"]
    # The input schema comes from the Python signature: state, questions and an optional model.
    assert decide["inputSchema"]["required"] == ["state", "questions"]
    assert decide["inputSchema"]["properties"]["model"]["default"] == "auto"
    # A dict return annotation gives the tool an output schema, so results are structured.
    assert decide["outputSchema"]["type"] == "object"
    assert "confidence is not permission" in decide["description"]


def test_tools_call_returns_text_and_structured_content(client, session):
    call = rpc(
        3,
        "tools/call",
        {"name": "decide", "arguments": {"state": "tech outage", "questions": DEPT}},
    )
    response = client.post("/mcp", json=call, headers=session)
    # The SDK streams a response as one SSE event, so a server could add notifications first.
    assert response.headers["content-type"].startswith("text/event-stream")
    result = payload(response)["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["answers"]["dept"]["choice"] == "tech"
    # The text block holds the same JSON for clients that read content only.
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


def test_a_tool_error_is_a_result_with_is_error_not_a_jsonrpc_error(client, session):
    bad = {"q": {"type": "maybe", "instructions": "?"}}
    call = rpc(4, "tools/call", {"name": "decide", "arguments": {"state": "x", "questions": bad}})
    message = payload(client.post("/mcp", json=call, headers=session))
    # The model gets the message and can retry; a JSON-RPC error would end the call instead.
    assert "error" not in message
    result = message["result"]
    assert result["isError"] is True
    assert "invalid_request" in result["content"][0]["text"]
    assert "Hint:" in result["content"][0]["text"]
    assert "structuredContent" not in result


def test_an_unknown_tool_is_also_an_is_error_result(client, session):
    call = rpc(5, "tools/call", {"name": "nope", "arguments": {}})
    result = payload(client.post("/mcp", json=call, headers=session))["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"] == "Unknown tool: nope"


def test_ping_answers_with_an_empty_result(client, session):
    assert payload(client.post("/mcp", json=rpc(6, "ping"), headers=session))["result"] == {}


def test_the_resource_and_prompt_lists_are_empty(client, session):
    resources = payload(client.post("/mcp", json=rpc(7, "resources/list"), headers=session))
    prompts = payload(client.post("/mcp", json=rpc(8, "prompts/list"), headers=session))
    assert resources["result"] == {"resources": []}
    assert prompts["result"] == {"prompts": []}


def test_a_request_without_a_session_id_is_400(client, session):
    headers = {key: value for key, value in session.items() if key != "mcp-session-id"}
    response = client.post("/mcp", json=rpc(9, "ping"), headers=headers)
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Bad Request: Missing session ID"


def test_an_unknown_session_id_is_404(client, session):
    response = client.post(
        "/mcp", json=rpc(9, "ping"), headers={**session, "mcp-session-id": "x" * 32}
    )
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Session not found"


def test_the_client_must_accept_json_and_sse(client, session):
    response = client.post(
        "/mcp", json=rpc(9, "ping"), headers={**session, "accept": "application/json"}
    )
    assert response.status_code == 406


def test_delete_ends_the_session(client, session):
    assert client.delete("/mcp", headers=session).status_code == 200
    assert client.post("/mcp", json=rpc(10, "ping"), headers=session).status_code == 404


def test_a_second_initialize_opens_a_second_independent_session(client, session):
    response = client.post("/mcp", json=INITIALIZE, headers=HEADERS)
    second = response.headers["mcp-session-id"]
    assert second != session["mcp-session-id"]
    # Each session is one client; both reach the same DecisionService behind the server.
    assert client.post("/mcp", json=rpc(2, "ping"), headers=session).status_code == 200


def test_a_modern_request_carries_the_version_in_a_header_and_needs_no_session(client):
    # The SDK's client sends this first: server/discover is the single request of 2026-07-28.
    # The header routes it, the _meta envelope must repeat the version and carry the client
    # capabilities, and Mcp-Method must match the body. The server answers in plain JSON, mints
    # no session, and streams nothing. Without the headers the same body is a 400 (section 3.6).
    envelope = {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    }
    headers = {**HEADERS, "mcp-protocol-version": "2026-07-28", "mcp-method": "server/discover"}
    response = client.post("/mcp", json=rpc(1, "server/discover", envelope), headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "mcp-session-id" not in response.headers
    result = payload(response)["result"]
    assert result["capabilities"]["tools"] == {"listChanged": True}
    assert result["instructions"].startswith("local-jev answers typed questions")

    no_headers = client.post("/mcp", json=rpc(1, "server/discover", envelope), headers=HEADERS)
    assert no_headers.status_code == 400
    assert "session" in payload(no_headers)["error"]["message"].lower()
