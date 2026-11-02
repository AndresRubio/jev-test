"""The facts about the mcp SDK that docs/mcp-sdk-notes.md states and no other test pins.

Each test here is named in the notes. When the SDK pin in pyproject.toml moves and one of
them fails, the note it pins is what changed."""

import dataclasses

import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.shared.exceptions import MCPDeprecationWarning, MCPError
from mcp_types.jsonrpc import HEADER_MISMATCH, INVALID_PARAMS, METHOD_NOT_FOUND
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from local_jev.app import create_app
from local_jev.service import DecisionService
from tests.fakes import FakeEngine
from tests.test_app import free_port, serving
from tests.test_mcp_wire import HEADERS, rpc

MODERN = "2026-07-28"
META = {
    "io.modelcontextprotocol/protocolVersion": MODERN,
    "io.modelcontextprotocol/clientCapabilities": {},
}
MODERN_HEADERS = {**HEADERS, "mcp-protocol-version": MODERN, "mcp-method": "server/discover"}


def make_app(settings):
    return create_app(settings, service=DecisionService(settings, FakeEngine()))


@pytest.fixture
def client(settings):
    with TestClient(make_app(settings), base_url=f"http://127.0.0.1:{settings.port}") as client:
        yield client


# The inbound ladder: envelope first, headers second, version last.


def test_a_modern_request_without_the_envelope_keys_is_invalid_params(client):
    meta = {"io.modelcontextprotocol/protocolVersion": MODERN}
    body = rpc(1, "server/discover", {"_meta": meta})
    error = client.post("/mcp", json=body, headers=MODERN_HEADERS).json()["error"]
    assert error["code"] == INVALID_PARAMS
    assert "clientCapabilities" in error["message"]


def test_a_method_header_that_disagrees_with_the_body_is_a_header_mismatch(client):
    headers = {**MODERN_HEADERS, "mcp-method": "tools/list"}
    body = rpc(1, "server/discover", {"_meta": META})
    error = client.post("/mcp", json=body, headers=headers).json()["error"]
    assert error["code"] == HEADER_MISMATCH
    assert "mcp-method" in error["message"]


def test_the_header_rung_runs_before_the_version_rung(client):
    """A version nobody supports and a mismatched method header: the mismatch is reported."""
    meta = {**META, "io.modelcontextprotocol/protocolVersion": "2099-01-01"}
    headers = {**MODERN_HEADERS, "mcp-protocol-version": "2099-01-01", "mcp-method": "tools/list"}
    body = rpc(1, "server/discover", {"_meta": meta})
    assert (
        client.post("/mcp", json=body, headers=headers).json()["error"]["code"] == HEADER_MISMATCH
    )


# The client's handshake and its error mapping, over real HTTP.


class RefusingDiscover:
    """The app behind a wrapper that answers server/discover with a JSON-RPC METHOD_NOT_FOUND
    at 404, the way a server from before 2026-07-28 does."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        headers = dict(scope["headers"]) if scope["type"] == "http" else {}
        if headers.get(b"mcp-method") == b"server/discover":
            error = {"code": METHOD_NOT_FOUND, "message": "Method not found"}
            response = JSONResponse({"jsonrpc": "2.0", "id": None, "error": error}, status_code=404)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


@pytest.mark.anyio
async def test_the_client_falls_back_to_initialize_when_discover_is_refused(settings):
    settings = dataclasses.replace(settings, port=free_port())
    with serving(RefusingDiscover(make_app(settings)), settings.port) as base:
        async with Client(f"{base}/mcp") as session:
            assert session.protocol_version == "2025-11-25"
            assert session.server_info.name == "local-jev"
            tools = (await session.list_tools()).tools
    assert [tool.name for tool in tools] == ["decide", "decide_batch", "status"]


@pytest.mark.anyio
async def test_a_404_without_a_json_rpc_body_is_method_not_found_before_any_session():
    gone = PlainTextResponse("gone", status_code=404)
    not_found = Starlette(routes=[Route("/mcp", gone, methods=["POST"])])
    with serving(not_found, free_port()) as base:
        with pytest.raises(ExceptionGroup) as failed:
            async with Client(f"{base}/mcp"):
                pass
    assert failed.group_contains(MCPError, match="Not Found")
    assert {e.error.code for e in leaves(failed.value) if isinstance(e, MCPError)} == {
        METHOD_NOT_FOUND
    }


def leaves(exc):
    """Every exception inside nested ExceptionGroups."""
    if isinstance(exc, ExceptionGroup):
        return [leaf for inner in exc.exceptions for leaf in leaves(inner)]
    return [exc]


# The deprecated logging capability.


@pytest.mark.anyio
async def test_the_logging_methods_on_the_context_are_deprecated():
    server = MCPServer("logging-probe")

    @server.tool()
    async def shout(text: str, ctx: Context) -> str:
        await ctx.info(text)
        return text.upper()

    with pytest.warns(MCPDeprecationWarning, match="logging capability is deprecated"):
        async with Client(server) as session:
            result = await session.call_tool("shout", {"text": "hi"})
    assert result.content[0].text == "HI"
