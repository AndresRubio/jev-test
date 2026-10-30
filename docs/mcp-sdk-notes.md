# Notes on the `mcp` SDK 2.3.0

Facts about the SDK that the repo depends on and that the SDK's own docs do not state, each with the test that pins it. Read this before reading the SDK source under `.venv/lib/python3.12/site-packages/mcp`. When a fact here stops matching the SDK, the named test fails first; update both.

## Server side

local-jev builds its server with `MCPServer` from `mcp.server` and raises `ToolError` for tool failures. A dict return type gives the tool an `outputSchema` and the SDK fills `structuredContent`; a `ToolError` becomes a result with `isError`, never a JSON-RPC error. Pinned by `test_tools_call_returns_text_and_structured_content` and `test_a_tool_error_is_a_result_with_is_error_not_a_jsonrpc_error` in `tests/test_mcp_wire.py`.

The server's Starlette app routes on one header. A POST with `MCP-Protocol-Version` set to a modern version (`2026-07-28`) goes to the per-request path; without that header the same body goes to the legacy session path, where a request other than `initialize` gets 400 "Bad Request: Missing session ID". The routing is in `mcp/server/streamable_http_manager.py`, `_handle_request`.

A modern request must carry, in this order of checking (`mcp/shared/inbound.py`, `classify_inbound_request`):

1. `params._meta` with `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities`; `io.modelcontextprotocol/clientInfo` is optional. Missing keys give `INVALID_PARAMS`.
2. The header `MCP-Protocol-Version` equal to the envelope's version and the header `Mcp-Method` equal to the body's `method`. A mismatch gives `HEADER_MISMATCH` before the version is judged.
3. A version the server supports.

A modern request opens no session, and the answer is plain JSON. Pinned by `test_a_modern_request_carries_the_version_in_a_header_and_needs_no_session` in `tests/test_mcp_wire.py`.

The logging capability (`ctx.log`, `ctx.info` and the rest on the request context) is deprecated as of `2026-07-28` (SEP-2577) and emits `MCPDeprecationWarning`. Neither local-jev nor the examples use it.

## Client side

`Client(url)` with no `mode` sends `server/discover` first and falls back to `initialize` when the server answers with an error. Against local-jev the negotiated version is `2026-07-28`, so the client's requests go down the modern path above. Pinned by `test_the_client_lists_the_tools_and_calls_decide` in `tests/test_examples.py`, which checks the printed version.

`Client(url)` builds its own `httpx2` client and accepts no headers. A bearer token goes through `httpx2.AsyncClient(headers=...)` passed as `http_client` to `streamable_http_client`, and that transport to `Client`. Pinned by `test_the_token_goes_out_as_a_bearer_header` in `tests/test_examples.py`; the pattern is `connect` in `examples/minimal_mcp_client.py`.

The models use snake_case where the wire uses camelCase: `tool.input_schema`, `result.structured_content`, `result.is_error`, `client.server_info`, `client.protocol_version`.

A non-2xx answer to a request becomes a JSON-RPC error for the caller, in this order (`mcp/client/streamable_http.py`, around line 398):

1. If the body is `application/json` and parses as a JSON-RPC error, that error is delivered as the answer.
2. Otherwise 404 becomes `METHOD_NOT_FOUND` "Not Found" when no session exists yet, or `INVALID_REQUEST` "Session terminated" when one does.
3. Any other status becomes `INTERNAL_ERROR` (-32603) "Server returned an error response".

local-jev's 401 for a missing token has no JSON-RPC body, so it reaches the caller as case 3, without its status. `Client` raises it as `MCPError` inside an `ExceptionGroup`, so a test uses `pytest.raises(ExceptionGroup)` and `group_contains(MCPError, match=...)`. Pinned by `test_the_token_goes_out_as_a_bearer_header`.
