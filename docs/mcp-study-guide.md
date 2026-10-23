# MCP study guide: how local-jev speaks the Model Context Protocol

This guide uses local-jev as a worked example of an MCP server. It is written for someone who wants to explain MCP in an interview and prefers a real codebase over a slide deck. Every claim about the protocol below was checked against the `mcp` Python SDK 2.3.0 that this repo pins, or against the messages the server sends; the trace in section 3 comes from `tests/test_mcp_wire.py`.

Read in this order:

1. This guide.
2. `src/local_jev/api_mcp.py` (144 lines): the server and its three tools.
3. `src/local_jev/app.py` (81 lines): how the server is mounted and secured.
4. `tests/test_mcp_wire.py`: the protocol, one raw message per test.
5. `examples/minimal_mcp_server.py` and `tests/test_examples.py`: resources, prompts and stdio, which local-jev does not use.

## 1. MCP in one page

The Model Context Protocol is an open standard, published by Anthropic in November 2024, for connecting an LLM application to outside tools and data. The vocabulary:

- A **host** is the application the user runs: Claude Code, Claude Desktop, Cursor, Codex. It holds the model and decides what the model may do.
- The host creates one **client** per server. The client owns the connection and the session.
- A **server** is a program that exposes capabilities. It knows nothing about the model; it only answers requests from its client.

The wire format is JSON-RPC 2.0. There are three message shapes:

| Shape | Fields | Example in local-jev |
|---|---|---|
| Request | `id`, `method`, `params` | `tools/call` |
| Response | `id`, then `result` or `error` | the answer to `tools/call` |
| Notification | `method`, `params`, no `id` | `notifications/initialized` |

A server offers up to three kinds of **primitive**. The spec describes who controls each one:

| Primitive | Controlled by | What it is | local-jev |
|---|---|---|---|
| Tools | the model | functions the model may call, with a JSON Schema for the arguments | `decide`, `decide_batch`, `status` |
| Resources | the application | data addressed by a URI, which the host may show the user or add to the context | none |
| Prompts | the user | message templates the user picks, with arguments the host fills in | none |

A client can offer primitives too: **sampling** (the server asks the client's model to complete something), **roots** (the folders the server may work in) and **elicitation** (the server asks the user a question mid-call, since protocol version 2025-06-18). local-jev uses none of these; `examples/minimal_mcp_server.py` shows the server-side ones.

A session has a **lifecycle**:

1. The client sends `initialize` with the protocol version it wants, its capabilities and its name.
2. The server answers with the version it will speak, its capabilities, its name and optional `instructions` for the model.
3. The client sends `notifications/initialized`. From here on either side may send requests.
4. The session ends when the transport closes, or with an HTTP `DELETE` on Streamable HTTP.

There are two standard **transports**:

- stdio: the client starts the server as a child process and exchanges one JSON message per line on stdin and stdout. Anything else written to stdout corrupts the stream, so logs go to stderr. This is what most editor integrations use for a server they own.
- Streamable HTTP: one URL, here `http://127.0.0.1:7711/mcp`. The client `POST`s each message. The server answers a request with either a plain JSON body or a `text/event-stream` (SSE) that carries notifications and then the response. A `GET` on the same URL opens a stream for messages the server starts. A session id travels in the `Mcp-Session-Id` header. It replaced the older HTTP+SSE transport, which needed two URLs, in version 2025-03-26.

Protocol versions are dates. The SDK in this repo knows `2024-11-05`, `2025-03-26`, `2025-06-18` and `2025-11-25`, which it calls the handshake versions, and `2026-07-28`, where every HTTP request is self-contained: no `initialize`, no session id, and the version and client capabilities travel in each request's `_meta`. local-jev's tests speak `2025-06-18`, the version most agents use today, and the SDK serves either.

## 2. Where each part of the protocol lives in this repo

| Concern | File | What to look at |
|---|---|---|
| The server and its tools | `src/local_jev/api_mcp.py` | `MCPServer(name=..., instructions=..., version=...)`, three `@mcp.tool(description=...)` functions, `_call` turning a `ServiceError` into a `ToolError` |
| Tool descriptions | `src/local_jev/api_mcp.py` | `RULES`, `DECIDE_DESCRIPTION`, `batch_description(timeout_seconds)`. Section 6.2 of the design spec treats them as part of the interface, and `tests/test_api_mcp.py` pins their phrases |
| Transport and mounting | `src/local_jev/app.py` | `mcp.streamable_http_app(streamable_http_path="/mcp", ...)` mounted at `/`, next to the REST routes |
| Session manager lifetime | `src/local_jev/app.py` | the `lifespan` runs `mcp.session_manager.run()` after the service starts and before it stops |
| Request body limit | `src/local_jev/app.py` | `max_request_body_size=MAX_BODY_BYTES`, the same 5 MB the REST routes enforce |
| Host and Origin checks | `src/local_jev/security.py` and `app.py` | `SecurityMiddleware` runs first on every request; `mcp_transport_security()` sets the SDK's own check to the same allowlist |
| Bearer token | `src/local_jev/security.py` | checked on `/mcp` like any other path except `/health` |
| The logic behind the tools | `src/local_jev/service.py` | `DecisionService.decide` and `decide_batch` take the same body from REST and MCP, with an `interface` tag for the log |
| Registering the server in a host | `src/local_jev/cli.py`, `cmd_connect` | the `claude mcp add` command, the Codex TOML, the Cursor JSON |
| In-memory tests | `tests/test_api_mcp.py` | `Client(build_mcp_server(service))`: no HTTP, no transport |
| HTTP tests with the SDK client | `tests/test_app.py` | `Client(f"{base}/mcp")` against uvicorn, plus the Origin and Host tables |
| Raw wire tests | `tests/test_mcp_wire.py` | the JSON-RPC messages themselves, through `TestClient` |

Three things to notice about the shape of the code.

MCP is a thin face over the service. `api_mcp.py` holds no business logic. Each tool builds the same body the REST route receives and hands it to `DecisionService`. Validation, guards, the inference queue, logging and the batch reader are shared. An interviewer who asks "how would you add MCP to an existing service?" is asking for this shape.

The SDK generates the schemas. `decide(state: str | dict | list, questions: dict[str, dict], model: str = "auto") -> dict` becomes a tool with an `inputSchema` whose `required` is `["state", "questions"]`, a default for `model`, and an `outputSchema` because the return annotation is a dict. The server never writes JSON Schema by hand.

The description is derived from configuration. `batch_description()` puts the batch time limit and the number of items to split at into the text, so the model reads the limit the server will apply, and `test_decide_batch_states_the_time_limit_the_config_sets` checks that.

## 3. The wire, message by message

The messages below are what `tests/test_mcp_wire.py` sends and receives through the real app with a fake engine. Headers and bodies are trimmed where marked.

### 3.1 initialize

The client posts, with `Accept: application/json, text/event-stream`:

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize",
 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "wire-test", "version": "0"}}}
```

The server answers `200` with `Content-Type: text/event-stream` and `Mcp-Session-Id: c0784af7...` (32 hex characters). The body is one SSE event:

```text
event: message
data: {"jsonrpc":"2.0","id":1,"result":{
  "protocolVersion":"2025-06-18",
  "capabilities":{"prompts":{"listChanged":false},
                  "resources":{"listChanged":false,"subscribe":false},
                  "tools":{"listChanged":false}},
  "serverInfo":{"name":"local-jev","version":"0.1.0"},
  "instructions":"local-jev answers typed questions (yes/no, choice, score) ..."}}
```

Points to make:

- The session id travels in a header, outside the JSON-RPC message. Every later request must carry it.
- `capabilities` tells the client which methods exist. `listChanged: false` means the server will not send `notifications/tools/list_changed`, so the client may cache `tools/list`.
- `instructions` is free text for the model. local-jev uses it to say what the model is good and bad at. This is the only place a server speaks to the model outside of tool descriptions.
- The SDK answers even a request with no streaming need as SSE. That is allowed: the response is the last event. `streamable_http_app(json_response=True)` would make it plain JSON.

### 3.2 notifications/initialized

```json
{"jsonrpc": "2.0", "method": "notifications/initialized"}
```

No `id`, so no response is possible. The server answers `202 Accepted` with an empty body. Before it arrives, the server should send the client nothing but pings and log messages.

### 3.3 tools/list

```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
```

The result lists three tools. For `decide`, trimmed:

```json
{"name": "decide",
 "description": "Answer typed questions about one state ... confidence is not permission ...",
 "inputSchema": {"type": "object", "required": ["state", "questions"],
   "properties": {"state": {"anyOf": [{"type": "string"}, {"type": "object"}, {"type": "array"}]},
                  "questions": {"type": "object", "additionalProperties": {"type": "object"}},
                  "model": {"type": "string", "default": "auto"}}},
 "outputSchema": {"type": "object", "additionalProperties": true}}
```

The model sees the name, the description and the input schema. That is the whole interface, which is why the description carries the rules about questions and why the tests pin them.

### 3.4 tools/call, a good call

```json
{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
 "params": {"name": "decide",
            "arguments": {"state": "please check billing",
                          "questions": {"dept": {"type": "choice", "instructions": "Which team?",
                                                 "criteria": {"billing": "refunds", "tech": "bugs"}}}}}}
```

The result has two copies of the answer:

```json
{"isError": false,
 "content": [{"type": "text", "text": "{\n  \"model\": \"fake-model\",\n  \"answers\": {...}\n}"}],
 "structuredContent": {"model": "fake-model",
                       "answers": {"dept": {"type": "choice", "choice": "billing",
                                            "probabilities": {"billing": 0.8, "tech": 0.2}, ...}},
                       "usage": {...}, "local": {...}}}
```

`structuredContent` arrived in version 2025-06-18 and must match `outputSchema`. The `content` text block is the same JSON, for clients that read content only. A client that understands both can validate the structured copy and show the text.

### 3.5 tools/call, a bad call

```json
{"jsonrpc": "2.0", "id": 4, "method": "tools/call",
 "params": {"name": "decide", "arguments": {"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}}}}
```

```json
{"isError": true,
 "content": [{"type": "text", "text": "Error executing tool decide: invalid_request: question 'q' has type 'maybe' Hint: type must be 'noul' (yes/no), 'choice' or 'score'"}]}
```

This is a **tool error**, which the spec distinguishes from a **protocol error**:

- A tool error is a normal JSON-RPC `result` with `isError: true`. The model reads the text, so the message should say how to fix the call. local-jev's `ServiceError.for_model()` includes a `Hint:` for that reason, and `_call` in `api_mcp.py` raises the SDK's `ToolError` with it.
- A protocol error is a JSON-RPC `error` with a numeric code: a malformed message, an unknown method (`-32601`), a missing session. The host handles it; the model never sees it.

An unknown tool name is also a tool error in this SDK: `{"isError": true, "content": [{"type": "text", "text": "Unknown tool: nope"}]}`.

### 3.6 Session rules of Streamable HTTP

| Request | Response | Why |
|---|---|---|
| `POST` without `Mcp-Session-Id` after initialize | `400`, `"Bad Request: Missing session ID"` | the server is stateful; each session has its own transport |
| `POST` with an unknown or expired id | `404`, `"Session not found"` | the client must start over with `initialize` |
| `POST` whose `Accept` lacks `text/event-stream` | `406` | the server may need to stream |
| `DELETE /mcp` with the session id | `200`; later requests get `404` | explicit termination |
| `GET /mcp` with `Accept: text/event-stream` | an SSE stream | for messages the server starts, such as list-changed notifications |

These HTTP responses carry a JSON-RPC error body with `id: null`, because they are about the transport, not about any request. Sessions in the SDK expire after 30 idle minutes and are capped at 10,000 by default.

## 4. Design decisions worth explaining

These are the choices an interviewer can dig into, with the reasoning this repo recorded.

### Why Streamable HTTP and not stdio

With stdio each client starts its own server process. local-jev holds about 1.5 GB of model weights in memory and takes about 0.8 s to load them, and several agents on the Mac may use it at once, so one long-running process on a port is the right shape. A launchd login agent keeps it up. The cost is that a network listener needs its own security, which stdio gets for free from the process boundary.

### Loopback is not enough

The server binds to 127.0.0.1, which keeps other machines out but not other programs. Two attacks remain:

- *DNS rebinding.* A web page on `evil.example` makes the browser resolve that name to 127.0.0.1 and sends a request the same-origin policy allows. The `Host` header still says `evil.example`, so `SecurityMiddleware` refuses any Host that is not `127.0.0.1`, `localhost` or `[::1]` on the configured port, with a `421`. The MCP spec tells every local HTTP server to do this.
- *A page on a local origin.* The `Origin` header names the page that sent the request. The middleware allows only loopback origins over plain http and refuses the rest, including `Origin: null`, with a `403`. The SDK has the same check, `TransportSecuritySettings`, and `app.py` configures it with the same allowlist as a second line of defence. The docstring of `mcp_transport_security()` lists where the two checks differ and why the middleware runs first.

Every account on the Mac can also reach 127.0.0.1, so an optional bearer token (`LOCAL_JEV_TOKEN`) guards every path but `/health`. This is a plain header check, not MCP's OAuth 2.1 authorization flow, which is meant for servers on the network.

### Tool descriptions are the prompt

The model never reads the README. It reads `instructions` once and the three descriptions on every turn. So the rules about how to ask a question (literal reading, no arithmetic, positive phrasing, at most 20 options, probabilities are not calibrated) live in the description, and the batch description quotes the server's own time limit. `tests/test_api_mcp.py` fails if a rule is dropped.

### Keep the data out of the context

`decide_batch` reads a file of up to 5,000 items on the server side and returns counts, the top matches and the path of a results file. The items never pass through the model's context window. This is the argument for running classification in a tool instead of asking the model to read everything, and the `decide` description tells the model to prefer it for many items.

### Structured output and the text copy

A dict return type gives each tool an `outputSchema`, and the SDK fills `structuredContent`. The text block holds the same JSON so older clients still work. The REST responses and the MCP results are the same dictionaries, produced once in `DecisionService`.

### Errors that the model can act on

A `ServiceError` has an HTTP status for REST and a code, message and hint for everyone. Over MCP it becomes `isError: true` with the hint in the text. One subtlety in `_call`: a message can repeat a path the caller sent as the JSON escape `\ud800`, a lone surrogate that is not valid UTF-8, so the text is re-encoded with `backslashreplace` before it goes on the wire.

### Lifespan order

The Starlette lifespan starts the service (inference worker, prefetch, log files), then runs the SDK's session manager, and stops them in reverse. `test_the_lifespan_starts_the_service_then_mcp_and_stops_them_in_reverse` pins the order. The session manager must run inside the app's lifespan or every request fails with "session manager not started".

### One body limit

`MAX_BODY_BYTES` is passed to `streamable_http_app`, so a 5 MB JSON-RPC message is refused with a `413` exactly like a REST body. Without it the SDK's default of 4 MB would apply to MCP only.

### What local-jev leaves out, and why

No resources (there is no data to browse; a results file path in the batch answer does the job), no prompts (the tool descriptions carry the guidance), no sampling or elicitation (the server never needs the model or the user), no progress notifications (calls take tens of milliseconds, batches have their own timeout), no event store for resumable streams, no stateless mode, and no OAuth. Each would be a few lines with this SDK; the example in section 5 shows the server-side primitives.

## 5. The minimal example: every server primitive on stdio

`examples/minimal_mcp_server.py` is under 90 lines, half of them comments. It registers:

- a tool, `word_count(text, ctx)`, that returns a dict and raises `ToolError` on empty input. The `ctx: Context` parameter is injected by the SDK, stays out of the input schema and gives the tool `report_progress`, elicitation, resource reads and list-changed notifications;
- a static resource, `study://notes`, listed by `resources/list`;
- a resource template, `study://notes/{topic}`, listed by `resources/templates/list`, whose parameter comes from the URI. The SDK refuses `..` in it before the function runs;
- a prompt, `review(code, focus="correctness")`, which returns user messages. `prompts/list` reports `code` as required and `focus` as optional from the signature.

`tests/test_examples.py` drives it with the SDK's in-memory client, which negotiates `2026-07-28` and needs no transport. To run it under a host:

```bash
claude mcp add --transport stdio study-server -- uv run python examples/minimal_mcp_server.py
```

## 6. Interview questions, with the answers this repo gives

### What problem does MCP solve?

Every host used to need its own integration for every tool. MCP is one protocol between hosts and servers, so a server written once works in Claude Code, Cursor and Codex, and a host supports every server at once. local-jev's `connect` command prints three configurations for the same URL.

### What is the difference between a tool, a resource and a prompt?

Who controls it. The model calls tools. The application fetches resources, by URI, to put data in front of the user or the model. The user picks prompts. local-jev needs only tools; the example shows all three.

### How does a client learn what a server can do?

`initialize` returns the capabilities, then `tools/list`, `resources/list`, `resources/templates/list` and `prompts/list` return the items. The `listChanged` flags say whether the server will announce changes, so the client knows whether it may cache.

### Where does the schema for tool arguments come from?

The server sends a JSON Schema in `inputSchema`. The Python SDK builds it from type annotations through pydantic. `decide`'s `required: ["state", "questions"]` and the `"default": "auto"` on `model` come straight from the signature.

### How does a server tell the model how to use a tool well?

In the tool description and the server `instructions`. Nothing else reaches the model. local-jev puts the rules of the model's behaviour there and tests their presence.

### What is the difference between a tool error and a protocol error?

A tool error is a result with `isError: true` and text the model reads and can act on. A protocol error is a JSON-RPC `error` for the host: malformed message, unknown method, missing session. Validation failures in local-jev are tool errors with a hint, so the model retries with a fixed call.

### What is `structuredContent`?

Since version 2025-06-18 a tool can declare an `outputSchema` and return `structuredContent` that matches it, next to the `content` blocks. Clients can validate it and read fields without parsing text.

### Compare stdio and Streamable HTTP

stdio: the client owns the process, one server per client, no network, no auth needed, logs must avoid stdout. Streamable HTTP: one URL, many clients, sessions by header, server-initiated messages over SSE, needs Host and Origin checks and authentication. local-jev chose HTTP because the model weights are expensive to hold and the server should run at login.

### What does `Mcp-Session-Id` do?

The server mints it on `initialize` and the client sends it on every later request. The server keeps one transport per session. A missing id is `400`, an unknown one `404`, and `DELETE` ends the session.

### Why does the server answer with `text/event-stream` instead of JSON?

Because while a request runs the server may send notifications (progress, logs) that belong to that request, and SSE lets it put them in front of the response on the same connection. The response is the last event. A server that never streams can set `json_response=True`.

### What is DNS rebinding and why does a local MCP server care?

A hostile page's domain is pointed at 127.0.0.1 so the browser treats a request to the local server as same-origin. The server sees a `Host` header that is not its own and must refuse it. local-jev refuses with `421` in its middleware and again in the SDK's transport security.

### How would you authenticate a Streamable HTTP server?

On a network, with the OAuth 2.1 flow the spec defines, where the server is a resource server and the host obtains a token. On loopback, local-jev uses a static bearer token in the `Authorization` header, set through the host's header configuration (`claude mcp add --header ...`), which keeps other accounts on the same machine out.

### How do you keep a large dataset out of the model's context?

Do the work in the tool and return a summary. `decide_batch` returns counts, the top matches and a results file path for up to 5,000 items.

### How do you test an MCP server?

Three levels, all in this repo: the SDK's in-memory client against the server object (`tests/test_api_mcp.py`), the SDK's HTTP client against a real uvicorn (`tests/test_app.py`), and raw JSON-RPC over the transport to check protocol rules (`tests/test_mcp_wire.py`). A fake engine stands in for the model, so none of them needs weights.

### What is version negotiation?

The client proposes a version in `initialize`; the server answers with that version if it supports it, or with one it does support, and the client disconnects if it cannot speak that one. On HTTP the client then repeats the version in the `MCP-Protocol-Version` header. Versions are dates; the ones in this repo are listed in section 1.

### What is sampling?

A server asks the client to run its model on messages the server composes. It lets a server use the model without holding an API key. local-jev does not use it.

### What is elicitation?

A server asks the user a question in the middle of a request, through the client, for example to confirm an action or fill in a missing value. Added in 2025-06-18.

### What happens on a notification?

Nothing is sent back. Over HTTP the server returns `202 Accepted` with an empty body.

### How would you add MCP to an existing service?

Build the service so that it takes plain bodies and returns plain dicts, then write a thin adapter per interface. local-jev's `api_mcp.py` and `api_rest.py` are each about 100 lines over a shared `DecisionService`.

### What would you change in local-jev's MCP server?

Options recorded in this repo: generate a bearer token by default, add a resource that exposes the results file instead of a path, and send progress notifications during long batches. Each is a small change with this SDK.

## 7. Try it yourself

With `local-jev serve` running on port 7711, the whole handshake in curl. The `-D -` prints the response headers so the session id is visible:

```bash
curl -s -D - http://127.0.0.1:7711/mcp -H 'Accept: application/json, text/event-stream' -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

Copy the `mcp-session-id` value into `SID`, then:

```bash
curl -s http://127.0.0.1:7711/mcp -H 'Accept: application/json, text/event-stream' -H 'Content-Type: application/json' -H "Mcp-Session-Id: $SID" -H 'MCP-Protocol-Version: 2025-06-18' -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'
```

```bash
curl -s http://127.0.0.1:7711/mcp -H 'Accept: application/json, text/event-stream' -H 'Content-Type: application/json' -H "Mcp-Session-Id: $SID" -H 'MCP-Protocol-Version: 2025-06-18' -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

```bash
curl -s http://127.0.0.1:7711/mcp -H 'Accept: application/json, text/event-stream' -H 'Content-Type: application/json' -H "Mcp-Session-Id: $SID" -H 'MCP-Protocol-Version: 2025-06-18' -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"status","arguments":{}}}'
```

Without a running server, the same exchange runs against the app in-process:

```bash
uv run pytest tests/test_mcp_wire.py tests/test_examples.py -v
```

To see the server from a host's side, register it and ask the agent to list its tools:

```bash
local-jev connect --apply
```

## 8. References

- The specification, with the lifecycle, transports and each primitive: https://modelcontextprotocol.io/specification
- The Python SDK this repo pins: https://github.com/modelcontextprotocol/python-sdk
- Section 6.2 of the design spec, the MCP interface local-jev promises: `docs/superpowers/specs/2026-09-25-local-jev-design.md`
- Section 9 of the design spec, security and privacy, for the Host, Origin and token rules.
