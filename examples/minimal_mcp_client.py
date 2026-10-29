"""The smallest MCP client: connect to local-jev over Streamable HTTP, list its tools, call one.

This is the other side of src/local_jev/api_mcp.py. A host such as Claude Code does the same
three things, then hands the tool list to the model and relays its calls. docs/mcp-study-guide.md
section 5 walks through it; tests/test_examples.py runs it against the app with a fake engine.

Run it with `local-jev serve` listening on port 7711:

    uv run python examples/minimal_mcp_client.py "The site is down and nobody can log in"

Another URL goes in LOCAL_JEV_MCP_URL, and LOCAL_JEV_TOKEN is sent as a bearer token when set.
"""

import asyncio
import json
import os
import sys

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

DEFAULT_URL = "http://127.0.0.1:7711/mcp"
DEFAULT_TEXT = "The site is down and nobody can log in"

# One typed question in local-jev's format: a choice between three teams. The description of the
# `decide` tool, which the model reads on every turn, explains the other types and the rules.
QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": ["billing", "tech", "sales"],
    }
}


def connect(url: str, token: str | None = None) -> Client:
    """A client for url. `Client(url)` builds its own HTTP client without headers, so a
    bearer token needs an httpx2 client passed to the transport."""
    if token is None:
        return Client(url)
    http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"})
    return Client(streamable_http_client(url, http_client=http))


async def run(url: str, text: str, token: str | None = None) -> dict:
    """Connect, list the tools, call `decide` on text; return the structured result."""
    async with connect(url, token) as client:
        # Entering the context did the handshake. By default the SDK asks `server/discover` first
        # (protocol 2026-07-28, no session) and falls back to `initialize` on an older server.
        print(f"protocol {client.protocol_version}, server {client.server_info.name}")

        # The list is what a host gives the model: a name, a description and a JSON Schema each.
        for tool in (await client.list_tools()).tools:
            required = tool.input_schema.get("required", [])
            print(f"tool {tool.name}({', '.join(required)})")

        result = await client.call_tool("decide", {"state": text, "questions": QUESTIONS})
        if result.is_error:
            # A tool error, not a protocol error: the text is written for the model to act on.
            raise SystemExit(f"decide failed: {result.content[0].text}")
        # `content` carries the same JSON as text for clients that only read text.
        answer = result.structured_content["answers"]["team"]
        print(f"team {answer['choice']} {json.dumps(answer['probabilities'])}")
        return result.structured_content


if __name__ == "__main__":
    asyncio.run(
        run(
            os.environ.get("LOCAL_JEV_MCP_URL", DEFAULT_URL),
            sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT,
            os.environ.get("LOCAL_JEV_TOKEN"),
        )
    )
