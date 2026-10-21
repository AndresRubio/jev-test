"""The smallest MCP server that shows every server primitive: a tool, two resources and a prompt.

local-jev's real server (src/local_jev/api_mcp.py) exposes tools only and serves them over
Streamable HTTP. This file is the companion for study: it runs over stdio, the transport most
editors and desktop agents use for a server they start themselves, and it exercises the parts
local-jev leaves out. docs/mcp-study-guide.md walks through it.

Run it under an MCP client:

    claude mcp add --transport stdio study-server -- uv run python examples/minimal_mcp_server.py

or talk to it in-process with the SDK's client, as tests/test_examples.py does.
"""

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.prompts.base import UserMessage

# A real server would read this from a file or a database. The point is that a resource is
# data the host may show the user or put into the model's context; it is not a function call.
NOTES = {
    "mcp": "MCP is JSON-RPC 2.0 over a transport, with tools, resources and prompts on top.",
    "transports": "stdio for a server the client starts; Streamable HTTP for a shared one.",
}

mcp = MCPServer(
    name="study-server",
    version="0.1.0",
    instructions=(
        "A study server. Call word_count to count words, read study://notes/{topic} for a "
        "note, and use the review prompt to ask for a code review."
    ),
)


@mcp.tool()
async def word_count(text: str, ctx: Context) -> dict[str, int]:
    """Count the words and characters in text.

    The return annotation makes the SDK publish an outputSchema and put the dict in
    structuredContent as well as in a text block. ctx is the request's view of the session: it
    reports progress, asks the user a question (elicitation), reads a resource or publishes a
    list-changed event. The SDK fills it in and keeps it out of the tool's input schema.
    """
    if not text.strip():
        # ToolError becomes a result with isError=true, so the model sees the message. Any other
        # exception also becomes isError=true, but with the SDK's generic wording.
        raise ToolError("text is empty: send the text to count")
    # Sent as notifications/progress when the client asked for progress with a progressToken in
    # the request's _meta; dropped otherwise.
    await ctx.report_progress(1, 1, "counted")
    return {"words": len(text.split()), "characters": len(text)}


@mcp.resource("study://notes", name="notes", description="The list of study notes")
def list_notes() -> str:
    """A static resource: one fixed URI, listed by resources/list."""
    return "\n".join(f"study://notes/{topic}" for topic in NOTES)


@mcp.resource("study://notes/{topic}", name="note", description="One study note")
def read_note(topic: str) -> str:
    """A resource template: the {topic} in the URI becomes the parameter.

    Templates appear in resources/templates/list, not in resources/list, because the client
    fills them in. The SDK rejects path traversal in the parameter before this runs.
    """
    if topic not in NOTES:
        raise ValueError(f"no note about {topic!r}; the topics are {', '.join(NOTES)}")
    return NOTES[topic]


@mcp.prompt(description="Ask for a review of a piece of code")
def review(code: str, focus: str = "correctness") -> list[UserMessage]:
    """A prompt is a message template the user picks, and the host fills in its arguments.

    The server never sees the model's answer. It only hands back the messages.
    """
    return [UserMessage(f"Review this code for {focus}. Reply with a list of findings.\n\n{code}")]


if __name__ == "__main__":
    # stdio: JSON-RPC messages, one per line, on stdin and stdout. Anything else the process
    # prints to stdout corrupts the stream, so logs must go to stderr.
    mcp.run(transport="stdio")
