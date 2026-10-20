"""The MCP face of local-jev: decide, decide_batch and status."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from local_jev import __version__
from local_jev.errors import ServiceError

INSTRUCTIONS = (
    "local-jev answers typed questions (yes/no, choice, score) about a piece of text or JSON in tens of "
    "milliseconds, on this Mac, without generating text. Use it for fast classification, yes/no checks and "
    "rubric scores, and especially over many items with decide_batch(path=...), so the items never enter your "
    "context. It is a small encoder model: good at simple classification, weak at nuanced judgement, and its "
    "probabilities rank answers but are not calibrated."
)

# The rules that spec section 6.2 says the tool descriptions must state about questions and states.
RULES = (
    "Rules: questions are read literally, so say exactly what you mean. "
    "No arithmetic or counting; dates are read as text. "
    "Keep the state short and factual (send evidence, not summaries): the English model reads about 320 "
    "tokens of it and warns when it cut the rest. "
    "Phrase questions positively; negations are unreliable. "
    "Use at most 20 options; more are shortlisted. "
    "Probabilities rank answers but are not calibrated: confidence is not permission. "
    "The state is not treated as hostile, so never use this as the only check on untrusted text."
)

DECIDE_DESCRIPTION = (
    """Answer typed questions about one state (text, JSON object or array) with probabilities, in Jev's /v1/systemone format.

questions maps an id to one of:
- {"type": "noul", "instructions": "<yes/no question>", "criteria": {"true": "...", "false": "..."}} -> P(true) in "noul"
- {"type": "choice", "instructions": "...", "criteria": {"label": "description", ...} or ["label", ...]} -> "choice" and "probabilities"
- {"type": "score", "instructions": "...", "criteria": ["level 0", "level 1", ...]} (2-10 levels) -> expected level in "score"

"""
    + RULES
    + """ Prefer decide_batch with path for many items, so the items stay out of your context.

model: "auto" (routes by language), "english", "multilingual" or "typed-decisions"."""
)

BATCH_DESCRIPTION = (
    """Ask the same questions (at most 8, same format as decide) about many items, and get counts, the top matches and a results file, without the items entering your context.

Give exactly one of:
- path: absolute path to a local file. format: "jsonl" (one JSON object per line; field = dotted path to the text), "csv" (field = column), "lines" (one item per non-empty line), "chunks" (a long text split into pieces of about chunk_tokens tokens, with line ranges) or "auto" (by file extension).
- items: a short inline list of states.

filter (optional): {"question": id, "op": ">=" | "<=" | "==", "value": ...}. noul compares P(true), score compares the expected level, choice needs op "==" and one of its labels. matches holds up to top_k items. With a filter they are the items that pass it, most confident first (by the filter question's answer_confidence); without a filter they are the first top_k items in input order. summary counts answers per question; output_path is a JSONL file with every item's answers. On this Mac, expect about 47 short items per second with one question, about 30 with two questions and about 8 for items of about 430 tokens. A batch must finish within 300 seconds or it fails with a timeout error that says how many items were done, so split files of more than about 2,000 long items.

Each item is read like a decide state, so the rules below apply to every item. The token window in the rules applies to each item and, with format "chunks", to each chunk: keep chunk_tokens below it.

"""
    + RULES
)

STATUS_DESCRIPTION = (
    "Report the backend, its loaded and ready checkpoints and any still downloading, approximate peak memory in MB, "
    "the idle-unload timing, decide latency percentiles (median and 95th), the queue depth, the version and a "
    "summary of the configuration. Never loads a model."
)


def build_mcp_server(service: Any) -> MCPServer:
    mcp = MCPServer(name="local-jev", instructions=INSTRUCTIONS, version=__version__)

    @mcp.tool(description=DECIDE_DESCRIPTION)
    async def decide(
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
        model: str = "auto",
    ) -> dict[str, Any]:
        return await _call(
            service.decide(
                {"state": state, "questions": questions, "model": model}, interface="mcp"
            )
        )

    @mcp.tool(description=BATCH_DESCRIPTION)
    async def decide_batch(
        questions: dict[str, dict[str, Any]],
        path: str | None = None,
        items: list[Any] | None = None,
        format: Literal["auto", "jsonl", "csv", "lines", "chunks"] = "auto",
        field: str | None = None,
        chunk_tokens: int = 256,
        filter: dict[str, Any] | None = None,
        top_k: int = 20,
        model: str = "auto",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "questions": questions,
            "format": format,
            "chunk_tokens": chunk_tokens,
            "top_k": top_k,
            "model": model,
        }
        for key, value in (("path", path), ("items", items), ("field", field), ("filter", filter)):
            if value is not None:
                body[key] = value
        return await _call(service.decide_batch(body, interface="mcp"))

    @mcp.tool(description=STATUS_DESCRIPTION)
    async def status() -> dict[str, Any]:
        return service.status()

    return mcp


async def _call(awaitable: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    try:
        return await awaitable
    except ServiceError as exc:
        # A message can repeat what the caller sent, such as a path written as the JSON escape \ud800.
        # A lone surrogate is not valid UTF-8, so a response carrying it fails to serialize or to parse.
        text = exc.for_model().encode("utf-8", "backslashreplace").decode("utf-8")
        raise ToolError(text) from exc
