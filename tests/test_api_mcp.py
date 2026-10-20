import pytest
from mcp import Client

from local_jev.api_mcp import build_mcp_server
from local_jev.service import DecisionService
from tests.fakes import FakeEngine

Q = {
    "dept": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "refunds", "tech": "bugs"},
    }
}


@pytest.fixture
def service(settings):
    svc = DecisionService(settings, FakeEngine())
    yield svc
    svc.runtime.shutdown()
    svc.log.close()


@pytest.mark.anyio
async def test_the_three_tools_are_listed_with_their_rules(service):
    async with Client(build_mcp_server(service)) as client:
        listing = await client.list_tools()
    tools = {tool.name: tool for tool in listing.tools}
    assert set(tools) == {"decide", "decide_batch", "status"}
    assert "confidence is not permission" in tools["decide"].description
    assert "absolute path" in tools["decide_batch"].description


async def listed(service) -> dict[str, str]:
    """The tool descriptions as an agent sees them, through tools/list."""
    async with Client(build_mcp_server(service)) as client:
        listing = await client.list_tools()
    return {tool.name: tool.description for tool in listing.tools}


# One distinctive phrase for each rule that spec section 6.2 says the descriptions must state.
# The eighth rule, to prefer decide_batch with path for many items, belongs to decide alone.
RULE_PHRASES = [
    "read literally",
    "no arithmetic or counting",
    "dates are read as text",
    "evidence, not summaries",
    "about 320 tokens",
    "negations are unreliable",
    "at most 20 options",
    "confidence is not permission",
    "only check on untrusted text",
]

# What spec section 6.2 lists for status(), in the words of the payload.
STATUS_TERMS = [
    "backend",
    "loaded",
    "ready",
    "downloading",
    "memory",
    "idle-unload",
    "median",
    "95th",
    "queue depth",
    "version",
    "configuration",
    "never loads a model",
]


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["decide", "decide_batch"])
async def test_the_question_tools_state_the_rules_of_the_spec(service, tool):
    text = (await listed(service))[tool].lower()
    assert [phrase for phrase in RULE_PHRASES if phrase not in text] == []


@pytest.mark.anyio
async def test_decide_points_many_item_work_at_decide_batch(service):
    text = (await listed(service))["decide"].lower()
    assert "prefer decide_batch with path for many items" in text
    assert "so the items stay out of your context" in text


@pytest.mark.anyio
async def test_decide_batch_says_how_matches_are_ordered(service):
    text = (await listed(service))["decide_batch"].lower()
    assert "with a filter" in text
    assert "most confident first" in text
    assert "without a filter" in text
    assert "in input order" in text


@pytest.mark.anyio
async def test_decide_batch_applies_the_token_window_to_each_item_and_chunk(service):
    text = (await listed(service))["decide_batch"].lower()
    assert "each item is read like a decide state" in text
    assert "each chunk" in text
    assert "keep chunk_tokens below" in text


@pytest.mark.anyio
async def test_status_names_what_it_reports(service):
    text = (await listed(service))["status"].lower()
    assert [term for term in STATUS_TERMS if term not in text] == []


@pytest.mark.anyio
async def test_decide_returns_structured_jev_answers(service):
    async with Client(build_mcp_server(service)) as client:
        result = await client.call_tool("decide", {"state": "please check billing", "questions": Q})
    assert result.is_error is False
    assert result.structured_content["answers"]["dept"]["choice"] == "billing"


@pytest.mark.anyio
async def test_invalid_questions_become_tool_errors_with_hints(service):
    async with Client(build_mcp_server(service)) as client:
        result = await client.call_tool(
            "decide", {"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}}
        )
    assert result.is_error is True
    assert "invalid_request" in result.content[0].text
    assert "Hint:" in result.content[0].text


@pytest.mark.anyio
async def test_decide_batch_over_inline_items(service):
    async with Client(build_mcp_server(service)) as client:
        result = await client.call_tool(
            "decide_batch",
            {
                "questions": Q,
                "items": ["billing issue", "tech issue"],
                "filter": {"question": "dept", "op": "==", "value": "tech"},
            },
        )
    assert result.is_error is False
    assert result.structured_content["count"] == 2
    assert [match["id"] for match in result.structured_content["matches"]] == [2]


@pytest.mark.anyio
async def test_the_status_tool(service):
    async with Client(build_mcp_server(service)) as client:
        result = await client.call_tool("status", {})
    assert result.structured_content["backend"] == "fake"


@pytest.mark.anyio
async def test_a_lone_surrogate_in_an_error_message_is_escaped(service, tmp_path):
    # A caller can send "\ud800" as a JSON escape; the "no such file" message repeats the path.
    path = f"{tmp_path}/\ud800.jsonl"
    async with Client(build_mcp_server(service)) as client:
        result = await client.call_tool("decide_batch", {"questions": Q, "path": path})
    assert result.is_error is True
    text = result.content[0].text
    assert "no such file" in text
    assert "\\ud800.jsonl" in text
    text.encode("utf-8")  # a raw lone surrogate cannot be written to the wire
