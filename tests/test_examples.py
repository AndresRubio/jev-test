"""The two examples: the stdio server through the SDK's in-memory client, and the HTTP client
against the local-jev app with a fake engine."""

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from local_jev.app import create_app
from local_jev.service import DecisionService
from tests.fakes import FakeEngine
from tests.test_app import free_port, serving

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered first, so the SDK can resolve the annotations the decorators read.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return load("minimal_mcp_server")


@pytest.fixture(scope="module")
def example_client():
    return load("minimal_mcp_client")


@pytest.mark.anyio
async def test_the_server_announces_its_instructions_and_every_primitive(example):
    async with Client(example.mcp) as client:
        capabilities = client.server_capabilities
        assert client.instructions.startswith("A study server.")
    assert capabilities.tools is not None
    assert capabilities.resources is not None
    assert capabilities.prompts is not None


@pytest.mark.anyio
async def test_the_tool_has_an_input_schema_from_the_signature_and_an_output_schema(example):
    async with Client(example.mcp) as client:
        tools = (await client.list_tools()).tools
    assert [tool.name for tool in tools] == ["word_count"]
    assert tools[0].input_schema["required"] == ["text"]
    # Context is the server's, not an argument the model fills in.
    assert "ctx" not in tools[0].input_schema["properties"]
    assert tools[0].output_schema["type"] == "object"


@pytest.mark.anyio
async def test_the_tool_returns_structured_content(example):
    async with Client(example.mcp) as client:
        result = await client.call_tool("word_count", {"text": "hello big world"})
    assert result.is_error is False
    assert result.structured_content == {"words": 3, "characters": 15}


@pytest.mark.anyio
async def test_a_tool_error_reaches_the_model_as_text(example):
    async with Client(example.mcp) as client:
        result = await client.call_tool("word_count", {"text": "   "})
    assert result.is_error is True
    assert "text is empty" in result.content[0].text


@pytest.mark.anyio
async def test_static_resources_and_templates_are_listed_apart(example):
    async with Client(example.mcp) as client:
        resources = (await client.list_resources()).resources
        templates = (await client.list_resource_templates()).resource_templates
    assert [str(resource.uri) for resource in resources] == ["study://notes"]
    assert [template.uri_template for template in templates] == ["study://notes/{topic}"]


@pytest.mark.anyio
async def test_a_template_parameter_comes_from_the_uri(example):
    async with Client(example.mcp) as client:
        result = await client.read_resource("study://notes/transports")
    assert result.contents[0].text.startswith("stdio for a server the client starts")


@pytest.mark.anyio
async def test_the_sdk_refuses_path_traversal_in_a_template_parameter(example):
    async with Client(example.mcp) as client:
        with pytest.raises(MCPError, match="Unknown resource"):
            await client.read_resource("study://notes/../secret")


@pytest.mark.anyio
async def test_the_prompt_lists_its_arguments_and_renders_them(example):
    async with Client(example.mcp) as client:
        prompts = (await client.list_prompts()).prompts
        rendered = await client.get_prompt("review", {"code": "x = 1"})
    assert [(argument.name, argument.required) for argument in prompts[0].arguments] == [
        ("code", True),
        ("focus", False),
    ]
    assert rendered.messages[0].role == "user"
    assert rendered.messages[0].content.text.startswith("Review this code for correctness.")


# The client example, against the real app over HTTP.


@pytest.fixture
def served(settings):
    """The app with a fake engine on a free port; yields the MCP URL."""
    settings = dataclasses.replace(settings, port=free_port())
    app = create_app(settings, service=DecisionService(settings, FakeEngine()))
    with serving(app, settings.port) as base:
        yield f"{base}/mcp"


@pytest.mark.anyio
async def test_the_client_lists_the_tools_and_calls_decide(example_client, served, capsys):
    result = await example_client.run(served, "tech outage: the site is down")
    assert result["answers"]["team"]["choice"] == "tech"
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "protocol 2026-07-28, server local-jev"
    assert out[1:4] == [
        "tool decide(state, questions)",
        "tool decide_batch(questions)",
        "tool status()",
    ]
    assert out[4].startswith("team tech {")


@pytest.mark.anyio
async def test_the_token_goes_out_as_a_bearer_header(example_client, settings):
    settings = dataclasses.replace(settings, port=free_port(), token="s3cret")
    app = create_app(settings, service=DecisionService(settings, FakeEngine()))
    with serving(app, settings.port) as base:
        # The SDK turns any HTTP error but 404 into a JSON-RPC internal error; the 401 is lost.
        with pytest.raises(ExceptionGroup) as denied:
            await example_client.run(f"{base}/mcp", "tech outage")
        assert denied.group_contains(MCPError, match="Server returned an error response")
        result = await example_client.run(f"{base}/mcp", "tech outage", token="s3cret")
    assert result["answers"]["team"]["choice"] == "tech"
