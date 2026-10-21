"""examples/minimal_mcp_server.py through the SDK's in-memory client: tools, resources, prompts."""

import importlib.util
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "minimal_mcp_server.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("minimal_mcp_server", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    # Registered first, so the SDK can resolve the annotations the decorators read.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
