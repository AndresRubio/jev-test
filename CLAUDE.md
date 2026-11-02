# local-jev

Python MCP server on `mcp==2.3.0` (pinned in `pyproject.toml`), Starlette and laya-mlx. Tests: `uv run pytest` (no network, no weights).

- `docs/mcp-study-guide.md`: how local-jev speaks MCP, with the file list to read in order.
- `docs/mcp-sdk-notes.md`: SDK behaviour the repo depends on, each fact with the test that pins it. Read it before opening the SDK source.
- `CODING_STANDARDS.md`: what review checks, including the rule for claims about the SDK.
