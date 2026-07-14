"""Unit tests for composeai.mcp's pure helpers (no SDK, no subprocess)."""

from __future__ import annotations

from types import SimpleNamespace

from composeai.mcp import _render_result, _spec_from_mcp_tool


def test_render_result_joins_text_blocks():
    blocks = [
        SimpleNamespace(type="text", text="line one"),
        SimpleNamespace(type="text", text="line two"),
    ]
    assert _render_result(blocks, None) == "line one\nline two"


def test_render_result_text_wins_over_structured():
    # Text-first: when a text block is present, it's rendered even though
    # structuredContent is also present (FastMCP sends both).
    blocks = [SimpleNamespace(type="text", text="echo: hi")]
    assert _render_result(blocks, {"result": "echo: hi"}) == "echo: hi"


def test_render_result_structured_fallback_when_no_text():
    assert _render_result([], {"total": 3}) == '{"total": 3}'


def test_render_result_unsupported_block():
    blocks = [SimpleNamespace(type="image", data="...", mimeType="image/png")]
    assert _render_result(blocks, None) == "[unsupported MCP content block: image]"


def test_spec_from_mcp_tool_defaults():
    spec = _spec_from_mcp_tool(
        "read_file", None, {"type": "object"}, requires_approval=False
    )
    assert spec.name == "read_file"
    assert spec.description == "(no description provided by the MCP server)"
    assert spec.strict is False
    assert spec.requires_approval is False


def test_spec_from_mcp_tool_passthrough_schema():
    schema = {"type": "object", "properties": {"x": {"minimum": 3}}}
    spec = _spec_from_mcp_tool("t", "desc", schema, requires_approval=True)
    assert spec.input_schema == schema  # unmodified, not sealed
    assert spec.requires_approval is True
