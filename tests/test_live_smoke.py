"""Live smoke tests against the real Anthropic and OpenAI APIs.

Skipped unless ``COMPOSE_LIVE_TESTS=1`` is set, so the normal offline test
run never needs network access or an API key. Requires ``ANTHROPIC_API_KEY``
and/or ``OPENAI_API_KEY`` in the environment when actually run.
"""

from __future__ import annotations

import os

import pytest

from composeai.messages import Message
from composeai.models.anthropic import AnthropicModel
from composeai.models.base import ModelRequest
from composeai.models.openai import OpenAIModel

pytestmark = pytest.mark.skipif(
    os.environ.get("COMPOSE_LIVE_TESTS") != "1",
    reason="live smoke tests only run with COMPOSE_LIVE_TESTS=1 (and a real API key)",
)


def test_live_tiny_text_completion():
    model = AnthropicModel("claude-haiku-4-5")
    request = ModelRequest(
        model="claude-haiku-4-5",
        messages=[Message.user("Reply with exactly the word: pong")],
        max_tokens=16,
    )
    response = model.complete(request)
    assert response.message.text.strip()
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


def test_live_structured_output():
    model = AnthropicModel("claude-haiku-4-5")
    request = ModelRequest(
        model="claude-haiku-4-5",
        messages=[Message.user("Give me the number 7 as JSON: {\"n\": <number>}")],
        output_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
        max_tokens=64,
    )
    response = model.complete(request)
    assert response.parsed is not None
    assert response.parsed["n"] == 7


def test_live_openai_tiny_text_completion():
    model = OpenAIModel("gpt-5.4-nano")
    request = ModelRequest(
        model="gpt-5.4-nano",
        messages=[Message.user("Reply with exactly the word: pong")],
        max_tokens=16,
    )
    response = model.complete(request)
    assert response.message.text.strip()
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


def test_live_openai_structured_output():
    model = OpenAIModel("gpt-5.4-nano")
    request = ModelRequest(
        model="gpt-5.4-nano",
        messages=[Message.user("Give me the number 7 as JSON: {\"n\": <number>}")],
        output_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
        max_tokens=64,
    )
    response = model.complete(request)
    assert response.parsed is not None
    assert response.parsed["n"] == 7


def test_live_openai_compatible_prompt_mode():
    """Gated: the module-level ``pytestmark`` above already requires
    ``COMPOSE_LIVE_TESTS=1``; this test additionally needs
    ``COMPOSE_LIVE_COMPAT_BASE_URL`` (e.g. a local Ollama at
    http://localhost:11434/v1) AND ``COMPOSE_LIVE_COMPAT_MODEL``. Verifies
    both native and prompt schema_mode against a real server end-to-end.
    """
    base_url = os.environ.get("COMPOSE_LIVE_COMPAT_BASE_URL")
    model_id = os.environ.get("COMPOSE_LIVE_COMPAT_MODEL")
    if not base_url or not model_id:
        pytest.skip("set COMPOSE_LIVE_COMPAT_BASE_URL and COMPOSE_LIVE_COMPAT_MODEL")

    from pydantic import BaseModel

    from composeai import agent, openai_compatible, prompt

    class LiveAnswer(BaseModel):
        answer: str

    for mode in ("native", "prompt"):
        model = openai_compatible(base_url, model_id, schema_mode=mode, timeout=120)

        @agent(model=model, name=f"live_compat_{mode}", max_repairs=1)
        def live_asker(question: str) -> LiveAnswer:
            """Answer in one word."""
            return prompt(question)

        result = live_asker("What color is the sky on a clear day?")
        assert isinstance(result, LiveAnswer)


def test_live_mcp_server_lists_and_echoes():
    """Extra gate: COMPOSE_LIVE_MCP_COMMAND, e.g.
    'npx -y @modelcontextprotocol/server-everything' -- any server whose
    first listed tool takes no required args or a single string arg."""
    import shlex

    command_line = os.environ.get("COMPOSE_LIVE_MCP_COMMAND")
    if not command_line:
        pytest.skip("set COMPOSE_LIVE_MCP_COMMAND to run the MCP live smoke")

    from composeai import mcp_tools

    parts = shlex.split(command_line)
    tools = mcp_tools(command=parts[0], args=parts[1:], connect_timeout=60)
    assert tools, "server listed no tools"
    names = [t.name for t in tools]
    assert all(isinstance(n, str) and n for n in names)
