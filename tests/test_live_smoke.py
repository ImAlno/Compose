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
