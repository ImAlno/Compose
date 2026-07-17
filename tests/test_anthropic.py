"""Tests for the Anthropic adapter, using a duck-typed fake client (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from conformance import contract

from composeai.errors import ComposeError, ConfigError, ProviderError
from composeai.messages import (
    ImagePart,
    Message,
    StopReason,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from composeai.models.anthropic import AnthropicModel
from composeai.models.base import ModelRequest, ToolSpec
from composeai.models.prices import ModelPrice, register_price

# --- Stub client helpers -----------------------------------------------------


def _usage(
    input_tokens: int = 10,
    output_tokens: int = 20,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_creation: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_creation=cache_creation,
    )


def _cache_creation(ephemeral_5m_input_tokens: int = 0, ephemeral_1h_input_tokens: int = 0):
    return SimpleNamespace(
        ephemeral_5m_input_tokens=ephemeral_5m_input_tokens,
        ephemeral_1h_input_tokens=ephemeral_1h_input_tokens,
    )


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(id: str, name: str, input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _thinking_block(thinking: str, signature: str | None = "sig-1") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=thinking, signature=signature)


def _response(content, stop_reason: str = "end_turn", usage: SimpleNamespace | None = None):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=usage if usage is not None else _usage(),
    )


class _StubMessages:
    """Duck-typed stand-in for ``anthropic.Anthropic().messages``."""

    def __init__(self, responses, stream_responses: Any = None) -> None:
        # `responses`/`stream_responses` is either a list consumed in order,
        # or a callable invoked with the call kwargs each time (for
        # open-ended stubs).
        self._responses = responses
        self._stream_responses = stream_responses
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if callable(self._responses):
            return self._responses(kwargs)
        return self._responses.pop(0)

    def stream(self, **kwargs: Any):
        self.stream_calls.append(kwargs)
        if callable(self._stream_responses):
            return self._stream_responses(kwargs)
        return self._stream_responses.pop(0)


class _StubClient:
    def __init__(self, responses, stream_responses: Any = None) -> None:
        self.messages = _StubMessages(responses, stream_responses)


def _model(responses, model_id: str = "claude-sonnet-5") -> AnthropicModel:
    return AnthropicModel(model_id, client=_StubClient(responses))


class _AsyncStubMessages:
    """Async twin of ``_StubMessages``: duck-typed stand-in for
    ``anthropic.AsyncAnthropic().messages`` (``create`` is a coroutine;
    ``stream`` stays a plain method that returns an async context manager,
    mirroring the real SDK's ``AsyncMessages.stream``)."""

    def __init__(self, responses, stream_responses: Any = None) -> None:
        self._responses = responses
        self._stream_responses = stream_responses
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if callable(self._responses):
            return self._responses(kwargs)
        return self._responses.pop(0)

    def stream(self, **kwargs: Any):
        self.stream_calls.append(kwargs)
        if callable(self._stream_responses):
            return self._stream_responses(kwargs)
        return self._stream_responses.pop(0)


class _AsyncStubClient:
    def __init__(self, responses, stream_responses: Any = None) -> None:
        self.messages = _AsyncStubMessages(responses, stream_responses)


def _amodel(responses, model_id: str = "claude-sonnet-5") -> AnthropicModel:
    model = AnthropicModel(model_id)
    model._async_client = _AsyncStubClient(responses)
    return model


# --- stream() stub helpers ----------------------------------------------------


class _StubMessageStream:
    """Duck-typed stand-in for ``anthropic.Anthropic().messages.stream()``'s
    context-manager result (``MessageStream``): iterates raw SSE-shaped
    events (as real ``anthropic`` yields them: raw ``content_block_start`` /
    ``content_block_delta`` / ``content_block_stop`` interleaved with the
    SDK's own derived per-kind events, which our adapter must tolerate and
    ignore) and exposes ``get_final_message()``.
    """

    def __init__(self, raw_events, final_message) -> None:
        self._raw_events = raw_events
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._raw_events)

    def get_final_message(self):
        return self._final_message


def _cb_start(index: int, block) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_start", index=index, content_block=block)


def _cb_delta(index: int, delta) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", index=index, delta=delta)


def _cb_stop(index: int) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_stop", index=index)


def _text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text_delta", text=text)


def _thinking_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking_delta", thinking=text)


def _input_json_delta(partial_json: str) -> SimpleNamespace:
    return SimpleNamespace(type="input_json_delta", partial_json=partial_json)


def _derived_text_event(text: str, snapshot: str) -> SimpleNamespace:
    # One of the SDK's own *derived* convenience events, interleaved into the
    # same iteration alongside the raw events above -- adapters must ignore
    # event types they don't recognize rather than mis-map them.
    return SimpleNamespace(type="text", text=text, snapshot=snapshot)


def _model_stream(stream_responses, model_id: str = "claude-sonnet-5") -> AnthropicModel:
    return AnthropicModel(model_id, client=_StubClient([], stream_responses))


class _AsyncStubMessageStream:
    """Async twin of ``_StubMessageStream``: an async-context-manager whose
    ``__aenter__`` returns an async-iterable over the same scripted raw
    events, with an awaitable ``get_final_message()`` (mirroring the real
    SDK's ``AsyncMessageStream``).
    """

    def __init__(self, raw_events, final_message) -> None:
        self._raw_events = raw_events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def __aiter__(self):
        for event in self._raw_events:
            yield event

    async def get_final_message(self):
        return self._final_message


def _amodel_stream(stream_responses, model_id: str = "claude-sonnet-5") -> AnthropicModel:
    model = AnthropicModel(model_id)
    model._async_client = _AsyncStubClient([], stream_responses)
    return model


# --- constructor / lazy client / missing key --------------------------------


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = AnthropicModel("claude-sonnet-5")
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        model.complete(request)


def test_injected_client_bypasses_key_requirement(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = _model([_response([_text_block("hi")])])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    response = model.complete(request)
    assert response.message.text == "hi"


def test_provider_sdk_client_not_built_when_client_injected():
    # No API key set anywhere, no real anthropic.Anthropic() construction
    # should be attempted since a client was injected.
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    model.complete(request)  # must not raise


# --- request mapping: system / temperature ----------------------------------


def test_system_omitted_when_none():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert "system" not in call


def test_system_included_when_set():
    responses = [_response([_text_block("ok")])]
    model = _model(responses)
    request = ModelRequest(
        model="claude-sonnet-5", messages=[Message.user("hi")], system="be terse"
    )
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["system"] == "be terse"


def test_temperature_omitted_when_none():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert "temperature" not in call


def test_temperature_passed_through_when_set_even_for_modern_models():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(
        model="claude-sonnet-5", messages=[Message.user("hi")], temperature=0.7
    )
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["temperature"] == 0.7


def test_max_tokens_passed_through():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")], max_tokens=42)
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["max_tokens"] == 42


# --- request mapping: message/content conversion ----------------------------


def test_text_part_maps_to_text_block():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hello")])
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["messages"][0] == {"role": "user", "content": [{"type": "text", "text": "hello"}]}


def test_image_part_base64_maps_to_base64_source():
    model = _model([_response([_text_block("ok")])])
    image = ImagePart(media_type="image/png", data="YWJj")
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user([image])])
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    block = call["messages"][0]["content"][0]
    assert block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"},
    }


def test_image_part_url_maps_to_url_source():
    model = _model([_response([_text_block("ok")])])
    image = ImagePart(media_type="image/png", url="https://example.com/x.png")
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user([image])])
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    block = call["messages"][0]["content"][0]
    assert block == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/x.png"},
    }


def test_tool_call_part_maps_to_tool_use_block():
    model = _model([_response([_text_block("ok")])])
    call_part = ToolCallPart(id="tc1", name="search", arguments={"q": "cats"})
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.assistant([call_part])])
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    block = call["messages"][0]["content"][0]
    assert block == {"type": "tool_use", "id": "tc1", "name": "search", "input": {"q": "cats"}}


def test_tool_result_part_maps_without_is_error_when_false():
    model = _model([_response([_text_block("ok")])])
    result = ToolResultPart(tool_call_id="tc1", content="42")
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user([result])])
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    block = call["messages"][0]["content"][0]
    assert block == {"type": "tool_result", "tool_use_id": "tc1", "content": "42"}
    assert "is_error" not in block


def test_tool_result_part_includes_is_error_when_true():
    model = _model([_response([_text_block("ok")])])
    result = ToolResultPart(tool_call_id="tc1", content="boom", is_error=True)
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user([result])])
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    block = call["messages"][0]["content"][0]
    assert block["is_error"] is True


def test_mixed_tool_result_and_other_content_in_one_message_raises():
    # Minor-findings cleanup (Phase 10): this is a caller bug (batch all tool
    # results for one turn into their own message), not a provider/SDK
    # failure -- so it's a ComposeError, not a ProviderError.
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[
            Message.user(
                [
                    ToolResultPart(tool_call_id="tc1", content="42"),
                    TextPart(text="oh also,"),
                ]
            )
        ],
    )
    with pytest.raises(ComposeError) as excinfo:
        model.complete(request)
    assert not isinstance(excinfo.value, ProviderError)


def test_multiple_tool_results_in_one_message_stay_together():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[
            Message.user(
                [
                    ToolResultPart(tool_call_id="tc1", content="a"),
                    ToolResultPart(tool_call_id="tc2", content="b"),
                ]
            )
        ],
    )
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert len(call["messages"]) == 1
    assert len(call["messages"][0]["content"]) == 2


# --- request mapping: tools / structured output ------------------------------


def test_tools_mapped_with_strict_and_input_schema():
    model = _model([_response([_text_block("ok")])])
    spec = ToolSpec(
        name="search",
        description="search the web",
        input_schema={"type": "object", "additionalProperties": False},
        strict=True,
    )
    request = ModelRequest(
        model="claude-sonnet-5", messages=[Message.user("hi")], tools=[spec]
    )
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["tools"] == [
        {
            "name": "search",
            "description": "search the web",
            "input_schema": {"type": "object", "additionalProperties": False},
            "strict": True,
        }
    ]


def test_tools_omitted_when_none():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert "tools" not in call


def test_output_schema_maps_to_output_config():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    model = _model([_response([_text_block('{"a": 1}')])])
    request = ModelRequest(
        model="claude-sonnet-5", messages=[Message.user("hi")], output_schema=schema
    )
    resp = model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    assert resp.parsed == {"a": 1}


def test_output_schema_can_coexist_with_tools():
    schema = {"type": "object"}
    spec = ToolSpec(name="f", description="d", input_schema={"type": "object"})
    model = _model([_response([_text_block('{"a": 1}')])])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("hi")],
        tools=[spec],
        output_schema=schema,
    )
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert "tools" in call
    assert "output_config" in call


def test_non_json_text_with_output_schema_raises_provider_error():
    model = _model([_response([_text_block("not json")])])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("hi")],
        output_schema={"type": "object"},
    )
    with pytest.raises(ProviderError):
        model.complete(request)


def test_tool_use_stop_with_output_schema_does_not_crash_and_parsed_is_none():
    # Regression: an @agent combining tools=[...] with a non-str output_type
    # sends output_schema on every turn (agentfn.py's _aperform_turn), so a
    # tool_use-only turn has empty message.text -- json.loads('') must not
    # be attempted (it would raise and mask the real TOOL_USE stop reason
    # behind a spurious ProviderError).
    model = _model(
        [_response([_tool_use_block("tu1", "search", {})], stop_reason="tool_use")]
    )
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("hi")],
        output_schema={"type": "object"},
    )
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.TOOL_USE
    assert resp.parsed is None


def test_refusal_stop_with_output_schema_does_not_crash_and_parsed_is_none():
    model = _model([_response([_text_block("I can't help with that")], stop_reason="refusal")])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("hi")],
        output_schema={"type": "object"},
    )
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.REFUSAL
    assert resp.parsed is None


def test_max_tokens_stop_with_output_schema_does_not_crash_and_parsed_is_none():
    model = _model([_response([_text_block('{"partial": tr')], stop_reason="max_tokens")])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("hi")],
        output_schema={"type": "object"},
    )
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.MAX_TOKENS
    assert resp.parsed is None


# --- response mapping: content blocks ---------------------------------------


def test_text_block_maps_to_text_part():
    model = _model([_response([_text_block("hello there")])])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.message.text == "hello there"


def test_tool_use_block_maps_to_tool_call_part():
    model = _model(
        [_response([_tool_use_block("tu1", "search", {"q": "cats"})], stop_reason="tool_use")]
    )
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp = model.complete(request)
    calls = [p for p in resp.message.parts if isinstance(p, ToolCallPart)]
    assert len(calls) == 1
    assert calls[0].id == "tu1"
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "cats"}
    assert resp.stop_reason == StopReason.TOOL_USE


def test_thinking_block_maps_to_thinking_part():
    model = _model([_response([_thinking_block("pondering...", signature="sig-xyz")])])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp = model.complete(request)
    thinking_parts = [p for p in resp.message.parts if isinstance(p, ThinkingPart)]
    assert len(thinking_parts) == 1
    part = thinking_parts[0]
    assert part.text == "pondering..."
    assert part.provider == "anthropic"
    assert part.provider_token is not None
    payload = json.loads(part.provider_token)
    assert payload["model"] == "claude-sonnet-5"
    assert payload["block"]["thinking"] == "pondering..."
    assert payload["block"]["signature"] == "sig-xyz"


# --- response mapping: stop reasons ------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("end_turn", StopReason.END_TURN),
        ("tool_use", StopReason.TOOL_USE),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.REFUSAL),
        ("stop_sequence", StopReason.OTHER),
        ("some_future_reason", StopReason.OTHER),
    ],
)
def test_stop_reason_mapping_table(raw, expected):
    model = _model([_response([_text_block("x")], stop_reason=raw)])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.stop_reason == expected
    assert resp.raw_stop_reason == raw


# --- response mapping: usage / cost ------------------------------------------


def test_usage_tokens_land_in_usage():
    usage = _usage(input_tokens=123, output_tokens=45)
    model = _model([_response([_text_block("x")], usage=usage)])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.input_tokens == 123
    assert resp.usage.output_tokens == 45


def test_usage_cost_computed_for_known_model():
    big_usage = _usage(input_tokens=1_000_000, output_tokens=1_000_000)
    model = _model(
        [_response([_text_block("x")], usage=big_usage)],
        model_id="claude-sonnet-5",
    )
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cost_usd == pytest.approx(3.0 + 15.0)
    assert resp.usage.cost_complete is True


def test_usage_cost_none_and_incomplete_for_unknown_model():
    model = _model([_response([_text_block("x")])], model_id="claude-unknown-9000")
    request = ModelRequest(model="claude-unknown-9000", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cost_usd is None
    assert resp.usage.cost_complete is False


def test_cache_creation_breakdown_used_when_present():
    register_price("anthropic", "cache-test-model", ModelPrice(input=10.0, output=50.0))
    usage = _usage(
        cache_creation=_cache_creation(
            ephemeral_5m_input_tokens=1_000_000, ephemeral_1h_input_tokens=1_000_000
        )
    )
    model = _model([_response([_text_block("x")], usage=usage)], model_id="cache-test-model")
    request = ModelRequest(model="cache-test-model", messages=[Message.user("hi")])
    resp = model.complete(request)
    # cache_write_5m = 12.5/MTok, cache_write_1h = 20.0/MTok (defaults off input=10)
    expected_cache_cost = 12.5 + 20.0
    input_output_cost = usage.input_tokens * 10.0 / 1e6 + usage.output_tokens * 50.0 / 1e6
    assert resp.usage.cost_usd == pytest.approx(expected_cache_cost + input_output_cost)


def test_cache_creation_absent_treated_as_all_5m():
    register_price("anthropic", "cache-test-model-2", ModelPrice(input=10.0, output=50.0))
    usage = _usage(cache_creation_input_tokens=1_000_000, cache_creation=None)
    model = _model([_response([_text_block("x")], usage=usage)], model_id="cache-test-model-2")
    request = ModelRequest(model="cache-test-model-2", messages=[Message.user("hi")])
    resp = model.complete(request)
    input_output_cost = usage.input_tokens * 10.0 / 1e6 + usage.output_tokens * 50.0 / 1e6
    assert resp.usage.cost_usd == pytest.approx(12.5 + input_output_cost)


def test_cache_read_tokens_land_in_usage():
    model = _model([_response([_text_block("x")], usage=_usage(cache_read_input_tokens=99))])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cache_read_tokens == 99


def test_cache_read_tokens_are_billed_additively_not_subtracted_from_input():
    # Companion to the OpenAI/openai-compatible "double-billed cached input
    # tokens" fix: verified against the installed anthropic SDK's
    # `types/usage.py` that `input_tokens` ("The number of input tokens
    # which were used") and `cache_read_input_tokens` ("The number of input
    # tokens read from the cache") are documented as separate, sibling
    # fields on `Usage` -- not one nested inside a "breakdown of" the other
    # the way OpenAI's `input_tokens_details.cached_tokens` is (see
    # models/openai.py's `_map_usage` comment). This is also how the SDK's
    # own streaming accumulator (`anthropic/lib/streaming/_messages.py`)
    # treats them: `input_tokens`, `cache_creation_input_tokens`, and
    # `cache_read_input_tokens` are each independently copied from the
    # event's usage with no subtraction between them. So, unlike OpenAI,
    # Anthropic's `input_tokens` EXCLUDES cache reads/writes -- billing
    # `input_tokens * input_price + cache_read_tokens * cache_read_price`
    # (no subtraction) is already correct here, and must stay that way.
    register_price(
        "anthropic", "cache-billing-test-model", ModelPrice(input=10.0, output=50.0, cache_read=1.0)
    )
    usage = _usage(input_tokens=200_000, output_tokens=100_000, cache_read_input_tokens=800_000)
    model = _model(
        [_response([_text_block("x")], usage=usage)], model_id="cache-billing-test-model"
    )
    request = ModelRequest(model="cache-billing-test-model", messages=[Message.user("hi")])
    resp = model.complete(request)
    expected = (200_000 * 10.0 + 800_000 * 1.0 + 100_000 * 50.0) / 1_000_000
    assert resp.usage.cost_usd == pytest.approx(expected)
    # Sanity: if this ever regressed to subtracting cache_read_tokens from
    # input_tokens the way OpenAI's fix does, the (wrong, smaller) result
    # would be -60_000*10.0 + 800_000*1.0 + 100_000*50.0 all over 1e6 (i.e.
    # a negative billable-input contribution) -- visibly different from the
    # correct additive total.
    assert expected == pytest.approx(7.8)


def test_reasoning_tokens_read_from_output_tokens_details():
    # The SDK reports raw thinking spend in
    # usage.output_tokens_details.thinking_tokens (a subset of output_tokens);
    # map it onto Usage.reasoning_tokens now that thinking is requestable.
    usage = _usage(output_tokens=45)
    usage.output_tokens_details = SimpleNamespace(thinking_tokens=7)
    model = _model([_response([_text_block("x")], usage=usage)])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.reasoning_tokens == 7


def test_reasoning_tokens_default_zero_without_output_tokens_details():
    # Regression guard: older usage payloads omit output_tokens_details
    # entirely -- reasoning_tokens must default to 0 rather than raise.
    usage = _usage(output_tokens=45)
    assert not hasattr(usage, "output_tokens_details")
    model = _model([_response([_text_block("x")], usage=usage)])
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.reasoning_tokens == 0


# --- thinking-token echo ------------------------------------------------------


def test_thinking_part_echoed_verbatim_when_same_model():
    model = _model(
        [
            _response([_thinking_block("first thought", signature="sig-a"), _text_block("done")]),
            _response([_text_block("continuing")]),
        ],
        model_id="claude-sonnet-5",
    )
    request1 = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp1 = model.complete(request1)

    request2 = ModelRequest(
        model="claude-sonnet-5",
        messages=[
            Message.user("hi"),
            Message.assistant(resp1.message.parts),
            Message.user("continue"),
        ],
    )
    model.complete(request2)
    call2 = model._client.messages.calls[1]  # pyright: ignore[reportAttributeAccessIssue]
    assistant_msg = call2["messages"][1]
    thinking_blocks = [b for b in assistant_msg["content"] if b["type"] == "thinking"]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0]["thinking"] == "first thought"
    assert thinking_blocks[0]["signature"] == "sig-a"


def test_thinking_part_dropped_when_different_model():
    producer = _model(
        [_response([_thinking_block("first thought", signature="sig-a"), _text_block("done")])],
        model_id="claude-sonnet-5",
    )
    request1 = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp1 = producer.complete(request1)

    consumer = _model([_response([_text_block("continuing")])], model_id="claude-haiku-4-5")
    request2 = ModelRequest(
        model="claude-haiku-4-5",
        messages=[
            Message.user("hi"),
            Message.assistant(resp1.message.parts),
            Message.user("continue"),
        ],
    )
    consumer.complete(request2)
    call = consumer._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assistant_msg = call["messages"][1]
    thinking_blocks = [b for b in assistant_msg["content"] if b["type"] == "thinking"]
    assert thinking_blocks == []


def test_thinking_part_dropped_when_different_provider():
    other_provider_part = ThinkingPart(text="hmm", provider="openai", provider_token="{}")
    model = _model([_response([_text_block("ok")])], model_id="claude-sonnet-5")
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.assistant([other_provider_part])],
    )
    model.complete(request)
    call = model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["messages"][0]["content"] == []


def test_thinking_part_with_missing_signature_omits_signature_key_when_echoed():
    # Minor-findings cleanup (Phase 10): a thinking block with no signature
    # (`signature=None`) must not round-trip as an explicit `"signature":
    # null` -- strip None-valued keys before re-sending (a live API 400 risk).
    model = _model(
        [
            _response([_thinking_block("first thought", signature=None), _text_block("done")]),
            _response([_text_block("continuing")]),
        ],
        model_id="claude-sonnet-5",
    )
    request1 = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    resp1 = model.complete(request1)

    request2 = ModelRequest(
        model="claude-sonnet-5",
        messages=[
            Message.user("hi"),
            Message.assistant(resp1.message.parts),
            Message.user("continue"),
        ],
    )
    model.complete(request2)
    call2 = model._client.messages.calls[1]  # pyright: ignore[reportAttributeAccessIssue]
    assistant_msg = call2["messages"][1]
    thinking_blocks = [b for b in assistant_msg["content"] if b["type"] == "thinking"]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0]["thinking"] == "first thought"
    assert "signature" not in thinking_blocks[0]


def test_malformed_thinking_provider_token_raises_provider_error():
    # Minor-findings cleanup (Phase 10): a hand-crafted ThinkingPart with
    # garbage JSON in provider_token must not surface a raw JSONDecodeError.
    bad_part = ThinkingPart(text="hmm", provider="anthropic", provider_token="{not valid json")
    model = _model([_response([_text_block("ok")])], model_id="claude-sonnet-5")
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.assistant([bad_part])],
    )
    with pytest.raises(ProviderError):
        model.complete(request)


# --- pause_turn continuation loop --------------------------------------------


def test_pause_turn_continues_and_returns_single_merged_response():
    model = _model(
        [
            _response([_text_block("part one")], stop_reason="pause_turn"),
            _response([_text_block("part two")], stop_reason="pause_turn"),
            _response([_text_block("part three")], stop_reason="end_turn"),
        ]
    )
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("go")])
    resp = model.complete(request)
    assert len(model._client.messages.calls) == 3  # pyright: ignore[reportAttributeAccessIssue]
    assert resp.stop_reason == StopReason.END_TURN
    assert resp.raw_stop_reason == "end_turn"
    assert resp.message.text == "part onepart twopart three"


def test_pause_turn_sums_usage_across_continuations():
    first_usage = _usage(input_tokens=10, output_tokens=5)
    second_usage = _usage(input_tokens=20, output_tokens=7)
    model = _model(
        [
            _response([_text_block("a")], stop_reason="pause_turn", usage=first_usage),
            _response([_text_block("b")], stop_reason="end_turn", usage=second_usage),
        ]
    )
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("go")])
    resp = model.complete(request)
    assert resp.usage.input_tokens == 30
    assert resp.usage.output_tokens == 12


def test_pause_turn_exceeding_limit_raises_provider_error():
    def always_pause(_kwargs):
        return _response([_text_block("still going")], stop_reason="pause_turn")

    model = _model(always_pause)
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("go")])
    with pytest.raises(ProviderError):
        model.complete(request)
    assert len(model._client.messages.calls) == 6  # pyright: ignore[reportAttributeAccessIssue]


def test_pause_turn_never_surfaces_as_a_stop_reason():
    model = _model(
        [
            _response([_text_block("a")], stop_reason="pause_turn"),
            _response([_text_block("b")], stop_reason="end_turn"),
        ]
    )
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("go")])
    resp = model.complete(request)
    assert resp.stop_reason != StopReason.OTHER or resp.raw_stop_reason != "pause_turn"
    assert resp.raw_stop_reason == "end_turn"


# --- SDK error mapping --------------------------------------------------------


def _httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def test_api_status_error_becomes_provider_error():
    req = _httpx_request()
    resp = httpx.Response(500, request=req, json={"error": {"message": "server exploded"}})
    sdk_error = anthropic.APIStatusError("server exploded", response=resp, body=None)

    def raiser(_kwargs):
        raise sdk_error

    model = _model(raiser, model_id="claude-sonnet-5")
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    with pytest.raises(ProviderError) as excinfo:
        model.complete(request)
    assert excinfo.value.provider == "anthropic"
    assert excinfo.value.model == "claude-sonnet-5"
    assert excinfo.value.__cause__ is sdk_error


def test_api_connection_error_becomes_provider_error():
    sdk_error = anthropic.APIConnectionError(message="connection boom", request=_httpx_request())

    def raiser(_kwargs):
        raise sdk_error

    model = _model(raiser)
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    with pytest.raises(ProviderError) as excinfo:
        model.complete(request)
    assert excinfo.value.__cause__ is sdk_error


def test_no_retry_loop_added_on_top_of_sdk():
    # A single SDK exception should surface as exactly one failed call --
    # composeai must not add its own retry loop on top of the SDK's.
    calls = []

    def raiser(_kwargs):
        calls.append(1)
        raise anthropic.APIConnectionError(message="boom", request=_httpx_request())

    model = _model(raiser)
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    with pytest.raises(ProviderError):
        model.complete(request)
    assert len(calls) == 1


# --- stream() ------------------------------------------------------------


def test_stream_yields_text_deltas_then_response_done():
    events = [
        _cb_start(0, _text_block("")),
        _cb_delta(0, _text_delta("hello")),
        _derived_text_event("hello", "hello"),
        _cb_delta(0, _text_delta(" there")),
        _cb_stop(0),
    ]
    final = _response([_text_block("hello there")])
    model = _model_stream([_StubMessageStream(events, final)])

    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    raw_events = list(model.stream(request))

    kinds = [e.kind for e in raw_events]
    assert kinds == ["text_delta", "text_delta", "response_done"]
    assert [e.text for e in raw_events[:2]] == ["hello", " there"]
    assert raw_events[-1].response is not None
    assert raw_events[-1].response.message.text == "hello there"
    assert raw_events[-1].response.stop_reason == StopReason.END_TURN


def test_stream_tool_use_yields_started_args_delta_finished():
    events = [
        _cb_start(0, _tool_use_block("tu1", "search", {})),
        _cb_delta(0, _input_json_delta('{"q": ')),
        _cb_delta(0, _input_json_delta('"cats"}')),
        _cb_stop(0),
    ]
    final = _response(
        [_tool_use_block("tu1", "search", {"q": "cats"})], stop_reason="tool_use"
    )
    model = _model_stream([_StubMessageStream(events, final)])

    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    raw_events = list(model.stream(request))

    kinds = [e.kind for e in raw_events]
    assert kinds == [
        "tool_call_started",
        "tool_args_delta",
        "tool_args_delta",
        "tool_call_finished",
        "response_done",
    ]
    started, delta1, delta2, finished, done = raw_events
    assert started.tool_call_id == "tu1"
    assert started.tool_name == "search"
    assert delta1.text == '{"q": '
    assert delta1.tool_call_id == "tu1"
    assert delta1.tool_name == "search"
    assert delta2.text == '"cats"}'
    assert finished.tool_call_id == "tu1"
    assert finished.tool_name == "search"
    assert done.response is not None
    assert done.response.stop_reason == StopReason.TOOL_USE
    calls = [p for p in done.response.message.parts if isinstance(p, ToolCallPart)]
    assert calls[0].arguments == {"q": "cats"}


def test_stream_thinking_delta_maps_and_content_block_start_yields_nothing():
    events = [
        _cb_start(0, _thinking_block("")),
        _cb_delta(0, _thinking_delta("pondering")),
        _cb_stop(0),
    ]
    final = _response([_thinking_block("pondering", signature="sig-1"), _text_block("done")])
    model = _model_stream([_StubMessageStream(events, final)])

    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    raw_events = list(model.stream(request))

    kinds = [e.kind for e in raw_events]
    # content_block_start for a thinking block yields nothing on its own.
    assert kinds == ["thinking_delta", "response_done"]
    assert raw_events[0].text == "pondering"


def test_stream_response_done_equals_complete_style_mapping():
    content = [_text_block("hello there")]
    # complete() path
    complete_model = _model([_response(content)])
    complete_resp = complete_model.complete(
        ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    )
    # stream() path, same eventual content
    events = [_cb_start(0, _text_block("")), _cb_delta(0, _text_delta("hello there")), _cb_stop(0)]
    stream_model = _model_stream([_StubMessageStream(events, _response(content))])
    raw_events = list(
        stream_model.stream(ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")]))
    )
    streamed_resp = raw_events[-1].response
    assert streamed_resp is not None

    assert streamed_resp.message.text == complete_resp.message.text
    assert streamed_resp.stop_reason == complete_resp.stop_reason
    assert streamed_resp.usage == complete_resp.usage


def test_stream_records_request_shape_via_stub_calls():
    events = [_cb_start(0, _text_block("")), _cb_delta(0, _text_delta("x")), _cb_stop(0)]
    final = _response([_text_block("x")])
    model = _model_stream([_StubMessageStream(events, final)])
    request = ModelRequest(
        model="claude-sonnet-5", messages=[Message.user("hi")], system="be terse"
    )
    list(model.stream(request))
    call = model._client.messages.stream_calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["system"] == "be terse"
    assert call["model"] == "claude-sonnet-5"


def test_stream_pause_turn_continuation_yields_deltas_across_calls_and_merges_response():
    first_events = [
        _cb_start(0, _text_block("")),
        _cb_delta(0, _text_delta("part one")),
        _cb_stop(0),
    ]
    first_final = _response([_text_block("part one")], stop_reason="pause_turn")
    second_events = [
        _cb_start(0, _text_block("")),
        _cb_delta(0, _text_delta("part two")),
        _cb_stop(0),
    ]
    second_final = _response([_text_block("part two")], stop_reason="end_turn")
    model = _model_stream(
        [
            _StubMessageStream(first_events, first_final),
            _StubMessageStream(second_events, second_final),
        ]
    )

    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("go")])
    raw_events = list(model.stream(request))

    text_deltas = [e.text for e in raw_events if e.kind == "text_delta"]
    assert text_deltas == ["part one", "part two"]
    done = raw_events[-1]
    assert done.kind == "response_done"
    assert done.response is not None
    assert done.response.message.text == "part onepart two"
    assert done.response.stop_reason == StopReason.END_TURN
    assert len(model._client.messages.stream_calls) == 2  # pyright: ignore[reportAttributeAccessIssue]


def test_stream_pause_turn_exceeding_limit_raises_provider_error():
    def always_pause(_kwargs):
        events = [
            _cb_start(0, _text_block("")),
            _cb_delta(0, _text_delta("still going")),
            _cb_stop(0),
        ]
        final = _response([_text_block("still going")], stop_reason="pause_turn")
        return _StubMessageStream(events, final)

    model = _model_stream(always_pause)
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("go")])
    with pytest.raises(ProviderError):
        list(model.stream(request))


def test_stream_sdk_error_on_enter_becomes_provider_error():
    sdk_error = anthropic.APIConnectionError(message="boom", request=_httpx_request())

    def raiser(_kwargs):
        raise sdk_error

    model = _model_stream(raiser)
    request = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    with pytest.raises(ProviderError) as excinfo:
        list(model.stream(request))
    assert excinfo.value.__cause__ is sdk_error


# --- protocol conformance / registry factory ---------------------------------


def test_anthropic_model_satisfies_model_protocol():
    from composeai.models.base import Model

    assert isinstance(_model([_response([_text_block("ok")])]), Model)


def test_create_model_factory_builds_anthropic_model():
    from composeai.models.anthropic import create_model

    instance = create_model("claude-sonnet-5")
    assert isinstance(instance, AnthropicModel)
    assert instance.model_id == "claude-sonnet-5"


# --- shared conformance contract, run against AnthropicModel+stub -----------


def test_contract_text_completion():
    model = _model([_response([_text_block("hello there")])])
    contract.assert_text_completion(model, model_id="claude-sonnet-5", expected_text="hello there")


def test_contract_tool_call():
    model = _model([_response([_tool_use_block("tu1", "search", {})], stop_reason="tool_use")])
    contract.assert_tool_call(model, model_id="claude-sonnet-5", tool_name="search")


def test_contract_tool_result_batching():
    model = _model([_response([_text_block("ok")])])
    contract.assert_tool_result_batching_survives_round_trip(
        model, model_id="claude-sonnet-5", tool_call_id="tc1"
    )


def test_contract_structured_output():
    model = _model([_response([_text_block('{"answer": 42}')])])
    contract.assert_structured_output(model, model_id="claude-sonnet-5", expected={"answer": 42})


def test_contract_tool_use_with_output_schema_does_not_crash():
    model = _model([_response([_tool_use_block("tu1", "search", {})], stop_reason="tool_use")])
    contract.assert_tool_use_with_output_schema_does_not_crash(
        model, model_id="claude-sonnet-5", tool_name="search"
    )


def test_contract_stop_reason_mapping():
    model = _model([_response([_text_block("x")], stop_reason="max_tokens")])
    contract.assert_stop_reason_mapping(
        model, model_id="claude-sonnet-5", expected=StopReason.MAX_TOKENS, expected_raw="max_tokens"
    )


def test_contract_usage_lands():
    usage = _usage(input_tokens=11, output_tokens=22)
    model = _model([_response([_text_block("x")], usage=usage)])
    contract.assert_usage_lands(
        model, model_id="claude-sonnet-5", expected_input=11, expected_output=22
    )


def test_contract_sdk_failure_raises_provider_error():
    def raiser(_kwargs):
        raise anthropic.APIConnectionError(message="boom", request=_httpx_request())

    model = _model(raiser)
    contract.assert_sdk_failure_raises_provider_error(model, model_id="claude-sonnet-5")


# --- shared conformance contract, async twins (acomplete) --------------------


def test_async_contract_text_completion():
    import asyncio

    model = _amodel([_response([_text_block("hello there")])])
    asyncio.run(
        contract.assert_async_text_completion(
            model, model_id="claude-sonnet-5", expected_text="hello there"
        )
    )


def test_async_contract_tool_call():
    import asyncio

    model = _amodel([_response([_tool_use_block("tu1", "search", {})], stop_reason="tool_use")])
    asyncio.run(
        contract.assert_async_tool_call(model, model_id="claude-sonnet-5", tool_name="search")
    )


def test_async_contract_tool_result_batching():
    import asyncio

    model = _amodel([_response([_text_block("ok")])])
    asyncio.run(
        contract.assert_async_tool_result_batching_survives_round_trip(
            model, model_id="claude-sonnet-5", tool_call_id="tc1"
        )
    )


def test_async_contract_structured_output():
    import asyncio

    model = _amodel([_response([_text_block('{"answer": 42}')])])
    asyncio.run(
        contract.assert_async_structured_output(
            model, model_id="claude-sonnet-5", expected={"answer": 42}
        )
    )


def test_async_contract_tool_use_with_output_schema_does_not_crash():
    import asyncio

    model = _amodel([_response([_tool_use_block("tu1", "search", {})], stop_reason="tool_use")])
    asyncio.run(
        contract.assert_async_tool_use_with_output_schema_does_not_crash(
            model, model_id="claude-sonnet-5", tool_name="search"
        )
    )


def test_async_contract_stop_reason_mapping():
    import asyncio

    model = _amodel([_response([_text_block("x")], stop_reason="max_tokens")])
    asyncio.run(
        contract.assert_async_stop_reason_mapping(
            model,
            model_id="claude-sonnet-5",
            expected=StopReason.MAX_TOKENS,
            expected_raw="max_tokens",
        )
    )


def test_async_contract_usage_lands():
    import asyncio

    usage = _usage(input_tokens=11, output_tokens=22)
    model = _amodel([_response([_text_block("x")], usage=usage)])
    asyncio.run(
        contract.assert_async_usage_lands(
            model, model_id="claude-sonnet-5", expected_input=11, expected_output=22
        )
    )


def test_async_contract_sdk_failure_raises_provider_error():
    import asyncio

    def raiser(_kwargs):
        raise anthropic.APIConnectionError(message="boom", request=_httpx_request())

    model = _amodel(raiser)
    asyncio.run(
        contract.assert_async_sdk_failure_raises_provider_error(model, model_id="claude-sonnet-5")
    )


def test_anthropic_model_timeout_passed_to_sdk_client(monkeypatch):
    import anthropic

    from composeai.models.anthropic import AnthropicModel

    captured: dict = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    model = AnthropicModel("claude-sonnet-5", api_key="k", timeout=45)
    model._get_client()
    assert captured["timeout"] == 45


# --- async twins (acomplete / astream) ---------------------------------------


def test_acomplete_maps_response_like_complete():
    import asyncio

    content = [_text_block("hello there")]
    usage = _usage(input_tokens=11, output_tokens=22)

    # complete() path
    complete_model = _model([_response(content, usage=usage)])
    complete_resp = complete_model.complete(
        ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    )

    # acomplete() path, same wire response
    async_model = _amodel([_response(content, usage=usage)])

    async def drive():
        return await async_model.acomplete(
            ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
        )

    async_resp = asyncio.run(drive())

    assert async_resp.message.text == complete_resp.message.text
    assert async_resp.stop_reason == complete_resp.stop_reason
    assert async_resp.usage == complete_resp.usage


def test_astream_events_match_stream():
    import asyncio

    events = [
        _cb_start(0, _text_block("")),
        _cb_delta(0, _text_delta("hello")),
        _derived_text_event("hello", "hello"),
        _cb_delta(0, _text_delta(" there")),
        _cb_stop(0),
    ]
    final = _response([_text_block("hello there")])

    # stream() path
    sync_model = _model_stream([_StubMessageStream(events, final)])
    sync_events = list(
        sync_model.stream(ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")]))
    )

    # astream() path, same scripted event sequence
    async_model = _amodel_stream([_AsyncStubMessageStream(events, final)])

    async def drive():
        result = []
        async for event in async_model.astream(
            ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
        ):
            result.append(event)
        return result

    async_events = asyncio.run(drive())

    assert [e.kind for e in async_events] == [e.kind for e in sync_events]
    async_resp = async_events[-1].response
    sync_resp = sync_events[-1].response
    assert async_resp is not None
    assert sync_resp is not None
    assert async_resp.message.text == sync_resp.message.text


def test_async_client_constructed_with_same_kwargs(monkeypatch):
    captured: dict = {}

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAsyncAnthropic)
    model = AnthropicModel("claude-sonnet-5", api_key="k", timeout=45)
    model._get_async_client()
    assert captured["api_key"] == "k"
    assert captured["timeout"] == 45

    captured_no_timeout: dict = {}

    class FakeAsyncAnthropicNoTimeout:
        def __init__(self, **kwargs):
            captured_no_timeout.update(kwargs)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAsyncAnthropicNoTimeout)
    model_no_timeout = AnthropicModel("claude-sonnet-5", api_key="k")
    model_no_timeout._get_async_client()
    assert "timeout" not in captured_no_timeout


# --- injected sync client vs. the async engine's acomplete/astream discovery -


def test_injected_sync_client_used_by_async_engine():
    """A ``client=`` injected at construction is a SYNC client -- the async
    engine's ``getattr(model, "acomplete", None)`` discovery (models/base.py)
    must not route to ``acomplete``/``astream`` (which would silently build
    an unrelated ``AsyncAnthropic`` from env/api_key instead, see
    ``AnthropicModel.__init__``). It must fall back to running the injected
    client's sync ``complete()`` off-thread, exactly as it would for a model
    with no async methods at all.
    """
    from composeai.agentfn import agent

    model = _model([_response([_text_block("Hello there.")])])
    assert model.acomplete is None
    assert model.astream is None

    @agent(model=model, max_turns=2, name="anthropic_injected_client_agent")
    def greeter(name: str) -> str:
        """You are a friendly greeter."""
        return f"Greet {name}"

    run = greeter.run("Ann")

    assert run.output == "Hello there."
    assert run.status == "completed"
    assert len(model._client.messages.calls) == 1


# --- 0.6.0 prompt_cache / thinking / effort ------------------------------


def _call_kwargs(model: AnthropicModel):
    return model._client.messages.calls[0]  # pyright: ignore[reportAttributeAccessIssue]


def test_prompt_cache_marks_system_block_single_turn():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("hi")],
        system="be terse",
        prompt_cache=True,
    )
    model.complete(request)
    call = _call_kwargs(model)
    assert call["system"] == [
        {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}
    ]
    # single-turn: no tail marker on the (only) message
    for block in call["messages"][-1]["content"]:
        assert "cache_control" not in block


def test_prompt_cache_marks_conversation_tail_when_multiturn():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("q1"), Message.assistant("a1"), Message.user("q2")],
        system="be terse",
        prompt_cache=True,
    )
    model.complete(request)
    call = _call_kwargs(model)
    last_blocks = call["messages"][-1]["content"]
    assert last_blocks[-1]["cache_control"] == {"type": "ephemeral"}
    # only the LAST block of the LAST message is marked
    for msg in call["messages"][:-1]:
        for block in msg["content"]:
            assert "cache_control" not in block
    for block in last_blocks[:-1]:
        assert "cache_control" not in block


def test_prompt_cache_without_system_still_marks_tail():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("q1"), Message.assistant("a1"), Message.user("q2")],
        prompt_cache=True,
    )
    model.complete(request)
    call = _call_kwargs(model)
    assert "system" not in call
    assert call["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_prompt_cache_off_is_byte_identical_to_050():
    model = _model([_response([_text_block("ok")])])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("hi")],
        system="be terse",
        prompt_cache=False,
    )
    model.complete(request)
    call = _call_kwargs(model)
    assert call["system"] == "be terse"  # plain string, as today
    assert "thinking" not in call
    for msg in call["messages"]:
        for block in msg["content"]:
            assert "cache_control" not in block


def test_thinking_true_maps_to_adaptive_summarized():
    model = _model([_response([_text_block("ok")])])
    model.complete(
        ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")], thinking=True)
    )
    assert _call_kwargs(model)["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }


def test_thinking_false_maps_to_disabled():
    model = _model([_response([_text_block("ok")])])
    model.complete(
        ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")], thinking=False)
    )
    assert _call_kwargs(model)["thinking"] == {"type": "disabled"}


def test_thinking_none_sends_nothing():
    model = _model([_response([_text_block("ok")])])
    model.complete(ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")]))
    assert "thinking" not in _call_kwargs(model)


def test_effort_lands_in_output_config():
    model = _model([_response([_text_block("ok")])])
    model.complete(
        ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")], effort="high")
    )
    assert _call_kwargs(model)["output_config"] == {"effort": "high"}


def test_effort_merges_with_structured_output_format():
    model = _model([_response([_text_block('{"x": 1}')])])
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    model.complete(
        ModelRequest(
            model="claude-sonnet-5",
            messages=[Message.user("hi")],
            output_schema=schema,
            effort="max",
        )
    )
    cfg = _call_kwargs(model)["output_config"]
    assert cfg["effort"] == "max"
    assert cfg["format"] == {"type": "json_schema", "schema": schema}


def test_stream_gets_same_new_kwargs():
    # Drive AnthropicModel.stream with the file's stream-stub helpers and
    # prompt_cache=True + thinking=True + effort="high", then assert the
    # captured stream kwargs carry the same "system" block list, tail
    # cache_control marker, "thinking", and "output_config" shapes the
    # complete() tests above pin.
    events = [
        _cb_start(0, _text_block("")),
        _cb_delta(0, _text_delta("ok")),
        _cb_stop(0),
    ]
    final = _response([_text_block("ok")])
    model = _model_stream([_StubMessageStream(events, final)])
    request = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("q1"), Message.assistant("a1"), Message.user("q2")],
        system="be terse",
        prompt_cache=True,
        thinking=True,
        effort="high",
    )
    list(model.stream(request))
    call = model._client.messages.stream_calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["system"] == [
        {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert call["output_config"] == {"effort": "high"}
