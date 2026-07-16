"""Tests for the OpenAI Responses API adapter, using a duck-typed fake client (no network).

Wire shapes below were verified against the installed ``openai`` SDK
(``openai/types/responses/*.py``) rather than assumed from memory -- see
scratchpad/reports/phase-4-report.md for the verification notes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from conformance import contract

from composeai.errors import ConfigError, ProviderError
from composeai.messages import (
    ImagePart,
    Message,
    StopReason,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from composeai.models.base import ModelRequest, ToolSpec
from composeai.models.openai import OpenAIModel
from composeai.models.prices import ModelPrice, register_price

# --- Stub client helpers -----------------------------------------------------


def _input_tokens_details(cached_tokens: int = 0, cache_write_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens)


def _output_tokens_details(reasoning_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(reasoning_tokens=reasoning_tokens)


def _usage(
    input_tokens: int = 10,
    output_tokens: int = 20,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=_input_tokens_details(cached_tokens=cached_tokens),
        output_tokens_details=_output_tokens_details(reasoning_tokens=reasoning_tokens),
        total_tokens=input_tokens + output_tokens,
    )


def _output_text_content(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="output_text", text=text)


def _refusal_content(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="refusal", refusal=text)


def _message_item(*contents: Any, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(
        type="message", id="msg_1", role="assistant", content=list(contents), status=status
    )


def _function_call_item(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=json.dumps(arguments),
        id=f"fc_{call_id}",
    )


def _reasoning_item(
    item_id: str = "rs_1",
    summary_texts: list[str] | None = None,
    content_texts: list[str] | None = None,
    encrypted_content: str | None = None,
) -> SimpleNamespace:
    summary_source = summary_texts or ["pondering..."]
    summary = [SimpleNamespace(type="summary_text", text=t) for t in summary_source]
    content = (
        [SimpleNamespace(type="reasoning_text", text=t) for t in content_texts]
        if content_texts
        else None
    )
    return SimpleNamespace(
        type="reasoning",
        id=item_id,
        summary=summary,
        content=content,
        encrypted_content=encrypted_content,
        status="completed",
    )


def _response(
    output: list[Any],
    status: str = "completed",
    usage: SimpleNamespace | None = None,
    incomplete_reason: str | None = None,
    error: SimpleNamespace | None = None,
) -> SimpleNamespace:
    incomplete_details = SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
    return SimpleNamespace(
        output=output,
        status=status,
        incomplete_details=incomplete_details,
        usage=usage if usage is not None else _usage(),
        error=error,
    )


class _StubResponses:
    """Duck-typed stand-in for ``openai.OpenAI().responses``."""

    def __init__(self, responses, stream_responses: Any = None) -> None:
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
        self.responses = _StubResponses(responses, stream_responses)


def _model(responses, model_id: str = "gpt-5-test") -> OpenAIModel:
    return OpenAIModel(model_id, client=_StubClient(responses))


class _AsyncStubResponses:
    """Async twin of ``_StubResponses``: duck-typed stand-in for
    ``openai.AsyncOpenAI().responses`` (``create`` is a coroutine; ``stream``
    stays a plain method that returns an async context manager, mirroring
    the real SDK's ``AsyncResponses.stream``)."""

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
        self.responses = _AsyncStubResponses(responses, stream_responses)


def _amodel(responses, model_id: str = "gpt-5-test") -> OpenAIModel:
    model = OpenAIModel(model_id)
    model._async_client = _AsyncStubClient(responses)
    return model


# --- stream() stub helpers ----------------------------------------------------


class _StubResponseStream:
    """Duck-typed stand-in for ``openai.OpenAI().responses.stream()``'s
    context-manager result (``ResponseStream``): iterates the raw
    ``response.*`` SSE-shaped events and exposes ``get_final_response()``.
    """

    def __init__(self, raw_events, final_response) -> None:
        self._raw_events = raw_events
        self._final_response = final_response

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._raw_events)

    def get_final_response(self):
        return self._final_response


def _stream_text_delta(text: str, item_id: str = "msg_1", output_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_text.delta",
        delta=text,
        item_id=item_id,
        output_index=output_index,
        content_index=0,
        sequence_number=0,
        logprobs=[],
    )


def _stream_item_added(item: Any, output_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.added", item=item, output_index=output_index, sequence_number=0
    )


def _stream_item_done(item: Any, output_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.done", item=item, output_index=output_index, sequence_number=0
    )


def _stream_function_call_arguments_delta(
    delta: str, item_id: str, output_index: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.function_call_arguments.delta",
        delta=delta,
        item_id=item_id,
        output_index=output_index,
        sequence_number=0,
    )


def _stream_reasoning_summary_text_delta(
    text: str, item_id: str = "rs_1", output_index: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.reasoning_summary_text.delta",
        delta=text,
        item_id=item_id,
        output_index=output_index,
        summary_index=0,
        sequence_number=0,
    )


def _stream_completed(response: Any) -> SimpleNamespace:
    return SimpleNamespace(type="response.completed", response=response, sequence_number=0)


def _stream_incomplete(response: Any) -> SimpleNamespace:
    # Real shape verified against the installed SDK's
    # ResponseIncompleteEvent: carries a `.response` field of the same
    # `Response` shape as ResponseCompletedEvent.response.
    return SimpleNamespace(type="response.incomplete", response=response, sequence_number=0)


def _stream_failed(response: Any) -> SimpleNamespace:
    # Real shape verified against the installed SDK's ResponseFailedEvent:
    # same `.response` shape as ResponseCompletedEvent.response.
    return SimpleNamespace(type="response.failed", response=response, sequence_number=0)


def _response_error(code: str = "server_error", message: str = "boom") -> SimpleNamespace:
    return SimpleNamespace(code=code, message=message)


def _model_stream(stream_responses, model_id: str = "gpt-5-test") -> OpenAIModel:
    return OpenAIModel(model_id, client=_StubClient([], stream_responses))


class _AsyncStubResponseStream:
    """Async twin of ``_StubResponseStream``: an async-context-manager whose
    ``__aenter__`` returns an async-iterable over the same scripted raw
    events, with an awaitable ``get_final_response()`` (mirroring the real
    SDK's ``AsyncResponseStream``).
    """

    def __init__(self, raw_events, final_response) -> None:
        self._raw_events = raw_events
        self._final_response = final_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def __aiter__(self):
        for event in self._raw_events:
            yield event

    async def get_final_response(self):
        return self._final_response


def _amodel_stream(stream_responses, model_id: str = "gpt-5-test") -> OpenAIModel:
    model = OpenAIModel(model_id)
    model._async_client = _AsyncStubClient([], stream_responses)
    return model


# --- constructor / lazy client / missing key --------------------------------


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = OpenAIModel("gpt-5-test")
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        model.complete(request)


def test_injected_client_bypasses_key_requirement(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = _model([_response([_message_item(_output_text_content("hi"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    response = model.complete(request)
    assert response.message.text == "hi"


def test_provider_sdk_client_not_built_when_client_injected():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    model.complete(request)  # must not raise


# --- request mapping: instructions / temperature / max_output_tokens --------


def test_instructions_omitted_when_system_none():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert "instructions" not in call


def test_instructions_included_when_system_set():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], system="be terse")
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["instructions"] == "be terse"


def test_temperature_omitted_when_none():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert "temperature" not in call


def test_temperature_passed_through_when_set():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], temperature=0.4)
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["temperature"] == 0.4


def test_max_output_tokens_passed_through():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], max_tokens=42)
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["max_output_tokens"] == 42


# --- request mapping: message/content conversion ----------------------------


def test_user_text_part_maps_to_message_item_with_input_text():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hello")])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]


def test_user_image_part_base64_maps_to_input_image_data_url():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    image = ImagePart(media_type="image/png", data="YWJj")
    request = ModelRequest(model="gpt-5-test", messages=[Message.user([image])])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    item = call["input"][0]["content"][0]
    assert item["type"] == "input_image"
    assert item["image_url"] == "data:image/png;base64,YWJj"


def test_user_image_part_url_maps_to_input_image_url():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    image = ImagePart(media_type="image/png", url="https://example.com/x.png")
    request = ModelRequest(model="gpt-5-test", messages=[Message.user([image])])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    item = call["input"][0]["content"][0]
    assert item["type"] == "input_image"
    assert item["image_url"] == "https://example.com/x.png"


def test_assistant_text_part_maps_to_assistant_message_item():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(
        model="gpt-5-test", messages=[Message.assistant("prior reply"), Message.user("go on")]
    )
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["input"][0] == {
        "role": "assistant",
        "content": [{"type": "input_text", "text": "prior reply"}],
    }


def test_tool_call_part_maps_to_function_call_item_with_call_id():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    call_part = ToolCallPart(id="tc1", name="search", arguments={"q": "cats"})
    request = ModelRequest(model="gpt-5-test", messages=[Message.assistant([call_part])])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    item = call["input"][0]
    assert item == {
        "type": "function_call",
        "call_id": "tc1",
        "name": "search",
        "arguments": json.dumps({"q": "cats"}),
    }


def test_tool_result_part_maps_without_error_prefix_when_not_error():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    result = ToolResultPart(tool_call_id="tc1", content="42")
    request = ModelRequest(model="gpt-5-test", messages=[Message.user([result])])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["input"][0] == {"type": "function_call_output", "call_id": "tc1", "output": "42"}


def test_tool_result_part_prepends_error_when_is_error():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    result = ToolResultPart(tool_call_id="tc1", content="boom", is_error=True)
    request = ModelRequest(model="gpt-5-test", messages=[Message.user([result])])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["input"][0]["output"] == "ERROR: boom"


def test_multiple_tool_results_become_separate_items_not_batched():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(
        model="gpt-5-test",
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
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["input"] == [
        {"type": "function_call_output", "call_id": "tc1", "output": "a"},
        {"type": "function_call_output", "call_id": "tc2", "output": "b"},
    ]


def test_history_echo_includes_function_call_items_with_same_call_id():
    # Round trip: a tool call from a prior response must be echoed back with
    # the *same* call_id when history is replayed.
    model = _model(
        [
            _response([_function_call_item("call-abc", "search", {"q": "x"})]),
            _response([_message_item(_output_text_content("done"))]),
        ]
    )
    request1 = ModelRequest(model="gpt-5-test", messages=[Message.user("search please")])
    resp1 = model.complete(request1)
    tool_call = next(p for p in resp1.message.parts if isinstance(p, ToolCallPart))
    assert tool_call.id == "call-abc"

    request2 = ModelRequest(
        model="gpt-5-test",
        messages=[
            Message.user("search please"),
            Message.assistant(resp1.message.parts),
            Message.user([ToolResultPart(tool_call_id=tool_call.id, content="result")]),
        ],
    )
    model.complete(request2)
    call2 = model._client.responses.calls[1]  # pyright: ignore[reportAttributeAccessIssue]
    function_call_items = [i for i in call2["input"] if i.get("type") == "function_call"]
    assert len(function_call_items) == 1
    assert function_call_items[0]["call_id"] == "call-abc"


# --- request mapping: tools / structured output -----------------------------


def test_tools_mapped_flat_with_parameters_and_strict():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    spec = ToolSpec(
        name="search",
        description="search the web",
        input_schema={"type": "object", "additionalProperties": False},
        strict=True,
    )
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], tools=[spec])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["tools"] == [
        {
            "type": "function",
            "name": "search",
            "description": "search the web",
            "parameters": {"type": "object", "additionalProperties": False},
            "strict": True,
        }
    ]


def test_tools_omitted_when_none():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert "tools" not in call


def test_output_schema_maps_to_text_format_json_schema():
    # "a" is required (and additionalProperties: false is set) so this
    # schema qualifies for strict mode -- see the dedicated
    # test_output_schema_*_strict_compatibility tests below for the
    # downgrade-to-strict-false cases.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    model = _model([_response([_message_item(_output_text_content('{"a": 1}'))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], output_schema=schema)
    resp = model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["text"] == {
        "format": {"type": "json_schema", "name": "response", "schema": schema, "strict": True}
    }
    assert resp.parsed == {"a": 1}


def test_output_schema_uses_title_as_name_when_valid():
    schema = {"type": "object", "title": "MyAnswer"}
    model = _model([_response([_message_item(_output_text_content('{"a": 1}'))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], output_schema=schema)
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["text"]["format"]["name"] == "MyAnswer"


# --- request mapping: strict-mode schema compatibility -----------------------
#
# Regression for the "strict:true forced onto schemas OpenAI's strict subset
# rejects" critical: @tool's generated schema for a keyword argument with a
# Python default (tools.py's `_build_param_model`) omits that property from
# `required` and carries a `default` key -- both rejected by OpenAI's strict
# JSON Schema subset (see test_tools.py's
# test_schema_required_param_has_no_default, which asserts exactly that
# shape is what @tool produces). `_schema.py`'s `seal_schema` also
# deliberately leaves a *schema-valued* `additionalProperties` in place for
# an open `dict[str, V]` mapping (not forced to `False`) -- also rejected by
# strict mode. Neither is a hypothetical: both are real shapes this
# codebase's own schema generation produces.


def test_default_valued_property_downgrades_tool_to_non_strict():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    spec = ToolSpec(name="search", description="search the web", input_schema=schema, strict=True)
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], tools=[spec])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["tools"][0]["strict"] is False
    # The schema itself is never mutated to "fix" it -- sent verbatim.
    assert call["tools"][0]["parameters"] == schema


def test_schema_valued_additional_properties_downgrades_tool_to_non_strict():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "object", "additionalProperties": {"type": "string"}}},
        "required": ["tags"],
        "additionalProperties": False,
    }
    spec = ToolSpec(name="tag", description="tag it", input_schema=schema, strict=True)
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], tools=[spec])
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["tools"][0]["strict"] is False
    assert call["tools"][0]["parameters"] == schema


def test_default_valued_property_downgrades_output_schema_to_non_strict():
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "default": 1.0},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }
    model = _model([_response([_message_item(_output_text_content('{"answer": "x"}'))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], output_schema=schema)
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["text"]["format"]["strict"] is False
    assert call["text"]["format"]["schema"] == schema


def test_non_json_output_text_with_output_schema_raises_provider_error():
    model = _model([_response([_message_item(_output_text_content("not json"))])])
    request = ModelRequest(
        model="gpt-5-test", messages=[Message.user("hi")], output_schema={"type": "object"}
    )
    with pytest.raises(ProviderError):
        model.complete(request)


def test_function_call_stop_with_output_schema_does_not_crash_and_parsed_is_none():
    # Regression: an @agent combining tools=[...] with a non-str output_type
    # sends output_schema on every turn, so a function_call-only turn has no
    # output_text item at all (message.text == '') -- json.loads('') must
    # not be attempted (it would raise and mask the already-computed
    # TOOL_USE stop reason behind a spurious ProviderError).
    model = _model([_response([_function_call_item("c1", "search", {})])])
    request = ModelRequest(
        model="gpt-5-test", messages=[Message.user("hi")], output_schema={"type": "object"}
    )
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.TOOL_USE
    assert resp.parsed is None


def test_refusal_stop_with_output_schema_does_not_crash_and_parsed_is_none():
    model = _model([_response([_message_item(_refusal_content("I can't help with that"))])])
    request = ModelRequest(
        model="gpt-5-test", messages=[Message.user("hi")], output_schema={"type": "object"}
    )
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.REFUSAL
    assert resp.parsed is None


def test_incomplete_stop_with_output_schema_does_not_crash_and_parsed_is_none():
    model = _model(
        [
            _response(
                [_message_item(_output_text_content('{"partial": tr'))],
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        ]
    )
    request = ModelRequest(
        model="gpt-5-test", messages=[Message.user("hi")], output_schema={"type": "object"}
    )
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.MAX_TOKENS
    assert resp.parsed is None


# --- response mapping: output items -----------------------------------------


def test_output_text_maps_to_text_part():
    model = _model([_response([_message_item(_output_text_content("hello there"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.message.text == "hello there"
    assert resp.stop_reason == StopReason.END_TURN


def test_function_call_item_maps_to_tool_call_part_with_parsed_arguments():
    model = _model([_response([_function_call_item("call-1", "search", {"q": "cats"})])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    calls = [p for p in resp.message.parts if isinstance(p, ToolCallPart)]
    assert len(calls) == 1
    assert calls[0].id == "call-1"
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "cats"}
    assert resp.stop_reason == StopReason.TOOL_USE


def test_malformed_function_call_arguments_raise_provider_error():
    bad_item = SimpleNamespace(
        type="function_call", call_id="call-1", name="search", arguments="not json", id="fc_1"
    )
    model = _model([_response([bad_item])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError):
        model.complete(request)


def test_refusal_content_maps_to_text_part_and_refusal_stop_reason():
    model = _model([_response([_message_item(_refusal_content("I can't help with that"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.message.text == "I can't help with that"
    assert resp.stop_reason == StopReason.REFUSAL


def test_reasoning_item_maps_to_thinking_part():
    model = _model([_response([_reasoning_item(summary_texts=["step one", "step two"])])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    thinking_parts = [p for p in resp.message.parts if isinstance(p, ThinkingPart)]
    assert len(thinking_parts) == 1
    part = thinking_parts[0]
    assert part.text == "step onestep two"
    assert part.provider == "openai"
    assert part.provider_token is not None
    payload = json.loads(part.provider_token)
    assert payload["model"] == "gpt-5-test"
    assert payload["item"]["id"] == "rs_1"
    assert payload["item"]["summary"] == [
        {"type": "summary_text", "text": "step one"},
        {"type": "summary_text", "text": "step two"},
    ]


# --- response mapping: stop reasons ------------------------------------------


def test_function_call_present_yields_tool_use():
    model = _model([_response([_function_call_item("c1", "f", {})], status="completed")])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.TOOL_USE


def test_incomplete_max_output_tokens_yields_max_tokens_with_composed_raw():
    model = _model(
        [
            _response(
                [_message_item(_output_text_content("partial"))],
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        ]
    )
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.MAX_TOKENS
    assert resp.raw_stop_reason == "incomplete:max_output_tokens"


def test_completed_yields_end_turn():
    model = _model([_response([_message_item(_output_text_content("done"))], status="completed")])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.END_TURN
    assert resp.raw_stop_reason == "completed"


def test_unrecognized_status_yields_other():
    model = _model([_response([_message_item(_output_text_content("x"))], status="cancelled")])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.OTHER
    assert resp.raw_stop_reason == "cancelled"


# --- response mapping: usage / cost ------------------------------------------


def test_usage_tokens_land_in_usage():
    usage = _usage(input_tokens=123, output_tokens=45)
    model = _model([_response([_message_item(_output_text_content("x"))], usage=usage)])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.input_tokens == 123
    assert resp.usage.output_tokens == 45


def test_cached_tokens_map_to_cache_read_tokens():
    usage = _usage(cached_tokens=77)
    model = _model([_response([_message_item(_output_text_content("x"))], usage=usage)])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cache_read_tokens == 77


def test_reasoning_tokens_map_to_usage_reasoning_tokens():
    usage = _usage(reasoning_tokens=33)
    model = _model([_response([_message_item(_output_text_content("x"))], usage=usage)])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.reasoning_tokens == 33


def test_cache_creation_tokens_always_zero():
    model = _model([_response([_message_item(_output_text_content("x"))])])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cache_creation_tokens == 0


def test_usage_cost_computed_for_known_model():
    register_price("openai", "price-test-model", ModelPrice(input=2.0, output=8.0, cache_read=0.2))
    big_usage = _usage(input_tokens=1_000_000, output_tokens=1_000_000)
    model = _model(
        [_response([_message_item(_output_text_content("x"))], usage=big_usage)],
        model_id="price-test-model",
    )
    request = ModelRequest(model="price-test-model", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cost_usd == pytest.approx(2.0 + 8.0)
    assert resp.usage.cost_complete is True


def test_cached_tokens_are_not_double_billed():
    # Regression for the "double-billed cached input tokens" critical:
    # `input_tokens` is documented by the installed SDK's
    # `openai.types.responses.response_usage.InputTokensDetails` as "a
    # detailed breakdown of the input tokens" -- i.e. `cached_tokens` is a
    # SUBSET of `input_tokens`, not additive (confirmed unambiguously by the
    # structurally identical `openai.types.realtime.
    # RealtimeResponseUsageInputTokenDetails` docstring: "Cached tokens here
    # are counted as a subset of input tokens, meaning input tokens will
    # include cached and uncached tokens"). Billing the full 1,000,000
    # input_tokens at the input rate AND separately billing the 800,000
    # cached_tokens at the cache_read rate would charge the cached slice
    # twice. Correct cost = (input_tokens - cached_tokens) * input_price +
    # cached_tokens * cache_read_price + output_tokens * output_price.
    register_price(
        "openai", "cache-billing-test-model", ModelPrice(input=2.0, output=8.0, cache_read=0.2)
    )
    usage = _usage(input_tokens=1_000_000, output_tokens=100_000, cached_tokens=800_000)
    model = _model(
        [_response([_message_item(_output_text_content("x"))], usage=usage)],
        model_id="cache-billing-test-model",
    )
    request = ModelRequest(model="cache-billing-test-model", messages=[Message.user("hi")])
    resp = model.complete(request)
    expected = (200_000 * 2.0 + 800_000 * 0.2 + 100_000 * 8.0) / 1_000_000
    assert resp.usage.cost_usd == pytest.approx(expected)
    # Sanity: the naive (double-billing) computation would give a visibly
    # different, larger number (2.96) -- so this test would fail loudly, not
    # by a rounding error, if the fix regresses.
    assert resp.usage.cost_usd != pytest.approx(2.96)


def test_usage_cost_none_and_incomplete_for_unknown_model():
    model = _model(
        [_response([_message_item(_output_text_content("x"))])], model_id="gpt-unknown-9000"
    )
    request = ModelRequest(model="gpt-unknown-9000", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cost_usd is None
    assert resp.usage.cost_complete is False


# --- reasoning-item echo ------------------------------------------------------


def test_reasoning_part_echoed_verbatim_when_same_model():
    model = _model(
        [
            _response(
                [
                    _reasoning_item(item_id="rs-a", summary_texts=["thought"]),
                    _message_item(_output_text_content("done")),
                ]
            ),
            _response([_message_item(_output_text_content("continuing"))]),
        ],
        model_id="gpt-5-test",
    )
    request1 = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp1 = model.complete(request1)

    request2 = ModelRequest(
        model="gpt-5-test",
        messages=[
            Message.user("hi"),
            Message.assistant(resp1.message.parts),
            Message.user("continue"),
        ],
    )
    model.complete(request2)
    call2 = model._client.responses.calls[1]  # pyright: ignore[reportAttributeAccessIssue]
    # The reasoning item is a top-level input item, not message content.
    top_level_reasoning = [i for i in call2["input"] if i.get("type") == "reasoning"]
    assert len(top_level_reasoning) == 1
    assert top_level_reasoning[0]["id"] == "rs-a"


def test_reasoning_part_dropped_when_different_model():
    producer = _model(
        [_response([_reasoning_item(item_id="rs-a"), _message_item(_output_text_content("done"))])],
        model_id="gpt-5-test",
    )
    request1 = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    resp1 = producer.complete(request1)

    consumer = _model(
        [_response([_message_item(_output_text_content("continuing"))])], model_id="gpt-5-other"
    )
    request2 = ModelRequest(
        model="gpt-5-other",
        messages=[
            Message.user("hi"),
            Message.assistant(resp1.message.parts),
            Message.user("continue"),
        ],
    )
    consumer.complete(request2)
    call = consumer._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert all(i.get("type") != "reasoning" for i in call["input"])


def test_reasoning_part_dropped_when_different_provider():
    other_provider_part = ThinkingPart(text="hmm", provider="anthropic", provider_token="{}")
    model = _model([_response([_message_item(_output_text_content("ok"))])], model_id="gpt-5-test")
    request = ModelRequest(
        model="gpt-5-test",
        messages=[Message.assistant([other_provider_part])],
    )
    model.complete(request)
    call = model._client.responses.calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    # The assistant message contributed no content and no reasoning item.
    assert call["input"] == []


def test_malformed_reasoning_provider_token_raises_provider_error():
    # Mirrors test_anthropic.py's
    # test_malformed_thinking_provider_token_raises_provider_error: a
    # hand-crafted/corrupted ThinkingPart.provider_token (e.g. from a
    # hand-edited cassette or a resumed agent_state) must not surface a raw
    # json.JSONDecodeError -- every error composeai raises is a ComposeError
    # (errors.py's module docstring).
    bad_part = ThinkingPart(text="hmm", provider="openai", provider_token="{not valid json")
    model = _model([_response([_message_item(_output_text_content("ok"))])], model_id="gpt-5-test")
    request = ModelRequest(model="gpt-5-test", messages=[Message.assistant([bad_part])])
    with pytest.raises(ProviderError):
        model.complete(request)


def test_reasoning_provider_token_missing_item_key_raises_provider_error():
    # Valid JSON, but missing the "item" key _echo_reasoning_item relies on
    # (e.g. a hand-edited/corrupted persisted state) -- a raw KeyError must
    # not escape either.
    bad_part = ThinkingPart(
        text="hmm", provider="openai", provider_token=json.dumps({"model": "gpt-5-test"})
    )
    model = _model([_response([_message_item(_output_text_content("ok"))])], model_id="gpt-5-test")
    request = ModelRequest(model="gpt-5-test", messages=[Message.assistant([bad_part])])
    with pytest.raises(ProviderError):
        model.complete(request)


# --- SDK error mapping --------------------------------------------------------


def _httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/responses")


def test_api_status_error_becomes_provider_error():
    req = _httpx_request()
    resp = httpx.Response(500, request=req, json={"error": {"message": "server exploded"}})
    sdk_error = openai.APIStatusError("server exploded", response=resp, body=None)

    def raiser(_kwargs):
        raise sdk_error

    model = _model(raiser, model_id="gpt-5-test")
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError) as excinfo:
        model.complete(request)
    assert excinfo.value.provider == "openai"
    assert excinfo.value.model == "gpt-5-test"
    assert excinfo.value.__cause__ is sdk_error


def test_api_connection_error_becomes_provider_error():
    sdk_error = openai.APIConnectionError(message="connection boom", request=_httpx_request())

    def raiser(_kwargs):
        raise sdk_error

    model = _model(raiser)
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError) as excinfo:
        model.complete(request)
    assert excinfo.value.__cause__ is sdk_error


def test_no_retry_loop_added_on_top_of_sdk():
    calls = []

    def raiser(_kwargs):
        calls.append(1)
        raise openai.APIConnectionError(message="boom", request=_httpx_request())

    model = _model(raiser)
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError):
        model.complete(request)
    assert len(calls) == 1


# --- stream() ------------------------------------------------------------


def test_stream_yields_text_deltas_then_response_done():
    final = _response([_message_item(_output_text_content("hello there"))])
    events = [_stream_text_delta("hello"), _stream_text_delta(" there"), _stream_completed(final)]
    model = _model_stream([_StubResponseStream(events, final)])

    raw_events = list(model.stream(ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])))

    kinds = [e.kind for e in raw_events]
    assert kinds == ["text_delta", "text_delta", "response_done"]
    assert [e.text for e in raw_events[:2]] == ["hello", " there"]
    done_response = raw_events[-1].response
    assert done_response is not None
    assert done_response.message.text == "hello there"
    assert done_response.stop_reason == StopReason.END_TURN


def test_stream_tool_call_yields_started_args_delta_finished():
    added_item = _function_call_item("call-1", "search", {})
    done_item = _function_call_item("call-1", "search", {"q": "cats"})
    final = _response([done_item])
    events = [
        _stream_item_added(added_item),
        _stream_function_call_arguments_delta('{"q": ', "fc_call-1"),
        _stream_function_call_arguments_delta('"cats"}', "fc_call-1"),
        _stream_item_done(done_item),
        _stream_completed(final),
    ]
    model = _model_stream([_StubResponseStream(events, final)])

    raw_events = list(model.stream(ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])))

    kinds = [e.kind for e in raw_events]
    assert kinds == [
        "tool_call_started",
        "tool_args_delta",
        "tool_args_delta",
        "tool_call_finished",
        "response_done",
    ]
    started, delta1, delta2, finished, done = raw_events
    assert started.tool_call_id == "call-1"
    assert started.tool_name == "search"
    assert delta1.text == '{"q": '
    assert delta1.tool_call_id == "call-1"
    assert delta2.text == '"cats"}'
    assert finished.tool_call_id == "call-1"
    assert finished.tool_name == "search"
    assert done.response is not None
    assert done.response.stop_reason == StopReason.TOOL_USE
    calls = [p for p in done.response.message.parts if isinstance(p, ToolCallPart)]
    assert calls[0].arguments == {"q": "cats"}


def test_stream_reasoning_summary_delta_maps_to_thinking_delta():
    final = _response([_reasoning_item(summary_texts=["pondering"])])
    events = [_stream_reasoning_summary_text_delta("pondering"), _stream_completed(final)]
    model = _model_stream([_StubResponseStream(events, final)])

    raw_events = list(model.stream(ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])))

    kinds = [e.kind for e in raw_events]
    assert kinds == ["thinking_delta", "response_done"]
    assert raw_events[0].text == "pondering"


def test_stream_response_done_equals_complete_style_mapping():
    output = [_message_item(_output_text_content("hello there"))]
    complete_model = _model([_response(output)])
    complete_resp = complete_model.complete(
        ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    )

    final = _response(output)
    events = [_stream_text_delta("hello there"), _stream_completed(final)]
    stream_model = _model_stream([_StubResponseStream(events, final)])
    raw_events = list(
        stream_model.stream(ModelRequest(model="gpt-5-test", messages=[Message.user("hi")]))
    )
    streamed_resp = raw_events[-1].response
    assert streamed_resp is not None

    assert streamed_resp.message.text == complete_resp.message.text
    assert streamed_resp.stop_reason == complete_resp.stop_reason
    assert streamed_resp.usage == complete_resp.usage


def test_stream_records_request_shape_via_stub_calls():
    final = _response([_message_item(_output_text_content("x"))])
    events = [_stream_text_delta("x"), _stream_completed(final)]
    model = _model_stream([_StubResponseStream(events, final)])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")], system="be terse")
    list(model.stream(request))
    call = model._client.responses.stream_calls[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert call["instructions"] == "be terse"
    assert call["model"] == "gpt-5-test"


def test_stream_sdk_error_becomes_provider_error():
    sdk_error = openai.APIConnectionError(message="boom", request=_httpx_request())

    def raiser(_kwargs):
        raise sdk_error

    model = _model_stream(raiser)
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError) as excinfo:
        list(model.stream(request))
    assert excinfo.value.__cause__ is sdk_error


def test_stream_without_completed_event_raises_provider_error():
    # A stream that ends without ever seeing response.completed is a
    # malformed/incomplete stream -- must surface as ProviderError, not
    # silently return a bogus response.
    model = _model_stream([_StubResponseStream([_stream_text_delta("x")], None)])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError):
        list(model.stream(request))


def test_stream_response_incomplete_max_output_tokens_yields_max_tokens_not_provider_error():
    # A response that legitimately hits max_output_tokens ends via a
    # response.incomplete event, not response.completed -- streaming must
    # map it the same way complete() does (StopReason.MAX_TOKENS), not raise
    # a spurious "stream ended without response.completed" ProviderError.
    final = _response(
        [_message_item(_output_text_content("partial"))],
        status="incomplete",
        incomplete_reason="max_output_tokens",
    )
    events = [_stream_text_delta("partial"), _stream_incomplete(final)]
    model = _model_stream([_StubResponseStream(events, final)])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    raw_events = list(model.stream(request))
    done = raw_events[-1]
    assert done.kind == "response_done"
    assert done.response is not None
    assert done.response.stop_reason == StopReason.MAX_TOKENS
    assert done.response.raw_stop_reason == "incomplete:max_output_tokens"


def test_stream_response_incomplete_other_reason_yields_other_not_provider_error():
    final = _response(
        [_message_item(_output_text_content("partial"))],
        status="incomplete",
        incomplete_reason="content_filter",
    )
    events = [_stream_incomplete(final)]
    model = _model_stream([_StubResponseStream(events, final)])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    raw_events = list(model.stream(request))
    done = raw_events[-1]
    assert done.response is not None
    assert done.response.stop_reason == StopReason.OTHER


def test_stream_response_failed_raises_provider_error_with_failure_reason():
    final = _response(
        [],
        status="failed",
        error=_response_error(code="server_error", message="the model is overloaded"),
    )
    events = [_stream_failed(final)]
    model = _model_stream([_StubResponseStream(events, final)])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError, match="the model is overloaded"):
        list(model.stream(request))


def test_complete_response_failed_raises_provider_error_with_failure_reason():
    # Parity with the streaming case: a non-streaming call whose response
    # comes back with status="failed" (a normal 200-level response body, not
    # an SDK-raised exception) must also raise ProviderError rather than
    # silently mapping to StopReason.OTHER.
    final = _response(
        [], status="failed", error=_response_error(code="server_error", message="boom")
    )
    model = _model([final])
    request = ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError, match="boom"):
        model.complete(request)


# --- protocol conformance / registry factory ---------------------------------


def test_openai_model_satisfies_model_protocol():
    from composeai.models.base import Model

    assert isinstance(_model([_response([_message_item(_output_text_content("ok"))])]), Model)


def test_create_model_factory_builds_openai_model():
    from composeai.models.openai import create_model

    instance = create_model("gpt-5-test")
    assert isinstance(instance, OpenAIModel)
    assert instance.model_id == "gpt-5-test"


# --- shared conformance contract ---------------------------------------------


def test_contract_text_completion():
    model = _model([_response([_message_item(_output_text_content("hello there"))])])
    contract.assert_text_completion(model, model_id="gpt-5-test", expected_text="hello there")


def test_contract_tool_call():
    model = _model([_response([_function_call_item("c1", "search", {})])])
    contract.assert_tool_call(model, model_id="gpt-5-test", tool_name="search")


def test_contract_tool_result_batching():
    model = _model([_response([_message_item(_output_text_content("ok"))])])
    contract.assert_tool_result_batching_survives_round_trip(
        model, model_id="gpt-5-test", tool_call_id="tc1"
    )


def test_contract_structured_output():
    model = _model([_response([_message_item(_output_text_content('{"answer": 42}'))])])
    contract.assert_structured_output(model, model_id="gpt-5-test", expected={"answer": 42})


def test_contract_tool_use_with_output_schema_does_not_crash():
    model = _model([_response([_function_call_item("c1", "search", {})])])
    contract.assert_tool_use_with_output_schema_does_not_crash(
        model, model_id="gpt-5-test", tool_name="search"
    )


def test_contract_stop_reason_mapping():
    model = _model(
        [
            _response(
                [_message_item(_output_text_content("partial"))],
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        ]
    )
    contract.assert_stop_reason_mapping(
        model,
        model_id="gpt-5-test",
        expected=StopReason.MAX_TOKENS,
        expected_raw="incomplete:max_output_tokens",
    )


def test_contract_usage_lands():
    usage = _usage(input_tokens=11, output_tokens=22)
    model = _model([_response([_message_item(_output_text_content("x"))], usage=usage)])
    contract.assert_usage_lands(model, model_id="gpt-5-test", expected_input=11, expected_output=22)


def test_contract_sdk_failure_raises_provider_error():
    def raiser(_kwargs):
        raise openai.APIConnectionError(message="boom", request=_httpx_request())

    model = _model(raiser)
    contract.assert_sdk_failure_raises_provider_error(model, model_id="gpt-5-test")


# --- shared conformance contract, async twins (acomplete) --------------------


def test_async_contract_text_completion():
    import asyncio

    model = _amodel([_response([_message_item(_output_text_content("hello there"))])])
    asyncio.run(
        contract.assert_async_text_completion(
            model, model_id="gpt-5-test", expected_text="hello there"
        )
    )


def test_async_contract_tool_call():
    import asyncio

    model = _amodel([_response([_function_call_item("c1", "search", {})])])
    asyncio.run(contract.assert_async_tool_call(model, model_id="gpt-5-test", tool_name="search"))


def test_async_contract_tool_result_batching():
    import asyncio

    model = _amodel([_response([_message_item(_output_text_content("ok"))])])
    asyncio.run(
        contract.assert_async_tool_result_batching_survives_round_trip(
            model, model_id="gpt-5-test", tool_call_id="tc1"
        )
    )


def test_async_contract_structured_output():
    import asyncio

    model = _amodel([_response([_message_item(_output_text_content('{"answer": 42}'))])])
    asyncio.run(
        contract.assert_async_structured_output(
            model, model_id="gpt-5-test", expected={"answer": 42}
        )
    )


def test_async_contract_tool_use_with_output_schema_does_not_crash():
    import asyncio

    model = _amodel([_response([_function_call_item("c1", "search", {})])])
    asyncio.run(
        contract.assert_async_tool_use_with_output_schema_does_not_crash(
            model, model_id="gpt-5-test", tool_name="search"
        )
    )


def test_async_contract_stop_reason_mapping():
    import asyncio

    model = _amodel(
        [
            _response(
                [_message_item(_output_text_content("partial"))],
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        ]
    )
    asyncio.run(
        contract.assert_async_stop_reason_mapping(
            model,
            model_id="gpt-5-test",
            expected=StopReason.MAX_TOKENS,
            expected_raw="incomplete:max_output_tokens",
        )
    )


def test_async_contract_usage_lands():
    import asyncio

    usage = _usage(input_tokens=11, output_tokens=22)
    model = _amodel([_response([_message_item(_output_text_content("x"))], usage=usage)])
    asyncio.run(
        contract.assert_async_usage_lands(
            model, model_id="gpt-5-test", expected_input=11, expected_output=22
        )
    )


def test_async_contract_sdk_failure_raises_provider_error():
    import asyncio

    def raiser(_kwargs):
        raise openai.APIConnectionError(message="boom", request=_httpx_request())

    model = _amodel(raiser)
    asyncio.run(
        contract.assert_async_sdk_failure_raises_provider_error(model, model_id="gpt-5-test")
    )


def test_openai_model_timeout_passed_to_sdk_client(monkeypatch):
    import openai

    from composeai.models.openai import OpenAIModel

    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    model = OpenAIModel("gpt-5.5", api_key="k", timeout=45)
    model._get_client()
    assert captured["timeout"] == 45


# --- async twins (acomplete / astream) ---------------------------------------


def test_acomplete_maps_response_like_complete():
    import asyncio

    output = [_message_item(_output_text_content("hello there"))]
    usage = _usage(input_tokens=11, output_tokens=22)

    # complete() path
    complete_model = _model([_response(output, usage=usage)])
    complete_resp = complete_model.complete(
        ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
    )

    # acomplete() path, same wire response
    async_model = _amodel([_response(output, usage=usage)])

    async def drive():
        return await async_model.acomplete(
            ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
        )

    async_resp = asyncio.run(drive())

    assert async_resp.message.text == complete_resp.message.text
    assert async_resp.stop_reason == complete_resp.stop_reason
    assert async_resp.usage == complete_resp.usage


def test_astream_events_match_stream():
    import asyncio

    final = _response([_message_item(_output_text_content("hello there"))])
    events = [_stream_text_delta("hello"), _stream_text_delta(" there"), _stream_completed(final)]

    # stream() path
    sync_model = _model_stream([_StubResponseStream(events, final)])
    sync_events = list(
        sync_model.stream(ModelRequest(model="gpt-5-test", messages=[Message.user("hi")]))
    )

    # astream() path, same scripted event sequence
    async_model = _amodel_stream([_AsyncStubResponseStream(events, final)])

    async def drive():
        result = []
        async for event in async_model.astream(
            ModelRequest(model="gpt-5-test", messages=[Message.user("hi")])
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
    import openai

    from composeai.models.openai import OpenAIModel

    captured: dict = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    model = OpenAIModel("gpt-5-test", api_key="k", timeout=45)
    model._get_async_client()
    assert captured["api_key"] == "k"
    assert captured["timeout"] == 45

    captured_no_timeout: dict = {}

    class FakeAsyncOpenAINoTimeout:
        def __init__(self, **kwargs):
            captured_no_timeout.update(kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAINoTimeout)
    model_no_timeout = OpenAIModel("gpt-5-test", api_key="k")
    model_no_timeout._get_async_client()
    assert "timeout" not in captured_no_timeout


# --- injected sync client vs. the async engine's acomplete/astream discovery -


def test_injected_sync_client_used_by_async_engine():
    """A ``client=`` injected at construction is a SYNC client -- the async
    engine's ``getattr(model, "acomplete", None)`` discovery (models/base.py)
    must not route to ``acomplete``/``astream`` (which would silently build
    an unrelated ``AsyncOpenAI`` from env/api_key instead, see
    ``OpenAIModel.__init__``). It must fall back to running the injected
    client's sync ``complete()`` off-thread, exactly as it would for a model
    with no async methods at all.
    """
    from composeai.agentfn import agent

    model = _model([_response([_message_item(_output_text_content("Hello there."))])])
    assert model.acomplete is None
    assert model.astream is None

    @agent(model=model, max_turns=2, name="openai_injected_client_agent")
    def greeter(name: str) -> str:
        """You are a friendly greeter."""
        return f"Greet {name}"

    run = greeter.run("Ann")

    assert run.output == "Hello there."
    assert run.status == "completed"
    assert len(model._client.responses.calls) == 1
