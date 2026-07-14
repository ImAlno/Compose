"""Tests for the OpenAI-compatible (Chat Completions) adapter, using a duck-typed fake client.

Wire shapes were verified against the installed ``openai`` SDK
(``openai/types/chat/*.py``) rather than assumed from memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from conformance import contract
from pydantic import BaseModel

from composeai.errors import ProviderError
from composeai.messages import (
    ImagePart,
    Message,
    StopReason,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from composeai.models.base import ModelRequest, ToolSpec
from composeai.models.compatible import OpenAICompatibleModel, openai_compatible
from composeai.models.prices import ModelPrice, register_price

# --- Stub client helpers -----------------------------------------------------


def _prompt_tokens_details(cached_tokens: int = 0, cache_write_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens)


def _completion_tokens_details(reasoning_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(reasoning_tokens=reasoning_tokens)


def _usage(
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=_prompt_tokens_details(
            cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens
        ),
        completion_tokens_details=_completion_tokens_details(reasoning_tokens=reasoning_tokens),
    )


def _tool_call(id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


def _message(
    content: str | None = None, tool_calls=None, refusal: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, tool_calls=tool_calls, refusal=refusal, role="assistant"
    )


def _choice(message: SimpleNamespace, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(message=message, finish_reason=finish_reason, index=0)


def _response(
    choices: list[SimpleNamespace], usage: SimpleNamespace | None = None
) -> SimpleNamespace:
    return SimpleNamespace(choices=choices, usage=usage if usage is not None else _usage())


class _FakeSDKStream:
    """Duck-typed stand-in for the ``openai`` SDK's ``Stream`` object -- what
    ``client.chat.completions.create(stream=True)`` really returns.

    The real ``Stream`` is itself a context manager (``__enter__`` returns
    self; ``__exit__``/``close()`` release the underlying HTTP response) as
    well as a plain iterator over chunks -- this mirrors both so the adapter
    can be tested using ``with client.chat.completions.create(**kwargs) as
    stream:`` exactly as it runs against the real SDK.
    """

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    def __enter__(self) -> _FakeSDKStream:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.closed = True

    def __iter__(self) -> Iterator[Any]:
        return iter(self._chunks)


class _StubCompletions:
    """Duck-typed stand-in for ``openai.OpenAI().chat.completions``."""

    def __init__(self, responses) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []
        self.last_stream: _FakeSDKStream | None = None

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if callable(self._responses):
            result = self._responses(kwargs)
        else:
            result = self._responses.pop(0)
        # A streaming test scripts a bare list of chunks (see the `_chunk()`
        # helpers below) where a non-streaming test scripts a single
        # `_response(...)` object -- wrap only the former, matching the real
        # SDK's `create(stream=True)` -> `Stream` vs. `create()` -> `ChatCompletion`.
        if isinstance(result, list):
            self.last_stream = _FakeSDKStream(result)
            return self.last_stream
        return result


class _StubChat:
    def __init__(self, responses) -> None:
        self.completions = _StubCompletions(responses)


class _StubClient:
    def __init__(self, responses) -> None:
        self.chat = _StubChat(responses)


def _model(
    responses,
    model_id: str = "llama3-test",
    base_url: str = "http://localhost:11434/v1",
) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(model_id, base_url=base_url, client=_StubClient(responses))


# --- stream() stub helpers ----------------------------------------------------


def _delta(
    content: str | None = None, tool_calls=None, refusal: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls, refusal=refusal, role=None)


def _tool_call_delta(
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    function = (
        SimpleNamespace(name=name, arguments=arguments)
        if (name is not None or arguments is not None)
        else None
    )
    return SimpleNamespace(
        index=index, id=id, type="function" if id else None, function=function
    )


def _chunk(delta: SimpleNamespace, finish_reason: str | None = None) -> SimpleNamespace:
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(choices=[choice], usage=None)


def _usage_only_chunk(usage: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(choices=[], usage=usage)


# --- factory / api_key default -----------------------------------------------


def test_openai_compatible_returns_model_satisfying_protocol():
    from composeai.models.base import Model

    model = openai_compatible("http://localhost:11434/v1", "llama3-test")
    assert isinstance(model, Model)
    assert isinstance(model, OpenAICompatibleModel)


def test_openai_compatible_factory_sets_model_id_and_base_url():
    model = openai_compatible("http://localhost:8000/v1", "some-model")
    assert isinstance(model, OpenAICompatibleModel)
    assert model.model_id == "some-model"
    assert model._base_url == "http://localhost:8000/v1"


def test_default_api_key_is_unused_string(monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = _StubChat([_response([_choice(_message(content="hi"))])])

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    model = OpenAICompatibleModel("llama3-test", base_url="http://localhost:11434/v1")
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    model.complete(request)
    assert captured["api_key"] == "unused"
    assert captured["base_url"] == "http://localhost:11434/v1"


def test_explicit_api_key_passed_through(monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = _StubChat([_response([_choice(_message(content="hi"))])])

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    model = OpenAICompatibleModel(
        "llama3-test", base_url="http://localhost:11434/v1", api_key="sk-real-key"
    )
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    model.complete(request)
    assert captured["api_key"] == "sk-real-key"


def test_injected_client_bypasses_key_entirely():
    model = _model([_response([_choice(_message(content="hi"))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.message.text == "hi"


# --- request mapping: system / messages --------------------------------------


def test_system_maps_to_system_role_message():
    model = _model([_response([_choice(_message(content="ok"))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")], system="be terse")
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["messages"][0] == {"role": "system", "content": "be terse"}


def test_system_omitted_when_none():
    model = _model([_response([_choice(_message(content="ok"))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert all(m.get("role") != "system" for m in call["messages"])


def test_user_text_part_maps_to_user_message():
    model = _model([_response([_choice(_message(content="ok"))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hello")])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_user_image_part_maps_to_image_url_content():
    model = _model([_response([_choice(_message(content="ok"))])])
    image = ImagePart(media_type="image/png", url="https://example.com/x.png")
    request = ModelRequest(model="llama3-test", messages=[Message.user([image])])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    part = call["messages"][0]["content"][0]
    assert part == {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}


def test_user_image_part_base64_maps_to_data_url():
    model = _model([_response([_choice(_message(content="ok"))])])
    image = ImagePart(media_type="image/png", data="YWJj")
    request = ModelRequest(model="llama3-test", messages=[Message.user([image])])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    part = call["messages"][0]["content"][0]
    assert part["image_url"]["url"] == "data:image/png;base64,YWJj"


def test_assistant_tool_call_part_maps_to_nested_tool_calls():
    model = _model([_response([_choice(_message(content="ok"))])])
    call_part = ToolCallPart(id="tc1", name="search", arguments={"q": "cats"})
    request = ModelRequest(model="llama3-test", messages=[Message.assistant([call_part])])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    msg = call["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["tool_calls"] == [
        {
            "id": "tc1",
            "type": "function",
            "function": {"name": "search", "arguments": json.dumps({"q": "cats"})},
        }
    ]


def test_tool_result_part_maps_to_tool_role_message():
    model = _model([_response([_choice(_message(content="ok"))])])
    result = ToolResultPart(tool_call_id="tc1", content="42")
    request = ModelRequest(model="llama3-test", messages=[Message.user([result])])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["messages"][0] == {"role": "tool", "tool_call_id": "tc1", "content": "42"}


def test_tool_result_part_with_is_error_prepends_error_prefix():
    model = _model([_response([_choice(_message(content="ok"))])])
    result = ToolResultPart(tool_call_id="tc1", content="boom", is_error=True)
    request = ModelRequest(model="llama3-test", messages=[Message.user([result])])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["messages"][0]["content"] == "ERROR: boom"


def test_multiple_tool_results_become_separate_tool_messages():
    model = _model([_response([_choice(_message(content="ok"))])])
    request = ModelRequest(
        model="llama3-test",
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
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["messages"] == [
        {"role": "tool", "tool_call_id": "tc1", "content": "a"},
        {"role": "tool", "tool_call_id": "tc2", "content": "b"},
    ]


def test_thinking_parts_are_never_echoed_even_same_provider_and_model():
    part = ThinkingPart(text="hmm", provider="openai-compatible", provider_token="{}")
    model = _model([_response([_choice(_message(content="ok"))])], model_id="llama3-test")
    request = ModelRequest(
        model="llama3-test", messages=[Message.assistant([part, TextPart(text="hi")])]
    )
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["messages"] == [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]


# --- request mapping: tools / structured output -----------------------------


def test_tools_mapped_nested_under_function():
    model = _model([_response([_choice(_message(content="ok"))])])
    spec = ToolSpec(
        name="search",
        description="search the web",
        input_schema={"type": "object", "additionalProperties": False},
        strict=True,
    )
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")], tools=[spec])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search the web",
                "parameters": {"type": "object", "additionalProperties": False},
                "strict": True,
            },
        }
    ]


def test_tools_omitted_when_none():
    model = _model([_response([_choice(_message(content="ok"))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert "tools" not in call


def test_output_schema_maps_to_response_format_json_schema():
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
    model = _model([_response([_choice(_message(content='{"a": 1}'))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")], output_schema=schema)
    resp = model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": True},
    }
    assert resp.parsed == {"a": 1}


def test_output_schema_uses_title_as_name_when_valid():
    schema = {"type": "object", "title": "MyAnswer"}
    model = _model([_response([_choice(_message(content='{"a": 1}'))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")], output_schema=schema)
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["response_format"]["json_schema"]["name"] == "MyAnswer"


# --- request mapping: strict-mode schema compatibility -----------------------
#
# Same rationale as models/openai.py's identical tests: a real
# OpenAI-compatible server that enforces strict JSON-schema validation
# (vLLM guided decoding, or a real OpenAI endpoint reached via base_url)
# rejects these shapes exactly like the Responses API does.


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
    model = _model([_response([_choice(_message(content="ok"))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")], tools=[spec])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["tools"][0]["function"]["strict"] is False
    assert call["tools"][0]["function"]["parameters"] == schema


def test_schema_valued_additional_properties_downgrades_tool_to_non_strict():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "object", "additionalProperties": {"type": "string"}}},
        "required": ["tags"],
        "additionalProperties": False,
    }
    spec = ToolSpec(name="tag", description="tag it", input_schema=schema, strict=True)
    model = _model([_response([_choice(_message(content="ok"))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")], tools=[spec])
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["tools"][0]["function"]["strict"] is False
    assert call["tools"][0]["function"]["parameters"] == schema


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
    model = _model([_response([_choice(_message(content='{"answer": "x"}'))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")], output_schema=schema)
    model.complete(request)
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["response_format"]["json_schema"]["strict"] is False
    assert call["response_format"]["json_schema"]["schema"] == schema


def test_non_json_content_with_output_schema_raises_provider_error():
    model = _model([_response([_choice(_message(content="not json"))])])
    request = ModelRequest(
        model="llama3-test", messages=[Message.user("hi")], output_schema={"type": "object"}
    )
    with pytest.raises(ProviderError):
        model.complete(request)


def test_tool_calls_stop_with_output_schema_does_not_crash_and_parsed_is_none():
    # Regression: an @agent combining tools=[...] with a non-str output_type
    # sends output_schema on every turn, so a tool_calls-only completion has
    # message.content == None (message.text == '') -- json.loads('') must
    # not be attempted (it would raise and mask the real TOOL_USE stop
    # reason behind a spurious ProviderError).
    tool_calls = [_tool_call("call-1", "search", {})]
    choice = _choice(_message(tool_calls=tool_calls), finish_reason="tool_calls")
    model = _model([_response([choice])])
    request = ModelRequest(
        model="llama3-test", messages=[Message.user("hi")], output_schema={"type": "object"}
    )
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.TOOL_USE
    assert resp.parsed is None


def test_length_stop_with_output_schema_does_not_crash_and_parsed_is_none():
    model = _model(
        [_response([_choice(_message(content='{"partial": tr'), finish_reason="length")])]
    )
    request = ModelRequest(
        model="llama3-test", messages=[Message.user("hi")], output_schema={"type": "object"}
    )
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.MAX_TOKENS
    assert resp.parsed is None


def test_server_rejecting_response_format_surfaces_as_provider_error():
    def raiser(_kwargs):
        req = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
        resp = httpx.Response(400, request=req, json={"error": {"message": "unsupported"}})
        raise openai.APIStatusError("unsupported", response=resp, body=None)

    model = _model(raiser)
    request = ModelRequest(
        model="llama3-test", messages=[Message.user("hi")], output_schema={"type": "object"}
    )
    with pytest.raises(ProviderError):
        model.complete(request)


# --- response mapping: tool calls / text -------------------------------------


def test_content_maps_to_text_part():
    model = _model([_response([_choice(_message(content="hello there"))])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.message.text == "hello there"


def test_malformed_tool_call_arguments_raise_provider_error():
    bad_tool_call = SimpleNamespace(
        id="call-1", type="function", function=SimpleNamespace(name="search", arguments="not json")
    )
    choice = _choice(_message(tool_calls=[bad_tool_call]), finish_reason="tool_calls")
    model = _model([_response([choice])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError):
        model.complete(request)


def test_tool_calls_map_to_tool_call_parts():
    tool_calls = [_tool_call("call-1", "search", {"q": "cats"})]
    choice = _choice(_message(tool_calls=tool_calls), finish_reason="tool_calls")
    model = _model([_response([choice])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    calls = [p for p in resp.message.parts if isinstance(p, ToolCallPart)]
    assert len(calls) == 1
    assert calls[0].id == "call-1"
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "cats"}
    assert resp.stop_reason == StopReason.TOOL_USE


# --- response mapping: finish_reason table ------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("stop", StopReason.END_TURN),
        ("length", StopReason.MAX_TOKENS),
        ("tool_calls", StopReason.TOOL_USE),
        ("content_filter", StopReason.REFUSAL),
        ("function_call", StopReason.OTHER),
        ("some_future_reason", StopReason.OTHER),
    ],
)
def test_finish_reason_mapping_table(raw, expected):
    model = _model([_response([_choice(_message(content="x"), finish_reason=raw)])])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.stop_reason == expected
    assert resp.raw_stop_reason == raw


def test_message_refusal_field_maps_to_refusal_regardless_of_finish_reason():
    # A genuine model refusal typically comes back with finish_reason still
    # "stop" (not "content_filter") and the refusal text on the dedicated
    # `message.refusal` field -- composeai must classify this as REFUSAL
    # (and surface the refusal text as the message content) rather than
    # silently treating it as a normal successful END_TURN answer, which
    # would let a structured-output request feed the refusal text straight
    # into json.loads and fail with a confusing ProviderError instead of
    # ModelRefusalError.
    model = _model(
        [_response([_choice(_message(refusal="I can't help with that"), finish_reason="stop")])]
    )
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.stop_reason == StopReason.REFUSAL
    assert resp.message.text == "I can't help with that"


# --- response mapping: usage / cost ------------------------------------------


def test_usage_tokens_land_in_usage():
    usage = _usage(prompt_tokens=123, completion_tokens=45)
    model = _model([_response([_choice(_message(content="x"))], usage=usage)])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.input_tokens == 123
    assert resp.usage.output_tokens == 45


def test_cached_tokens_and_reasoning_tokens_land_when_present():
    usage = _usage(cached_tokens=5, reasoning_tokens=9)
    model = _model([_response([_choice(_message(content="x"))], usage=usage)])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cache_read_tokens == 5
    assert resp.usage.reasoning_tokens == 9


def test_cache_write_tokens_map_to_cache_creation_tokens():
    # `prompt_tokens_details.cache_write_tokens` genuinely exists on the
    # wire (verified against the installed SDK's
    # `openai.types.completion_usage.PromptTokensDetails`, which documents
    # it as "The unadjusted number of prompt tokens written to cache") --
    # unlike models/openai.py's Responses adapter (which the project
    # deliberately keeps at cache_creation_tokens=0, see
    # test_cache_creation_tokens_always_zero over there), this field is
    # mapped through here per the brief's "if a cache-write field genuinely
    # exists in the SDK types, map it" instruction.
    usage = _usage(cache_write_tokens=42)
    model = _model([_response([_choice(_message(content="x"))], usage=usage)])
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cache_creation_tokens == 42


def test_unpriced_model_cost_none_and_incomplete():
    model = _model([_response([_choice(_message(content="x"))])], model_id="totally-unpriced-model")
    request = ModelRequest(model="totally-unpriced-model", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cost_usd is None
    assert resp.usage.cost_complete is False


def test_registered_price_under_openai_compatible_key_is_used():
    register_price("openai-compatible", "priced-local-model", ModelPrice(input=1.0, output=2.0))
    usage = _usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    model = _model(
        [_response([_choice(_message(content="x"))], usage=usage)], model_id="priced-local-model"
    )
    request = ModelRequest(model="priced-local-model", messages=[Message.user("hi")])
    resp = model.complete(request)
    assert resp.usage.cost_usd == pytest.approx(1.0 + 2.0)
    assert resp.usage.cost_complete is True


def test_cached_tokens_are_not_double_billed():
    # Regression for the "double-billed cached input tokens" critical:
    # `prompt_tokens` is documented by the installed SDK's
    # `openai.types.completion_usage.PromptTokensDetails` as a "breakdown of
    # tokens used in the prompt" -- i.e. `cached_tokens` is a SUBSET of
    # `prompt_tokens`, not additive (confirmed unambiguously by the
    # structurally-identical `openai.types.realtime.
    # RealtimeResponseUsageInputTokenDetails` docstring: "Cached tokens here
    # are counted as a subset of input tokens, meaning input tokens will
    # include cached and uncached tokens"). Billing the full 1,000,000
    # prompt_tokens at the input rate AND separately billing the 800,000
    # cached_tokens at the cache_read rate would charge the cached slice
    # twice. Correct cost = (prompt_tokens - cached_tokens) * input_price +
    # cached_tokens * cache_read_price + completion_tokens * output_price.
    register_price(
        "openai-compatible",
        "cache-billing-test-model",
        ModelPrice(input=2.0, output=8.0, cache_read=0.2),
    )
    usage = _usage(prompt_tokens=1_000_000, completion_tokens=100_000, cached_tokens=800_000)
    model = _model(
        [_response([_choice(_message(content="x"))], usage=usage)],
        model_id="cache-billing-test-model",
    )
    request = ModelRequest(model="cache-billing-test-model", messages=[Message.user("hi")])
    resp = model.complete(request)
    expected = (200_000 * 2.0 + 800_000 * 0.2 + 100_000 * 8.0) / 1_000_000
    assert resp.usage.cost_usd == pytest.approx(expected)
    # Sanity: the naive (double-billing) computation would give a visibly
    # different, larger number -- 1_000_000*2.0 + 800_000*0.2 +
    # 100_000*8.0, i.e. 2.96 -- so this test would fail loudly, not by a
    # rounding error, if the fix regresses.
    assert resp.usage.cost_usd != pytest.approx(2.96)


# --- SDK error mapping --------------------------------------------------------


def test_api_connection_error_becomes_provider_error_with_compatible_provider_name():
    sdk_error = openai.APIConnectionError(
        message="connection boom", request=httpx.Request("POST", "http://localhost:11434/v1")
    )

    def raiser(_kwargs):
        raise sdk_error

    model = _model(raiser)
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError) as excinfo:
        model.complete(request)
    assert excinfo.value.provider == "openai-compatible"
    assert excinfo.value.__cause__ is sdk_error


def test_no_retry_loop_added_on_top_of_sdk():
    calls = []

    def raiser(_kwargs):
        calls.append(1)
        raise openai.APIConnectionError(
            message="boom", request=httpx.Request("POST", "http://localhost:11434/v1")
        )

    model = _model(raiser)
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError):
        model.complete(request)
    assert len(calls) == 1


# --- stream() ------------------------------------------------------------


def test_stream_yields_text_deltas_then_response_done():
    chunks = [
        _chunk(_delta(content="hello")),
        _chunk(_delta(content=" there")),
        _chunk(_delta(), finish_reason="stop"),
        _usage_only_chunk(_usage(prompt_tokens=10, completion_tokens=20)),
    ]
    model = _model([chunks])
    raw_events = list(
        model.stream(ModelRequest(model="llama3-test", messages=[Message.user("hi")]))
    )

    kinds = [e.kind for e in raw_events]
    assert kinds == ["text_delta", "text_delta", "response_done"]
    assert [e.text for e in raw_events[:2]] == ["hello", " there"]
    done = raw_events[-1]
    assert done.response is not None
    assert done.response.message.text == "hello there"
    assert done.response.stop_reason == StopReason.END_TURN
    assert done.response.usage.input_tokens == 10
    assert done.response.usage.output_tokens == 20


def test_stream_tool_call_yields_started_args_deltas_and_finished_at_finish_reason():
    chunks = [
        _chunk(_delta(tool_calls=[_tool_call_delta(0, id="call-1", name="search", arguments="")])),
        _chunk(_delta(tool_calls=[_tool_call_delta(0, arguments='{"q": ')])),
        _chunk(_delta(tool_calls=[_tool_call_delta(0, arguments='"cats"}')])),
        _chunk(_delta(), finish_reason="tool_calls"),
        _usage_only_chunk(_usage()),
    ]
    model = _model([chunks])
    raw_events = list(
        model.stream(ModelRequest(model="llama3-test", messages=[Message.user("hi")]))
    )

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


def test_stream_requests_include_usage_stream_option():
    chunks = [_chunk(_delta(), finish_reason="stop"), _usage_only_chunk(_usage())]
    model = _model([chunks])
    list(model.stream(ModelRequest(model="llama3-test", messages=[Message.user("hi")])))
    call = model._client.chat.completions.calls[0]  # type: ignore[attr-defined]
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}


def test_stream_uses_sdk_stream_as_a_context_manager():
    # Minor-findings cleanup (Phase 10): `create(stream=True)` returns the
    # SDK's own `Stream`, which is itself a context manager (releases the
    # underlying HTTP connection on `__exit__`) -- use it as one instead of
    # just iterating over it, so the connection is deterministically closed.
    chunks = [_chunk(_delta(content="hi"), finish_reason="stop"), _usage_only_chunk(_usage())]
    model = _model([chunks])
    list(model.stream(ModelRequest(model="llama3-test", messages=[Message.user("hi")])))
    stream = model._client.chat.completions.last_stream  # type: ignore[attr-defined]
    assert stream is not None
    assert stream.closed is True


def test_stream_degrades_to_zero_usage_when_server_omits_usage_chunk():
    register_price("openai-compatible", "no-usage-model", ModelPrice(input=1.0, output=2.0))
    chunks = [_chunk(_delta(content="hi"), finish_reason="stop")]  # no usage chunk at all
    model = _model([chunks], model_id="no-usage-model")
    raw_events = list(
        model.stream(ModelRequest(model="no-usage-model", messages=[Message.user("hi")]))
    )
    done_response = raw_events[-1].response
    assert done_response is not None
    usage = done_response.usage
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cost_usd is None
    assert usage.cost_complete is False


def test_stream_response_done_equals_complete_style_mapping():
    complete_model = _model([_response([_choice(_message(content="hello there"))])])
    complete_resp = complete_model.complete(
        ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    )

    chunks = [
        _chunk(_delta(content="hello there")),
        _chunk(_delta(), finish_reason="stop"),
        _usage_only_chunk(_usage()),
    ]
    stream_model = _model([chunks])
    raw_events = list(
        stream_model.stream(ModelRequest(model="llama3-test", messages=[Message.user("hi")]))
    )
    streamed_resp = raw_events[-1].response
    assert streamed_resp is not None

    assert streamed_resp.message.text == complete_resp.message.text
    assert streamed_resp.stop_reason == complete_resp.stop_reason
    assert streamed_resp.usage == complete_resp.usage


def test_stream_refusal_maps_to_refusal_stop_reason():
    chunks = [
        _chunk(_delta(refusal="I can't help")),
        _chunk(_delta(), finish_reason="content_filter"),
        _usage_only_chunk(_usage()),
    ]
    model = _model([chunks])
    raw_events = list(
        model.stream(ModelRequest(model="llama3-test", messages=[Message.user("hi")]))
    )
    done = raw_events[-1]
    assert done.response is not None
    assert done.response.stop_reason == StopReason.REFUSAL
    assert done.response.message.text == "I can't help"


def test_stream_prompt_mode_lenient_parse():
    """stream() shares _build_kwargs/_finalize -- prompt mode must work there too."""
    chunks = [
        _chunk(_delta(content='```json\n{"answer": "ok"}\n```')),
        _chunk(_delta(), finish_reason="stop"),
        _usage_only_chunk(_usage()),
    ]
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=_StubClient([chunks]), schema_mode="prompt"
    )
    raw_events = list(model.stream(_structured_request()))
    done = raw_events[-1]
    assert done.kind == "response_done"
    assert done.response is not None
    assert done.response.parsed == {"answer": "ok"}


def test_stream_sdk_error_becomes_provider_error():
    sdk_error = openai.APIConnectionError(
        message="boom", request=httpx.Request("POST", "http://localhost:11434/v1")
    )

    def raiser(_kwargs):
        raise sdk_error

    model = _model(raiser)
    request = ModelRequest(model="llama3-test", messages=[Message.user("hi")])
    with pytest.raises(ProviderError) as excinfo:
        list(model.stream(request))
    assert excinfo.value.__cause__ is sdk_error


# --- protocol conformance ------------------------------------------------------


def test_openai_compatible_model_satisfies_model_protocol():
    from composeai.models.base import Model

    assert isinstance(_model([_response([_choice(_message(content="ok"))])]), Model)


# --- shared conformance contract ---------------------------------------------


def test_contract_text_completion():
    model = _model([_response([_choice(_message(content="hello there"))])])
    contract.assert_text_completion(model, model_id="llama3-test", expected_text="hello there")


def test_contract_tool_call():
    tool_calls = [_tool_call("call-1", "search", {})]
    choice = _choice(_message(tool_calls=tool_calls), finish_reason="tool_calls")
    model = _model([_response([choice])])
    contract.assert_tool_call(model, model_id="llama3-test", tool_name="search")


def test_contract_tool_result_batching():
    model = _model([_response([_choice(_message(content="ok"))])])
    contract.assert_tool_result_batching_survives_round_trip(
        model, model_id="llama3-test", tool_call_id="tc1"
    )


def test_contract_structured_output():
    model = _model([_response([_choice(_message(content='{"answer": 42}'))])])
    contract.assert_structured_output(model, model_id="llama3-test", expected={"answer": 42})


def test_contract_tool_use_with_output_schema_does_not_crash():
    tool_calls = [_tool_call("call-1", "search", {})]
    choice = _choice(_message(tool_calls=tool_calls), finish_reason="tool_calls")
    model = _model([_response([choice])])
    contract.assert_tool_use_with_output_schema_does_not_crash(
        model, model_id="llama3-test", tool_name="search"
    )


def test_contract_stop_reason_mapping():
    model = _model([_response([_choice(_message(content="x"), finish_reason="length")])])
    contract.assert_stop_reason_mapping(
        model, model_id="llama3-test", expected=StopReason.MAX_TOKENS, expected_raw="length"
    )


def test_contract_usage_lands():
    usage = _usage(prompt_tokens=11, completion_tokens=22)
    model = _model([_response([_choice(_message(content="x"))], usage=usage)])
    contract.assert_usage_lands(
        model, model_id="llama3-test", expected_input=11, expected_output=22
    )


def test_contract_sdk_failure_raises_provider_error():
    def raiser(_kwargs):
        raise openai.APIConnectionError(
            message="boom", request=httpx.Request("POST", "http://localhost:11434/v1")
        )

    model = _model(raiser)
    contract.assert_sdk_failure_raises_provider_error(model, model_id="llama3-test")


def test_timeout_passed_to_sdk_client(monkeypatch):
    import openai

    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    model = OpenAICompatibleModel("llama3.2", base_url="http://localhost:11434/v1", timeout=90)
    model._get_client()
    assert captured["timeout"] == 90


def test_no_timeout_omits_sdk_kwarg(monkeypatch):
    import openai

    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    model = OpenAICompatibleModel("llama3.2", base_url="http://localhost:11434/v1")
    model._get_client()
    assert "timeout" not in captured


def test_construction_prices_register_for_cost_tracking():
    from composeai.models.prices import get_price

    openai_compatible(
        "http://x/v1", "kimi-priced-task2", input_price=0.6, output_price=2.5
    )
    price = get_price("openai-compatible", "kimi-priced-task2")
    assert price is not None
    assert price.input == 0.6
    assert price.output == 2.5


def test_construction_price_requires_both_sides():
    from composeai.errors import ConfigError

    with pytest.raises(ConfigError):
        openai_compatible("http://x/v1", "kimi-halfpriced-task2", input_price=0.6)


# --- schema_mode="prompt" ------------------------------------------------------

_TASK3_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _prompt_mode_client(content: str, captured: dict):
    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content=content, refusal=None, tool_calls=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                usage=None,
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def _structured_request() -> ModelRequest:
    return ModelRequest(
        model="m",
        messages=[Message.user("hi")],
        system=None,
        tools=None,
        output_schema=_TASK3_SCHEMA,
        max_tokens=100,
        temperature=None,
        provider="openai-compatible",
    )


def test_prompt_mode_embeds_schema_and_drops_response_format():
    captured: dict = {}
    client = _prompt_mode_client('{"answer": "ok"}', captured)
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=client, schema_mode="prompt"
    )
    response = model.complete(_structured_request())
    assert "response_format" not in captured
    last = captured["messages"][-1]
    assert last["role"] == "user"
    assert "JSON Schema" in last["content"][-1]["text"]
    assert response.parsed == {"answer": "ok"}


def test_prompt_mode_strips_markdown_fences():
    captured: dict = {}
    client = _prompt_mode_client('```json\n{"answer": "ok"}\n```', captured)
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=client, schema_mode="prompt"
    )
    response = model.complete(_structured_request())
    assert response.parsed == {"answer": "ok"}


def test_prompt_mode_extracts_first_balanced_object_from_prose():
    captured: dict = {}
    client = _prompt_mode_client(
        'Sure! Here is the JSON you asked for: {"answer": "ok"} Hope that helps.',
        captured,
    )
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=client, schema_mode="prompt"
    )
    response = model.complete(_structured_request())
    assert response.parsed == {"answer": "ok"}


def test_prompt_mode_skips_non_json_braces_before_object():
    captured: dict = {}
    client = _prompt_mode_client(
        'use {placeholder} then {"answer": "ok"}',
        captured,
    )
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=client, schema_mode="prompt"
    )
    response = model.complete(_structured_request())
    assert response.parsed == {"answer": "ok"}


def test_prompt_mode_unparseable_returns_parsed_none():
    captured: dict = {}
    client = _prompt_mode_client("no json here at all", captured)
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=client, schema_mode="prompt"
    )
    response = model.complete(_structured_request())
    assert response.parsed is None
    assert response.message.text == "no json here at all"


def test_prompt_mode_empty_content_reasoning_error_still_raises():
    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content=None, refusal=None, tool_calls=None)
            usage = SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=500,
                prompt_tokens_details=None,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=500),
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                usage=usage,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=client, schema_mode="prompt"
    )
    with pytest.raises(ProviderError, match="only reasoning tokens"):
        model.complete(_structured_request())


class _Task2Answer(BaseModel):
    # Module-level (not local to the test): a locally-defined pydantic model
    # would have its own string annotations (this file uses `from __future__
    # import annotations`) fail to resolve later, from inside agentfn.py, since
    # get_type_hints() can't see an enclosing test function's local scope --
    # see composeai._schema.resolve_annotations's docstring for the same issue
    # one level up (the agent function's own `-> _Task2Answer` return
    # annotation), which module-level placement sidesteps for both.
    answer: str


def test_prompt_mode_garbage_is_repairable_via_max_repairs():
    from composeai import prompt
    from composeai.agentfn import agent

    calls: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                content = "Sure, here's some prose with no JSON object in it at all."
            else:
                content = '{"answer": "ok"}'
            msg = SimpleNamespace(content=content, refusal=None, tool_calls=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                usage=None,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=client, schema_mode="prompt"
    )

    @agent(model=model, max_repairs=1)
    def task2_answerer(question: str) -> _Task2Answer:
        return prompt(question)

    result = task2_answerer("go")
    assert result == _Task2Answer(answer="ok")
    assert len(calls) == 2
    assert "did not match the required output schema" in str(calls[1]["messages"])


def test_append_schema_instruction_no_user_message_appends_one():
    from composeai.models.compatible import _append_schema_instruction

    api_messages: list[dict[str, Any]] = [{"role": "system", "content": "be brief"}]
    _append_schema_instruction(api_messages, {"type": "object"})
    assert api_messages[-1]["role"] == "user"
    assert "JSON Schema" in api_messages[-1]["content"][0]["text"]


def test_invalid_schema_mode_raises_config_error():
    from composeai.errors import ConfigError

    with pytest.raises(ConfigError):
        OpenAICompatibleModel("m", base_url="http://x/v1", schema_mode="bogus")


def test_native_mode_unchanged_sends_response_format():
    captured: dict = {}
    client = _prompt_mode_client('{"answer": "ok"}', captured)
    model = OpenAICompatibleModel("m", base_url="http://x/v1", client=client)
    model.complete(_structured_request())
    assert captured["response_format"]["type"] == "json_schema"


def test_reasoning_only_empty_content_raises_targeted_error():
    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content=None, refusal=None, tool_calls=None)
            usage = SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=500,
                prompt_tokens_details=None,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=500),
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                usage=usage,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    model = OpenAICompatibleModel("m", base_url="http://x/v1", client=client)
    with pytest.raises(ProviderError, match="only reasoning tokens"):
        model.complete(_structured_request())


# --- schema_mode="auto" -------------------------------------------------------


class _AutoModeCompletions:
    """Returns prose (ignoring response_format) when asked natively; valid JSON
    is only ever in the content, so success depends on prompt-mode parsing.

    Each wire call gets its own distinct, real usage (prompt_tokens/
    completion_tokens, with no cache/reasoning details) so a test can verify
    the demoted native call's "wasted" usage isn't silently discarded -- it
    must be summed into the retry's usage, not dropped, or trace/costs/
    Budget would under-count one full billed completion.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        call_number = len(self.calls)
        msg = SimpleNamespace(
            content='Sure! {"answer": "ok"} hope that helps',
            refusal=None,
            tool_calls=None,
        )
        usage = SimpleNamespace(
            prompt_tokens=7 * call_number,
            completion_tokens=13 * call_number,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage
        )


def test_auto_mode_demotes_to_prompt_and_sticks():
    completions = _AutoModeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=client, schema_mode="auto"
    )

    first = model.complete(_structured_request())
    assert first.parsed == {"answer": "ok"}
    # call 1 was native (sent response_format), call 2 was the prompt-mode retry
    assert len(completions.calls) == 2
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]
    # The first (native) call was real, billed spend -- its usage must reach
    # the returned response summed with the retry's, not be discarded.
    assert first.usage.input_tokens == 7 * 1 + 7 * 2
    assert first.usage.output_tokens == 13 * 1 + 13 * 2

    second = model.complete(_structured_request())
    assert second.parsed == {"answer": "ok"}
    # demotion is sticky: no native attempt this time
    assert len(completions.calls) == 3
    assert "response_format" not in completions.calls[2]
    # Once demoted, only one wire call happens -- no wasted usage to sum.
    assert second.usage.input_tokens == 7 * 3
    assert second.usage.output_tokens == 13 * 3


def test_auto_mode_native_success_never_demotes():
    calls: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            msg = SimpleNamespace(content='{"answer": "ok"}', refusal=None, tool_calls=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    model = OpenAICompatibleModel(
        "m", base_url="http://x/v1", client=client, schema_mode="auto"
    )
    model.complete(_structured_request())
    model.complete(_structured_request())
    assert len(calls) == 2
    assert all("response_format" in c for c in calls)


def test_auto_mode_accepted_by_factory():
    model = openai_compatible("http://x/v1", "m", schema_mode="auto")
    assert isinstance(model, OpenAICompatibleModel)
