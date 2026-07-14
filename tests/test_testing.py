import json

import pytest

from composeai.errors import ComposeError
from composeai.messages import (
    Message,
    StopReason,
    TextPart,
    ToolCallPart,
    Usage,
)
from composeai.models.base import ModelRequest, ModelResponse, RawStreamEvent
from composeai.testing import FakeModel


def _req(model: str = "fake/model") -> ModelRequest:
    return ModelRequest(model=model, messages=[Message.user("hi")])


# --- str script item ---


def test_str_item_yields_text_response_end_turn():
    fake = FakeModel(["hello there"])
    resp = fake.complete(_req())
    assert resp.message.role == "assistant"
    assert resp.message.text == "hello there"
    assert resp.stop_reason == StopReason.END_TURN
    assert resp.raw_stop_reason == "end_turn"


# --- tool_calls dict script item ---


def test_tool_calls_item_yields_tool_use_parts():
    fake = FakeModel([{"tool_calls": [{"name": "search", "arguments": {"q": "cats"}}]}])
    resp = fake.complete(_req())
    assert resp.stop_reason == StopReason.TOOL_USE
    calls = [p for p in resp.message.parts if isinstance(p, ToolCallPart)]
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "cats"}
    assert isinstance(calls[0].id, str) and len(calls[0].id) > 0


def test_tool_calls_item_autogenerates_missing_ids():
    tool_calls = [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]
    fake = FakeModel([{"tool_calls": tool_calls}])
    resp = fake.complete(_req())
    calls = [p for p in resp.message.parts if isinstance(p, ToolCallPart)]
    assert calls[0].id != calls[1].id
    assert all(c.id for c in calls)


def test_tool_calls_item_respects_given_id():
    fake = FakeModel([{"tool_calls": [{"name": "a", "arguments": {}, "id": "explicit-id"}]}])
    resp = fake.complete(_req())
    calls = [p for p in resp.message.parts if isinstance(p, ToolCallPart)]
    assert calls[0].id == "explicit-id"


def test_tool_calls_item_with_optional_text_includes_text_part():
    fake = FakeModel([{"tool_calls": [{"name": "a", "arguments": {}}], "text": "let me check"}])
    resp = fake.complete(_req())
    texts = [p for p in resp.message.parts if isinstance(p, TextPart)]
    assert len(texts) == 1
    assert texts[0].text == "let me check"


# --- json dict script item ---


def test_json_item_sets_parsed_and_end_turn():
    fake = FakeModel([{"json": {"answer": 42}}])
    resp = fake.complete(_req())
    assert resp.parsed == {"answer": 42}
    assert resp.stop_reason == StopReason.END_TURN


def test_json_item_message_contains_json_as_text():
    fake = FakeModel([{"json": {"answer": 42}}])
    resp = fake.complete(_req())
    assert "42" in resp.message.text


# --- ModelResponse passthrough ---


def test_model_response_item_returned_as_is():
    canned = ModelResponse(
        message=Message.assistant("canned"),
        stop_reason=StopReason.MAX_TOKENS,
        raw_stop_reason="max_tokens",
        usage=Usage(input_tokens=7, output_tokens=8),
        model_id="fake/model",
    )
    fake = FakeModel([canned])
    resp = fake.complete(_req())
    assert resp is canned


# --- callable script item ---


def test_callable_item_is_invoked_with_request_and_result_used():
    def handler(request: ModelRequest) -> str:
        return f"you said: {request.messages[0].text}"

    fake = FakeModel([handler])
    resp = fake.complete(ModelRequest(model="fake/model", messages=[Message.user("hola")]))
    assert resp.message.text == "you said: hola"


def test_callable_item_can_return_dict_form():
    def handler(request: ModelRequest) -> dict:
        return {"json": {"ok": True}}

    fake = FakeModel([handler])
    resp = fake.complete(_req())
    assert resp.parsed == {"ok": True}


# --- request recording ---


def test_requests_are_recorded_in_order():
    fake = FakeModel(["a", "b"])
    r1 = ModelRequest(model="fake/model", messages=[Message.user("1")])
    r2 = ModelRequest(model="fake/model", messages=[Message.user("2")])
    fake.complete(r1)
    fake.complete(r2)
    assert fake.requests == [r1, r2]


# --- default usage ---


def test_default_usage_is_10_input_20_output_cost_none_complete_false():
    """`cost_complete=False` (not the dataclass default `True`) matches every
    real adapter's convention: `cost_usd=None` paired with `cost_complete=True`
    is reserved for the zero-usage "nothing happened yet" sentinel (see
    `Usage`'s own docstring), not for a scripted call that consumed real,
    nonzero tokens at an unknown/unmodeled price."""
    fake = FakeModel(["hi"])
    resp = fake.complete(_req())
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 20
    assert resp.usage.cost_usd is None
    assert resp.usage.cost_complete is False


def test_custom_usage_overrides_default():
    custom = Usage(input_tokens=1, output_tokens=2, cost_usd=0.01, cost_complete=True)
    fake = FakeModel(["hi"], usage=custom)
    resp = fake.complete(_req())
    assert resp.usage == custom


# --- exhaustion ---


def test_exhausted_script_raises_compose_error_with_request_count():
    fake = FakeModel(["only one"])
    fake.complete(_req())
    with pytest.raises(ComposeError, match="2"):
        fake.complete(_req())


def test_exhausted_script_still_records_the_failing_request():
    fake = FakeModel([])
    with pytest.raises(ComposeError):
        fake.complete(_req())
    assert len(fake.requests) == 1


# --- protocol conformance ---


def test_fake_model_satisfies_model_protocol():
    from composeai.models.base import Model

    assert isinstance(FakeModel(["hi"]), Model)


# --- stream() ---


def test_stream_text_item_yields_word_deltas_then_response_done():
    fake = FakeModel(["hello there"])
    events = list(fake.stream(_req()))
    assert events[-1].kind == "response_done"
    assert events[-1].response is not None
    deltas = [e for e in events[:-1]]
    assert all(e.kind == "text_delta" for e in deltas)
    assert "".join(e.text or "" for e in deltas) == "hello there"
    # whitespace-preserving: multiple/irregular spacing round-trips exactly
    assert [e.text for e in deltas] == ["hello", " ", "there"]


def test_stream_text_deltas_preserve_irregular_whitespace():
    fake = FakeModel(["a  b\tc"])
    events = list(fake.stream(_req()))
    deltas = [e.text or "" for e in events if e.kind == "text_delta"]
    assert "".join(deltas) == "a  b\tc"


def test_stream_response_done_equals_complete_for_same_script_item():
    fake_stream = FakeModel(["hello there"])
    fake_complete = FakeModel(["hello there"])
    streamed = list(fake_stream.stream(_req()))
    completed = fake_complete.complete(_req())
    response_done = streamed[-1]
    assert response_done.response == completed


def test_stream_records_request():
    fake = FakeModel(["hi"])
    req = _req()
    list(fake.stream(req))
    assert fake.requests == [req]


def test_stream_tool_calls_item_yields_started_args_delta_finished():
    tool_call = {"name": "search", "arguments": {"q": "cats"}, "id": "tc1"}
    fake = FakeModel([{"tool_calls": [tool_call]}])
    events = list(fake.stream(_req()))
    kinds = [e.kind for e in events]
    assert kinds == ["tool_call_started", "tool_args_delta", "tool_call_finished", "response_done"]
    started, args_delta, finished, done = events
    assert started.tool_call_id == "tc1"
    assert started.tool_name == "search"
    assert args_delta.tool_call_id == "tc1"
    assert args_delta.tool_name == "search"
    assert args_delta.text == json.dumps({"q": "cats"})
    assert finished.tool_call_id == "tc1"
    assert finished.tool_name == "search"
    assert done.response is not None


def test_stream_tool_calls_item_with_text_yields_text_deltas_first():
    fake = FakeModel([{"tool_calls": [{"name": "a", "arguments": {}}], "text": "let me check"}])
    events = list(fake.stream(_req()))
    kinds = [e.kind for e in events]
    tool_kinds_start = kinds.index("tool_call_started")
    assert all(k == "text_delta" for k in kinds[:tool_kinds_start])
    assert "".join(e.text or "" for e in events[:tool_kinds_start]) == "let me check"
    assert kinds[tool_kinds_start:] == [
        "tool_call_started",
        "tool_args_delta",
        "tool_call_finished",
        "response_done",
    ]


def test_stream_json_item_yields_only_response_done_text_deltas():
    fake = FakeModel([{"json": {"answer": 42}}])
    events = list(fake.stream(_req()))
    # The message text is the JSON-encoded payload, streamed as word deltas.
    assert events[-1].kind == "response_done"
    assert events[-1].response is not None
    assert events[-1].response.parsed == {"answer": 42}


def test_stream_exhausted_script_raises_compose_error():
    fake = FakeModel([])
    with pytest.raises(ComposeError):
        list(fake.stream(_req()))


def test_fake_model_stream_returns_raw_stream_events():
    fake = FakeModel(["hi"])
    events = list(fake.stream(_req()))
    assert all(isinstance(e, RawStreamEvent) for e in events)


# --- thinking_delta streaming (capstone fix wave C) --------------------------


def test_stream_thinking_part_yields_thinking_deltas():
    """Regression: `_synthesize_stream_events` only handled `TextPart`/
    `ToolCallPart` -- a scripted (or replayed) response whose message
    included a `ThinkingPart` silently dropped it from the stream, even
    though real adapters emit `thinking_delta` events live and the final
    `response_done`'s `.response` always carried the full message anyway
    (giving false confidence that thinking-delta streaming worked when
    only ever tested offline)."""
    from composeai.messages import ThinkingPart

    response = ModelResponse(
        message=Message(
            role="assistant",
            parts=[ThinkingPart(text="hmm let me think"), TextPart(text="answer")],
        ),
        stop_reason=StopReason.END_TURN,
        raw_stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
        model_id="fake/model",
    )
    fake = FakeModel([response])
    events = list(fake.stream(_req()))
    kinds = [e.kind for e in events]
    assert "thinking_delta" in kinds

    thinking_text = "".join(
        e.text or "" for e in events if e.kind == "thinking_delta"
    )
    assert thinking_text == "hmm let me think"

    # Thinking deltas precede the text deltas (part order preserved).
    first_thinking = kinds.index("thinking_delta")
    first_text = kinds.index("text_delta")
    assert first_thinking < first_text


def test_stream_thinking_part_response_done_still_carries_full_message():
    from composeai.messages import ThinkingPart

    response = ModelResponse(
        message=Message(role="assistant", parts=[ThinkingPart(text="reasoning")]),
        stop_reason=StopReason.END_TURN,
        raw_stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
        model_id="fake/model",
    )
    fake = FakeModel([response])
    events = list(fake.stream(_req()))
    assert events[-1].kind == "response_done"
    assert events[-1].response == response


# --- async twins (acomplete / astream) ---


def test_fake_model_acomplete_consumes_script_in_order():
    import asyncio

    from composeai.models.base import ModelRequest

    model = FakeModel(["one", "two"])
    request = ModelRequest(
        model="fake", messages=[Message.user("hi")], system=None, tools=None,
        output_schema=None, max_tokens=10, temperature=None, provider=None,
    )

    async def drive():
        first = await model.acomplete(request)
        second = await model.acomplete(request)
        return first.message.text, second.message.text

    assert asyncio.run(drive()) == ("one", "two")


def test_fake_model_astream_yields_response_done_last():
    import asyncio

    from composeai.models.base import ModelRequest

    model = FakeModel(["hello world"])
    request = ModelRequest(
        model="fake", messages=[Message.user("hi")], system=None, tools=None,
        output_schema=None, max_tokens=10, temperature=None, provider=None,
    )

    async def drive():
        events = []
        async for event in model.astream(request):
            events.append(event)
        return events

    events = asyncio.run(drive())
    assert events[-1].kind == "response_done"
    assert events[-1].response is not None
