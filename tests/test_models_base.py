import pytest

from composeai.messages import Message, StopReason, Usage
from composeai.models.base import Model, ModelRequest, ModelResponse, RawStreamEvent, ToolSpec

# --- ToolSpec ---


def test_tool_spec_requires_name_description_schema():
    spec = ToolSpec(name="search", description="search the web", input_schema={"type": "object"})
    assert spec.name == "search"
    assert spec.description == "search the web"
    assert spec.input_schema == {"type": "object"}


def test_tool_spec_strict_defaults_true():
    spec = ToolSpec(name="f", description="d", input_schema={})
    assert spec.strict is True
    assert spec.requires_approval is False


def test_tool_spec_can_override_defaults():
    spec = ToolSpec(
        name="f", description="d", input_schema={}, strict=False, requires_approval=True
    )
    assert spec.strict is False
    assert spec.requires_approval is True


# --- ModelRequest ---


def test_model_request_minimal():
    req = ModelRequest(model="claude-sonnet-5", messages=[Message.user("hi")])
    assert req.model == "claude-sonnet-5"
    assert req.system is None
    assert req.tools is None
    assert req.output_schema is None
    assert req.max_tokens == 16000
    assert req.temperature is None


def test_model_request_all_fields():
    tool = ToolSpec(name="f", description="d", input_schema={})
    req = ModelRequest(
        model="claude-sonnet-5",
        messages=[Message.user("hi")],
        system="be nice",
        tools=[tool],
        output_schema={"type": "object"},
        max_tokens=100,
        temperature=0.5,
    )
    assert req.system == "be nice"
    assert req.tools == [tool]
    assert req.output_schema == {"type": "object"}
    assert req.max_tokens == 100
    assert req.temperature == 0.5


# --- ModelResponse ---


def test_model_response_minimal():
    msg = Message.assistant("hi there")
    resp = ModelResponse(
        message=msg,
        stop_reason=StopReason.END_TURN,
        raw_stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=2),
        model_id="claude-sonnet-5",
    )
    assert resp.message is msg
    assert resp.stop_reason == StopReason.END_TURN
    assert resp.raw_stop_reason == "end_turn"
    assert resp.usage.input_tokens == 1
    assert resp.model_id == "claude-sonnet-5"
    assert resp.parsed is None


def test_model_response_parsed_defaults_none_and_can_be_set():
    msg = Message.assistant('{"a": 1}')
    resp = ModelResponse(
        message=msg,
        stop_reason=StopReason.END_TURN,
        raw_stop_reason="end_turn",
        usage=Usage(),
        model_id="claude-sonnet-5",
        parsed={"a": 1},
    )
    assert resp.parsed == {"a": 1}


# --- Model protocol ---


def test_model_protocol_is_runtime_checkable():
    class Impl:
        def complete(self, request: ModelRequest) -> ModelResponse:
            raise NotImplementedError

    assert isinstance(Impl(), Model)


def test_model_protocol_rejects_objects_without_complete():
    class NotAModel:
        pass

    assert not isinstance(NotAModel(), Model)


def test_model_protocol_stays_complete_only_so_stream_less_models_still_satisfy_it():
    # Phase 5 adds streaming, but *not* as a required Model protocol member:
    # stream() is an optional extension (checked via getattr/hasattr at call
    # sites, not isinstance), so a duck-typed Model implementing only
    # complete() -- e.g. what registry.resolve()'s instance-passthrough path
    # must accept -- keeps satisfying isinstance(m, Model).
    assert not hasattr(Model, "stream")

    class CompleteOnly:
        def complete(self, request: ModelRequest) -> ModelResponse:
            raise NotImplementedError

    assert isinstance(CompleteOnly(), Model)


# --- RawStreamEvent ---


def test_raw_stream_event_defaults():
    event = RawStreamEvent(kind="text_delta")
    assert event.kind == "text_delta"
    assert event.text is None
    assert event.tool_call_id is None
    assert event.tool_name is None
    assert event.response is None


def test_raw_stream_event_is_frozen():
    event = RawStreamEvent(kind="text_delta")
    with pytest.raises(AttributeError):
        event.text = "x"  # type: ignore[misc]


def test_raw_stream_event_carries_tool_fields():
    event = RawStreamEvent(kind="tool_call_started", tool_call_id="tc1", tool_name="search")
    assert event.tool_call_id == "tc1"
    assert event.tool_name == "search"


def test_raw_stream_event_response_done_carries_full_response():
    resp = ModelResponse(
        message=Message.assistant("hi"),
        stop_reason=StopReason.END_TURN,
        raw_stop_reason="end_turn",
        usage=Usage(),
        model_id="m",
    )
    event = RawStreamEvent(kind="response_done", response=resp)
    assert event.response is resp
