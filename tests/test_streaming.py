"""Tests for ``@agent``'s streaming path -- ``AgentFunction.stream()`` / ``RunStream``.

Design principle under test: streaming and tracing are the same event bus.
``.stream()`` subscribes to the exact bus that tracing already publishes
span events to; adapters (here, ``FakeModel``) additionally publish token
deltas. Everything here runs offline against ``FakeModel``.

Deliberately *not* using ``from __future__ import annotations`` (same reason
as ``test_agent.py``): a test below defines a small pydantic model local to
the test function and references it in a decorated function's return
annotation, which only resolves to the real class -- rather than an inert
forward-reference string -- when annotations are evaluated eagerly.
"""

import threading

import pytest
from pydantic import BaseModel

from composeai import tracing
from composeai.agentfn import agent
from composeai.errors import ModelRefusalError
from composeai.messages import Message, StopReason, Usage
from composeai.models.base import ModelRequest, ModelResponse
from composeai.testing import FakeModel
from composeai.tools import tool


@tool
def noop() -> str:
    """Do nothing, just acknowledge."""
    return "ok"


def _refusal_response() -> ModelResponse:
    return ModelResponse(
        message=Message.assistant("I can't help with that."),
        stop_reason=StopReason.REFUSAL,
        raw_stop_reason="refusal",
        usage=Usage(input_tokens=5, output_tokens=5),
        model_id="fake",
    )


# --- event order -----------------------------------------------------------


def test_stream_event_order_on_tool_using_run():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}, "id": "tc1"}]},
            "All done.",
        ]
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    events = list(run_stream)
    kinds = [e.kind for e in events]

    assert kinds == [
        "span_started",  # agent
        "span_started",  # llm turn 1
        "tool_call_started",
        "tool_args_delta",
        "tool_call_finished",
        "span_finished",  # llm turn 1
        "span_started",  # tool
        "span_finished",  # tool
        "span_started",  # llm turn 2
        "text_delta",
        "text_delta",
        "text_delta",
        "span_finished",  # llm turn 2
        "span_finished",  # agent
        "run_finished",
    ]
    span_kinds = [
        e.data["kind"]
        for e in events
        if e.kind in ("span_started", "span_finished") and e.data is not None
    ]
    assert span_kinds == ["agent", "llm", "llm", "tool", "tool", "llm", "llm", "agent"]

    llm_span_started = events[1]
    tool_call_started = events[2]
    assert tool_call_started.span_id == llm_span_started.span_id
    assert tool_call_started.trace_id == llm_span_started.trace_id

    assert run_stream.run.output == "All done."


def test_stream_tool_events_carry_tool_call_id_in_data_and_name():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}, "id": "tc-xyz"}]},
            "done",
        ]
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    events = list(runner.stream())
    started = next(e for e in events if e.kind == "tool_call_started")
    assert started.data == {"id": "tc-xyz"}
    assert started.name == "noop"


def test_stream_text_delta_name_falls_back_to_slot_label():
    model = FakeModel(["hello"])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    events = list(runner.stream())
    text_events = [e for e in events if e.kind == "text_delta"]
    assert text_events  # sanity
    assert all(e.name == "FakeModel" for e in text_events)
    assert all(e.data is None for e in text_events)


# --- .run parity -------------------------------------------------------------


def test_stream_run_output_matches_plain_run_output():
    @agent(model=FakeModel(["same output"]), max_turns=3)
    def streaming_runner() -> str:
        """Runner."""
        return "go"

    @agent(model=FakeModel(["same output"]), max_turns=3)
    def plain_runner() -> str:
        """Runner."""
        return "go"

    run_stream = streaming_runner.stream()
    list(run_stream)  # drain
    streamed_result = run_stream.run
    plain_result = plain_runner.run()

    assert streamed_result.output == plain_result.output == "same output"
    assert streamed_result.status == plain_result.status == "completed"


def test_stream_run_property_works_without_manual_draining():
    model = FakeModel(["hi there"])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    # Never iterate -- .run must still join the worker thread and succeed.
    assert run_stream.run.output == "hi there"


# --- structured output -------------------------------------------------------


def test_stream_structured_output():
    class Out(BaseModel):
        x: int

    model = FakeModel([{"json": {"x": 5}}])

    @agent(model=model, max_turns=3)
    def runner() -> Out:
        """Runner."""
        return "go"  # pyright: ignore[reportReturnType]

    run_stream = runner.stream()
    list(run_stream)
    assert run_stream.run.output == Out(x=5)


# --- early break / close ------------------------------------------------------


def test_stream_early_break_then_close_then_run_still_returns():
    model = FakeModel(["hello there, a longer response with several word deltas in it"])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    it = iter(run_stream)
    first = next(it)
    assert first.kind == "span_started"

    run_stream.close()  # unsubscribe early; must not deadlock or disturb the worker
    result = run_stream.run
    assert result.output.startswith("hello there")


def test_stream_close_is_idempotent_and_thread_is_joinable():
    model = FakeModel(["short"])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    run_stream.close()
    run_stream.close()  # must not raise
    assert run_stream.run.output == "short"


def test_stream_worker_runs_on_a_background_thread():
    seen_thread_ids: list[int] = []

    def handler(request: ModelRequest) -> str:
        seen_thread_ids.append(threading.get_ident())
        return "ok"

    model = FakeModel([handler])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    list(run_stream)
    assert len(seen_thread_ids) == 1
    assert seen_thread_ids[0] != threading.get_ident()
    assert run_stream.run.output == "ok"


# --- exceptions ----------------------------------------------------------------


def test_stream_exception_reraised_from_full_iteration():
    model = FakeModel([_refusal_response()])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    with pytest.raises(ModelRefusalError):
        list(run_stream)


def test_stream_exception_reraised_from_run_property():
    model = FakeModel([_refusal_response()])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    with pytest.raises(ModelRefusalError):
        _ = run_stream.run


def test_stream_same_exception_object_from_iteration_and_run():
    model = FakeModel([_refusal_response()])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    caught: BaseException | None = None
    try:
        list(run_stream)
    except ModelRefusalError as exc:
        caught = exc
    assert caught is not None

    with pytest.raises(ModelRefusalError) as exc_info:
        _ = run_stream.run
    assert exc_info.value is caught


def test_run_finished_event_carries_failed_status_and_error_type():
    model = FakeModel([_refusal_response()])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    collected = []
    with pytest.raises(ModelRefusalError):
        for event in run_stream:
            collected.append(event)

    run_finished = collected[-1]
    assert run_finished.kind == "run_finished"
    assert run_finished.data == {"status": "failed", "error": "ModelRefusalError"}


def test_run_finished_event_carries_completed_status_on_success():
    model = FakeModel(["all good"])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    events = list(runner.stream())
    run_finished = events[-1]
    assert run_finished.kind == "run_finished"
    assert run_finished.data == {"status": "completed"}


# --- degrade path for a stream-less Model -------------------------------------


class _CompleteOnlyModel:
    """A minimal duck-typed Model implementing only complete() -- no stream()."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            message=Message.assistant(self._text),
            stop_reason=StopReason.END_TURN,
            raw_stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
            model_id=request.model,
        )


def test_stream_degrades_to_complete_for_stream_less_model():
    model = _CompleteOnlyModel("plain response")

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    kinds = [e.kind for e in run_stream]

    assert "text_delta" not in kinds
    assert "tool_call_started" not in kinds
    assert kinds[0] == "span_started"
    assert kinds[-1] == "run_finished"
    assert run_stream.run.output == "plain response"
    assert len(model.requests) == 1


# --- context propagation -------------------------------------------------------


def test_stream_propagates_enclosing_span_context():
    model = FakeModel(["done"])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    with tracing.span("flow", "outer") as outer:
        run_stream = runner.stream()
        list(run_stream)
        result = run_stream.run

    agent_span = next(s for s in result.trace.spans if s.kind == "agent")
    assert agent_span.parent_span_id == outer.span_id
    assert agent_span.trace_id == outer.trace_id
