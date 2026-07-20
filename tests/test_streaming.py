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
from composeai.runs import RunStream
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


def test_run_stream_has_cancel_that_is_idempotent_and_stores_the_event():
    model = FakeModel(["short"])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    assert hasattr(run_stream, "cancel") and callable(run_stream.cancel)
    # The Event is created at `.stream()` time and stored on the RunStream.
    assert isinstance(run_stream._cancel, threading.Event)
    assert not run_stream._cancel.is_set()
    run_stream.cancel()
    run_stream.cancel()  # idempotent -- must not raise
    assert run_stream._cancel.is_set()
    # v0.9.0: the checks are now live, but this single fast turn races the
    # cancel -- it either finished before the cancel landed (completed) or the
    # turn-top guard tripped first (cancelled). Either way `.run` never raises.
    assert run_stream.run.status in ("completed", "cancelled")


def test_cancel_plumbing_is_wired_to_checks():
    import inspect

    from composeai import agentfn

    # The Event reaches the innermost model-invocation function.
    assert "cancel" in inspect.signature(agentfn._ainvoke_model).parameters
    assert "cancel" in inspect.signature(agentfn._arun_agent_uncached).parameters
    assert "cancel" in inspect.signature(agentfn._aprocess_tool_use).parameters

    model = FakeModel(["short"])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    # v0.9.0: the checks are now live -- a cancel set before the first turn
    # boundary trips the turn-top guard on turn 1, so no model request is ever
    # made and the run ends cancelled (no output).
    run_stream.cancel()
    assert run_stream.run.status == "cancelled"
    assert run_stream.run.output is None
    assert len(model.requests) == 0


def test_cancel_between_turns_returns_cancelled_run_and_completes_in_flight_tool():
    entered = threading.Event()
    release = threading.Event()

    @tool
    def gate() -> str:
        """Block until released (proves an in-flight tool runs to completion)."""
        entered.set()
        release.wait(timeout=5)
        return "gate-done"

    # turn 1 -> call gate; turn 2 would call `gate` again (must never happen).
    model = FakeModel(
        [
            {"tool_calls": [{"name": "gate", "arguments": {}, "id": "g1"}]},
            {"tool_calls": [{"name": "gate", "arguments": {}, "id": "g2"}]},
            "done",
        ]
    )

    @agent(model=model, tools=[gate], max_turns=10)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    entered.wait(timeout=5)          # turn 1's gate tool is executing
    run_stream.cancel()             # cancel while a tool is in flight
    release.set()                   # let the in-flight tool finish (cooperative)

    events = list(run_stream)       # drain to completion; must NOT raise
    run = run_stream.run            # must NOT raise
    assert run.status == "cancelled"
    assert run.output is None
    finished = [e for e in events if e.kind == "run_finished"]
    assert len(finished) == 1
    assert finished[0].data == {"status": "cancelled"}
    # No NEW turn started: only turn 1's model request was ever made.
    assert len(model.requests) == 1


def test_cancel_before_tool_batch_prevents_the_tool_from_running():
    ready = threading.Event()
    holder: dict[str, RunStream[str]] = {}
    ran = []

    @tool
    def marker() -> str:
        """Records that it ran (it must not, once cancel is set)."""
        ran.append(1)
        return "ok"

    def turn1(request):
        ready.wait(timeout=5)           # wait until the test has the RunStream
        holder["rs"].cancel()          # set cancel DURING the model turn
        return {"tool_calls": [{"name": "marker", "arguments": {}, "id": "m1"}]}

    model = FakeModel([turn1, "should-not-happen"])

    @agent(model=model, tools=[marker], max_turns=10)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    holder["rs"] = run_stream
    ready.set()

    run = run_stream.run
    assert run.status == "cancelled"
    assert ran == []                   # the pre-tool guard stopped the batch
    assert len(model.requests) == 1    # no second turn


def test_cancelled_run_does_not_raise_from_iteration_or_run():
    ready = threading.Event()
    holder: dict[str, RunStream[str]] = {}

    def turn1(request):
        ready.wait(timeout=5)
        holder["rs"].cancel()
        return {"tool_calls": [{"name": "noop", "arguments": {}, "id": "n1"}]}

    model = FakeModel([turn1, "never"])

    @agent(model=model, tools=[noop], max_turns=10)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    holder["rs"] = run_stream
    ready.set()

    list(run_stream)                   # no exception on iteration (unlike a failed run)
    assert run_stream.run.status == "cancelled"


def test_cancel_aborts_in_flight_stream_and_closes_the_adapter_iterator():
    import asyncio

    from composeai.messages import Message, StopReason, Usage
    from composeai.models.base import ModelRequest, ModelResponse, RawStreamEvent

    class BlockingStreamModel:
        """Async-streaming fake that parks mid-stream and records aclose()."""

        def __init__(self) -> None:
            self.first_yielded = threading.Event()
            self.release = threading.Event()
            self.closed = False
            self._resp = ModelResponse(
                message=Message.assistant("one two"),
                stop_reason=StopReason.END_TURN,
                raw_stop_reason="end_turn",
                usage=Usage(input_tokens=1, output_tokens=2),
                model_id="blocking",
            )

        def complete(self, request: ModelRequest) -> ModelResponse:  # Model protocol
            return self._resp

        async def astream(self, request: ModelRequest):
            self.closed = False
            try:
                yield RawStreamEvent(kind="text_delta", text="one")
                self.first_yielded.set()
                # Park without blocking the loop until the test releases us.
                while not self.release.is_set():
                    await asyncio.sleep(0.005)
                yield RawStreamEvent(kind="text_delta", text="two")
                yield RawStreamEvent(kind="response_done", response=self._resp)
            finally:
                # Reached only if the generator is *finalized* (aclose throws
                # GeneratorExit here) -- a bare break would NOT run this now.
                self.closed = True

    model = BlockingStreamModel()

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream()
    it = iter(run_stream)
    # consume up to the first token delta emitted from event "one"
    for ev in it:
        if ev.kind == "text_delta":
            break
    model.first_yielded.wait(timeout=5)   # generator is parked mid-stream
    run_stream.cancel()                   # set cancel while the stream is in flight
    model.release.set()                   # let it yield "two"; loop's check then trips

    run = run_stream.run
    assert run.status == "cancelled"
    # The adapter generator was DETERMINISTICALLY finalized (aclose ran), not
    # left for GC -- this is the crux the explicit close() guarantees.
    assert model.closed is True


def test_cancel_under_enclosing_span_returns_cancelled_run_without_raising():
    # Regression guard (v0.9.0): a cancel that lands while an enclosing
    # `tracing.span` is active drives the run down the DIRECT
    # `_run_agent_uncached` branch (tracing.current_span() is not None), not the
    # trace-root standalone path -- so it bypasses `settle_agent_run`'s
    # `_Cancelled`->cancelled-Run conversion. The centralized conversion in
    # `_stream_run`'s worker must still turn it into a returned
    # `Run(status="cancelled")`, so neither iteration nor `.run` re-raise the
    # `_Cancelled` BaseException (the leak this test guards against).
    ready = threading.Event()
    holder: dict[str, RunStream[str]] = {}

    def turn1(request):
        ready.wait(timeout=5)
        holder["rs"].cancel()
        return {"tool_calls": [{"name": "noop", "arguments": {}, "id": "n1"}]}

    model = FakeModel([turn1, "never"])

    @agent(model=model, tools=[noop], max_turns=10)
    def runner() -> str:
        """Runner."""
        return "go"

    with tracing.span("flow", "outer"):
        run_stream = runner.stream()
        holder["rs"] = run_stream
        ready.set()
        events = list(run_stream)      # must NOT raise (pre-fix: leaked _Cancelled)
        run = run_stream.run           # must NOT raise

    assert run.status == "cancelled"
    assert run.output is None
    # Iteration actually streamed and drained cleanly. There is no
    # `run_finished` here: this is a nested run (the agent span parents to
    # "outer"), and `tracing.emit_run_finished` fires only for the root run.
    assert events
    assert not [e for e in events if e.kind == "run_finished"]


def test_cancel_inside_a_flow_returns_cancelled_run_without_raising():
    # Regression guard (v0.9.0): streaming a nested @agent inside an active
    # @flow body drives the run down `_run_agent_journaled` (runs.current_run_
    # context() is not None), which ALSO bypasses `settle_agent_run`. The same
    # centralized worker conversion must apply, so draining the stream and
    # reading `.run` from inside the flow body never re-raise `_Cancelled`.
    from composeai.events import Event
    from composeai.flow import flow

    ready = threading.Event()
    holder: dict[str, RunStream[str]] = {}
    captured_events: list[Event] = []

    def turn1(request):
        ready.wait(timeout=5)
        holder["rs"].cancel()
        return {"tool_calls": [{"name": "noop", "arguments": {}, "id": "n1"}]}

    model = FakeModel([turn1, "never"])

    @agent(model=model, tools=[noop], max_turns=10)
    def nested() -> str:
        """Nested streaming agent."""
        return "go"

    @flow
    def streaming_flow() -> str:
        run_stream = nested.stream()
        holder["rs"] = run_stream
        ready.set()
        captured_events.extend(run_stream)   # must NOT raise (pre-fix: leaked)
        assert run_stream.run.status == "cancelled"  # must NOT raise, in-flow
        return "flow-done"

    streaming_flow.run()

    cancelled = holder["rs"].run
    assert cancelled.status == "cancelled"
    assert cancelled.output is None
    # The stream drained and `.run` returned from inside the flow body without
    # re-raising `_Cancelled` (the leak this guards against). No `run_finished`
    # here either -- the nested agent's span parents to the flow span, so it is
    # not the root run.
    assert captured_events
    assert not [e for e in captured_events if e.kind == "run_finished"]
