import io
import json
import threading
import time

import pytest

from composeai import events
from composeai.messages import Usage
from composeai.tracing import (
    ErrorInfo,
    Span,
    Trace,
    content_capture_enabled,
    current_span,
    current_trace,
    propagate,
    render_trace,
    span,
    truncate_payload,
)

# --- Span basics ---


def test_span_defaults():
    s = Span(trace_id="t1", kind="agent", name="a", started_at=time.time())
    assert isinstance(s.span_id, str) and len(s.span_id) == 26
    assert s.parent_span_id is None
    assert s.status == "running"
    assert s.ended_at is None
    assert s.error is None
    assert s.input is None
    assert s.output is None
    assert s.usage is None
    assert s.attributes == {}
    assert s.replayed is False


def test_span_duration_ms_none_while_running():
    s = Span(trace_id="t1", kind="agent", name="a", started_at=100.0)
    assert s.duration_ms is None


def test_span_duration_ms_computed_after_end():
    s = Span(trace_id="t1", kind="agent", name="a", started_at=100.0, ended_at=100.5)
    assert s.duration_ms == 500.0


def test_span_is_mutable():
    s = Span(trace_id="t1", kind="agent", name="a", started_at=time.time())
    s.status = "ok"
    s.output = "done"
    assert s.status == "ok"
    assert s.output == "done"


# --- Trace: add / roots / children_of ---


def test_trace_add_and_roots_and_children_stable_order():
    trace = Trace(trace_id="t1")
    root = Span(trace_id="t1", kind="flow", name="root", started_at=0.0)
    child1 = Span(
        trace_id="t1", parent_span_id=root.span_id, kind="task", name="c1", started_at=0.0
    )
    child2 = Span(
        trace_id="t1", parent_span_id=root.span_id, kind="task", name="c2", started_at=0.0
    )
    trace.add(root)
    trace.add(child1)
    trace.add(child2)

    assert trace.roots() == [root]
    assert trace.children_of(root.span_id) == [child1, child2]
    assert trace.children_of("nonexistent") == []


# --- Trace: rollup / total usage ---


def test_trace_rollup_usage_no_double_count():
    trace = Trace(trace_id="t1")
    agent = Span(trace_id="t1", kind="agent", name="agent", started_at=0.0)
    llm1 = Span(
        trace_id="t1",
        parent_span_id=agent.span_id,
        kind="llm",
        name="l1",
        started_at=0.0,
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.01),
    )
    llm2 = Span(
        trace_id="t1",
        parent_span_id=agent.span_id,
        kind="llm",
        name="l2",
        started_at=0.0,
        usage=Usage(input_tokens=20, output_tokens=8, cost_usd=0.02),
    )
    trace.add(agent)
    trace.add(llm1)
    trace.add(llm2)

    rollup = trace.rollup_usage(agent)
    assert rollup.input_tokens == 30
    assert rollup.output_tokens == 13
    assert rollup.cost_usd == pytest.approx(0.03)
    assert agent.usage is None  # never stored on parent

    total = trace.total_usage()
    assert total.input_tokens == 30
    assert total.output_tokens == 13


def test_trace_rollup_usage_of_llm_span_itself():
    trace = Trace(trace_id="t1")
    llm = Span(trace_id="t1", kind="llm", name="l", started_at=0.0, usage=Usage(input_tokens=5))
    trace.add(llm)
    rollup = trace.rollup_usage(llm)
    assert rollup.input_tokens == 5


# --- Trace: status precedence ---


def test_trace_status_ok_when_all_ok():
    trace = Trace(trace_id="t1")
    root = Span(trace_id="t1", kind="flow", name="r", started_at=0.0, ended_at=1.0, status="ok")
    trace.add(root)
    assert trace.status == "ok"


def test_trace_status_running_when_root_running():
    trace = Trace(trace_id="t1")
    root = Span(trace_id="t1", kind="flow", name="r", started_at=0.0)
    trace.add(root)
    assert trace.status == "running"


def test_trace_status_paused_when_any_span_paused_and_no_error():
    trace = Trace(trace_id="t1")
    root = Span(trace_id="t1", kind="flow", name="r", started_at=0.0, status="running")
    child = Span(
        trace_id="t1",
        parent_span_id=root.span_id,
        kind="pause",
        name="p",
        started_at=0.0,
        status="paused",
    )
    trace.add(root)
    trace.add(child)
    assert trace.status == "paused"


def test_trace_status_error_takes_precedence_over_paused():
    trace = Trace(trace_id="t1")
    root = Span(
        trace_id="t1", kind="flow", name="r", started_at=0.0, ended_at=1.0, status="error"
    )
    child = Span(
        trace_id="t1",
        parent_span_id=root.span_id,
        kind="pause",
        name="p",
        started_at=0.0,
        status="paused",
    )
    trace.add(root)
    trace.add(child)
    assert trace.status == "error"


# --- Truncation ---


def test_truncate_payload_short_string_untouched():
    value, truncated = truncate_payload("short")
    assert value == "short"
    assert truncated is False


def test_truncate_payload_long_string_head_tail():
    limit = 20
    long_str = "a" * 15 + "b" * 15  # 30 chars, over limit
    value, truncated = truncate_payload(long_str, limit=limit)
    assert truncated is True
    half = limit // 2
    head = long_str[:half]
    tail = long_str[-half:]
    dropped = len(long_str) - 2 * half
    assert value == f"{head}…[{dropped} chars truncated]…{tail}"


def test_truncate_payload_nested_structures():
    limit = 10
    long_str = "x" * 20
    data = {"a": [long_str, {"b": long_str}], "c": "short"}
    value, truncated = truncate_payload(data, limit=limit)
    assert truncated is True
    assert value["c"] == "short"
    assert "…[" in value["a"][0]
    assert "…[" in value["a"][1]["b"]


def test_truncate_payload_non_string_scalars_pass_through():
    data = {"n": 5, "f": 1.5, "b": True, "none": None}
    value, truncated = truncate_payload(data)
    assert value == data
    assert truncated is False


# --- Content capture ---


def test_content_capture_enabled_by_default(monkeypatch):
    monkeypatch.delenv("COMPOSE_TRACE_CONTENT", raising=False)
    assert content_capture_enabled() is True


def test_content_capture_disabled_when_env_is_0(monkeypatch):
    monkeypatch.setenv("COMPOSE_TRACE_CONTENT", "0")
    assert content_capture_enabled() is False


def test_content_capture_enabled_for_any_other_value(monkeypatch):
    monkeypatch.setenv("COMPOSE_TRACE_CONTENT", "false")
    assert content_capture_enabled() is True


# --- Collector: span() context manager ---


def test_span_creates_root_trace_when_none_active():
    assert current_trace() is None
    with span("flow", "root", input={"x": 1}) as s:
        assert current_span() is s
        trace = current_trace()
        assert trace is not None
        assert trace.trace_id == s.trace_id
        assert s.parent_span_id is None
        assert s.status == "running"
    assert s.status == "ok"
    assert s.ended_at is not None
    assert current_trace() is None
    assert current_span() is None


def test_span_nesting_sets_parent_and_sibling_order():
    with span("flow", "root") as root:
        trace = current_trace()
        with span("task", "child1") as c1:
            assert c1.parent_span_id == root.span_id
        with span("task", "child2") as c2:
            assert c2.parent_span_id == root.span_id
        assert current_span() is root

    assert trace is not None
    assert trace.children_of(root.span_id) == [c1, c2]


def test_span_restores_parent_context_after_child_exits():
    with span("flow", "root") as root:
        with span("task", "child") as child:
            assert current_span() is child
        assert current_span() is root


def test_span_exception_path_sets_error_status_and_reraises_same_exception():
    original = ValueError("boom")
    captured: dict[str, Span] = {}
    with pytest.raises(ValueError) as exc_info:
        with span("tool", "failing") as s:
            captured["span"] = s
            raise original
    assert exc_info.value is original
    s = captured["span"]
    assert s.status == "error"
    assert s.error is not None
    assert s.error.type == "ValueError"
    assert s.error.message == "boom"
    assert "ValueError: boom" in s.error.stacktrace
    assert s.ended_at is not None


def test_span_does_not_alter_preexisting_exception_context():
    ctx_exc = RuntimeError("previous")
    original = ValueError("boom")
    original.__context__ = ctx_exc

    with pytest.raises(ValueError) as exc_info:
        with span("tool", "failing"):
            raise original

    assert exc_info.value is original
    assert original.__context__ is ctx_exc


def test_span_contextvars_restored_after_exception():
    assert current_span() is None
    assert current_trace() is None
    with pytest.raises(ValueError):
        with span("tool", "failing"):
            raise ValueError("boom")
    assert current_span() is None
    assert current_trace() is None


def test_span_emits_span_started_and_span_finished_events():
    bus = events.EventBus()
    sub = bus.subscribe()
    with events.use_bus(bus):
        with span("agent", "do-thing") as s:
            pass
    sub.close()
    received = list(sub)
    kinds = [e.kind for e in received]
    assert kinds == ["span_started", "span_finished"]
    started, finished = received
    assert started.trace_id == s.trace_id
    assert started.span_id == s.span_id
    assert started.name == "do-thing"
    assert started.data == {"kind": "agent"}
    assert finished.span_id == s.span_id
    assert finished.data is not None
    assert finished.data["status"] == "ok"


def test_span_emits_span_finished_with_error_status_on_exception():
    bus = events.EventBus()
    sub = bus.subscribe()
    with events.use_bus(bus):
        with pytest.raises(ValueError):
            with span("tool", "boom"):
                raise ValueError("x")
    sub.close()
    received = list(sub)
    finished = received[-1]
    assert finished.kind == "span_finished"
    assert finished.data is not None
    assert finished.data["status"] == "error"


# --- propagate() ---


def test_propagate_carries_current_span_into_thread():
    result = {}
    with span("flow", "root") as root:

        def worker():
            result["span"] = current_span()
            result["trace"] = current_trace()

        thread = threading.Thread(target=propagate(worker))
        thread.start()
        thread.join()

    assert result["span"] is root
    assert result["trace"] is not None
    assert result["trace"].trace_id == root.trace_id


def test_without_propagate_thread_does_not_see_current_span():
    result = {}
    with span("flow", "root"):

        def worker():
            result["span"] = current_span()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert result["span"] is None


# --- Content capture off keeps usage but drops payloads ---


def test_span_input_dropped_when_content_capture_disabled(monkeypatch):
    monkeypatch.setenv("COMPOSE_TRACE_CONTENT", "0")
    with span("llm", "call", input={"prompt": "secret"}) as s:
        s.usage = Usage(input_tokens=5, output_tokens=2, cost_usd=0.001)
    assert s.input is None
    assert s.usage is not None
    assert s.usage.input_tokens == 5
    assert s.status == "ok"
    assert s.ended_at is not None


def test_span_input_captured_when_content_capture_enabled(monkeypatch):
    monkeypatch.delenv("COMPOSE_TRACE_CONTENT", raising=False)
    with span("llm", "call", input={"prompt": "hello"}) as s:
        pass
    assert s.input == {"prompt": "hello"}


# --- to_dict / to_json ---


def test_to_dict_has_otel_attributes_and_top_level_cost():
    trace = Trace(trace_id="t1")
    llm = Span(
        trace_id="t1",
        kind="llm",
        name="call",
        started_at=0.0,
        ended_at=1.0,
        status="ok",
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=2,
            cache_creation_tokens=1,
            cost_usd=0.05,
        ),
        attributes={"provider": "anthropic", "model": "claude-sonnet-5"},
        input="hi",
        output="there",
    )
    trace.add(llm)
    d = trace.to_dict()
    assert d["trace_id"] == "t1"
    span_dict = d["spans"][0]
    assert span_dict["span_id"] == llm.span_id
    assert span_dict["trace_id"] == "t1"
    assert span_dict["kind"] == "llm"
    assert span_dict["cost_usd"] == 0.05
    attrs = span_dict["attributes"]
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 2
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 1
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-sonnet-5"
    assert attrs["provider"] == "anthropic"  # original attribute retained
    assert "_truncated" not in span_dict


def test_to_dict_error_as_dict():
    trace = Trace(trace_id="t1")
    s = Span(
        trace_id="t1",
        kind="tool",
        name="fail",
        started_at=0.0,
        ended_at=1.0,
        status="error",
        error=ErrorInfo(type="ValueError", message="boom", stacktrace="Traceback..."),
    )
    trace.add(s)
    d = trace.to_dict()
    assert d["spans"][0]["error"] == {
        "type": "ValueError",
        "message": "boom",
        "stacktrace": "Traceback...",
    }


def test_to_dict_no_usage_has_none_cost_and_no_otel_keys():
    trace = Trace(trace_id="t1")
    s = Span(trace_id="t1", kind="task", name="t", started_at=0.0, ended_at=1.0, status="ok")
    trace.add(s)
    d = trace.to_dict()
    span_dict = d["spans"][0]
    assert span_dict["cost_usd"] is None
    assert "gen_ai.usage.input_tokens" not in span_dict["attributes"]


def test_to_dict_truncates_and_flags():
    trace = Trace(trace_id="t1")
    long_text = "y" * 9000
    s = Span(trace_id="t1", kind="llm", name="call", started_at=0.0, output=long_text)
    trace.add(s)
    d = trace.to_dict()
    span_dict = d["spans"][0]
    assert span_dict["_truncated"] is True
    assert "…[" in span_dict["output"]


def test_to_json_round_trips_via_json_loads():
    trace = Trace(trace_id="t1")
    s = Span(trace_id="t1", kind="task", name="t", started_at=0.0)
    trace.add(s)
    text = trace.to_json()
    parsed = json.loads(text)
    assert parsed["trace_id"] == "t1"

    indented = trace.to_json(indent=2)
    assert "\n" in indented


# --- Trace.print ---


def test_trace_print_writes_render_trace_output_plus_newline_to_given_file():
    trace = Trace(trace_id="t1")
    s = Span(trace_id="t1", kind="task", name="t", started_at=0.0, ended_at=1.0, status="ok")
    trace.add(s)
    buf = io.StringIO()

    trace.print(file=buf, color=False)

    assert buf.getvalue() == render_trace(trace, color=False) + "\n"


def test_trace_print_defaults_to_stdout(capsys):
    trace = Trace(trace_id="t1")
    s = Span(trace_id="t1", kind="task", name="t", started_at=0.0, ended_at=1.0, status="ok")
    trace.add(s)

    trace.print(color=False)

    captured = capsys.readouterr()
    assert captured.out == render_trace(trace, color=False) + "\n"


# --- Reviewer additions: gated setters + safe trace export ---


def test_set_input_and_set_output_respect_content_capture(monkeypatch):
    monkeypatch.setenv("COMPOSE_TRACE_CONTENT", "0")
    with span("task", "t") as s:
        s.set_input("secret in")
        s.set_output({"secret": "out"})
    assert s.input is None
    assert s.output is None


def test_set_input_and_set_output_capture_when_enabled(monkeypatch):
    monkeypatch.delenv("COMPOSE_TRACE_CONTENT", raising=False)
    with span("task", "t") as s:
        s.set_input("in")
        s.set_output({"a": 1})
    assert s.input == "in"
    assert s.output == {"a": 1}


def test_to_dict_falls_back_to_repr_for_unserializable_payload():
    class Weird:
        def __repr__(self) -> str:
            return "<weird-object>"

    with span("task", "t") as s:
        trace = current_trace()
        assert trace is not None
        s.set_output(Weird())
    exported = trace.to_dict()
    assert exported["spans"][0]["output"] == "<weird-object>"
    # Export must not raise, and valid payloads elsewhere stay structured.
    json.loads(trace.to_json())
