"""Tests for ``composeai.combinators`` -- ``pipe``/``aggregate``/``map`` and
composition-time type checking (Phase 6).

Design principle under test: ``pipe()`` type-checks every consecutive stage
pair *at build time*, before any stage ever runs -- a wiring bug is a
``CompositionTypeError`` raised by ``pipe(...)`` itself, never a failure
partway through a (possibly expensive) run. Everything here runs offline
against ``FakeModel``.

Deliberately *not* using ``from __future__ import annotations`` (same
reason as ``test_agent.py``): several tests define a small pydantic model
local to the test function and reference it in a decorated function's
annotations, which only resolves to the real class -- rather than an inert
forward-reference string -- when annotations are evaluated eagerly.
"""

import time
from collections import defaultdict
from typing import Any

import pytest
from pydantic import BaseModel

import composeai as compose
from composeai import tracing
from composeai.agentfn import agent
from composeai.combinators import _types_compatible as types_compatible
from composeai.combinators import aggregate, pipe
from composeai.errors import CompositionTypeError
from composeai.testing import FakeModel


class FactSheet(BaseModel):
    facts: list[str]


class Summary(BaseModel):
    text: str


# --- _types_compatible matrix --------------------------------------------------


def test_types_compatible_equal_types():
    assert types_compatible(str, str) is True


def test_types_compatible_subclass_passes_reverse_fails():
    class Base:
        pass

    class Child(Base):
        pass

    assert types_compatible(Child, Base) is True
    assert types_compatible(Base, Child) is False


def test_types_compatible_any_either_side():
    assert types_compatible(Any, str) is True
    assert types_compatible(str, Any) is True
    assert types_compatible(Any, Any) is True


def test_types_compatible_object_either_side():
    assert types_compatible(object, str) is True
    assert types_compatible(str, object) is True


def test_types_compatible_generics_equal_pass():
    assert types_compatible(list[str], list[str]) is True


def test_types_compatible_generics_mismatch_fails():
    assert types_compatible(list[str], list[int]) is False


def test_types_compatible_unrelated_pydantic_models_fail():
    assert types_compatible(FactSheet, Summary) is False


def test_types_compatible_same_pydantic_model_passes():
    assert types_compatible(FactSheet, FactSheet) is True


# --- _types_compatible: Union/Optional (capstone fix wave A) -------------------


def test_types_compatible_concrete_output_feeds_wider_optional_input():
    """str -> str | None is safe: the downstream stage tolerates a
    narrower, concrete input than what it declares it can accept."""
    assert types_compatible(str, str | None) is True
    assert types_compatible(str, str | int) is True


def test_types_compatible_output_not_a_member_of_input_union_fails():
    assert types_compatible(bytes, str | int) is False


def test_types_compatible_union_output_passes_only_if_every_member_is_accepted():
    """A stage that might return str | int can only safely feed a stage
    whose input type accepts BOTH members -- not just one of them."""
    assert types_compatible(str | int, str | int) is True
    assert types_compatible(str | int, str) is False  # int member not accepted


def test_types_compatible_union_output_into_union_input_partial_overlap_fails():
    assert types_compatible(str | bytes, str | int) is False  # bytes not accepted


def test_pipe_accepts_str_output_feeding_optional_str_input():
    def a(x: str) -> str:
        return x

    def b(x: str | None) -> str:
        return x or "default"

    p = pipe(a, b)
    assert p("hello") == "hello"


# --- pipe(): stage count + type checking at build time -------------------------


def test_pipe_rejects_zero_or_one_stage():
    def f(x: str) -> str:
        return x

    with pytest.raises(CompositionTypeError):
        pipe()
    with pytest.raises(CompositionTypeError):
        pipe(f)


def test_pipe_type_mismatch_raises_before_any_api_spend():
    calls: list[str] = []

    @agent(model=FakeModel(["never used"]))
    def researcher(topic: str) -> FactSheet:
        """Research."""
        calls.append("researcher")
        return topic  # type: ignore[return-value]

    @agent(model=FakeModel(["never used"]))
    def copywriter(summary: Summary) -> str:
        """Write."""
        calls.append("copywriter")
        return "go"

    with pytest.raises(CompositionTypeError) as exc_info:
        pipe(researcher, copywriter)

    message = str(exc_info.value)
    assert "researcher" in message
    assert "copywriter" in message
    assert "FactSheet" in message
    assert "Summary" in message
    assert calls == []  # failed at composition time -- no API spend


def test_pipe_names_correct_stage_pair_in_a_three_stage_chain():
    @agent(model=FakeModel(["x"]))
    def a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["x"]))
    def b(x: str) -> FactSheet:
        """B."""
        return x  # type: ignore[return-value]

    @agent(model=FakeModel(["x"]))
    def c(summary: Summary) -> str:
        """C."""
        return "go"

    with pytest.raises(CompositionTypeError) as exc_info:
        pipe(a, b, c)

    message = str(exc_info.value)
    assert "stage 2" in message
    assert "stage 3" in message
    assert "b" in message
    assert "c" in message


def test_pipe_pydantic_chain_type_match_succeeds():
    @agent(model=FakeModel([{"json": {"facts": ["a"]}}]))
    def researcher(topic: str) -> FactSheet:
        """Research."""
        return topic  # type: ignore[return-value]

    @agent(model=FakeModel(["a summary"]))
    def copywriter(sheet: FactSheet) -> str:
        """Write."""
        return str(sheet)

    pipeline = pipe(researcher, copywriter)
    assert pipeline("quantum computing") == "a summary"


def test_pipe_missing_annotations_default_to_any_and_pass():
    def stage_one(x):
        return x

    def stage_two(x):
        return x

    pipeline = pipe(stage_one, stage_two)
    assert pipeline("hi") == "hi"


def test_pipe_with_tool_stage_gets_real_composition_type_checking():
    """Regression: a @tool used directly as a pipe()/aggregate() stage used
    to silently lose composition-time type checking -- _plain_callable_types
    introspected Tool.__call__'s untyped (*args, **kwargs) signature instead
    of the wrapped function's real one, so input_type/output_type always
    came back Any regardless of the tool's actual signature, and a real
    type mismatch involving a Tool stage would build successfully instead
    of raising."""
    from composeai.tools import tool

    @tool
    def to_upper(text: str) -> str:
        """Uppercase.

        Args:
            text: input text
        """
        return text.upper()

    @tool
    def count_words(n: int) -> int:
        """Count.

        Args:
            n: a number
        """
        return n

    # str -> str is fine.
    p = pipe(to_upper, to_upper)
    assert p("hi") == "HI"

    # str -> int is a real mismatch and must now be caught at build time.
    with pytest.raises(CompositionTypeError) as exc_info:
        pipe(to_upper, count_words)
    message = str(exc_info.value)
    assert "str" in message
    assert "int" in message


# --- input_type / output_type introspection ------------------------------------


def test_agent_function_input_type_from_first_param():
    @agent(model=FakeModel(["x"]))
    def f(topic: str) -> str:
        """F."""
        return topic

    assert f.input_type is str


def test_agent_function_input_type_any_when_no_params():
    @agent(model=FakeModel(["x"]))
    def f() -> str:
        """F."""
        return "go"

    assert f.input_type is Any


def test_pipeline_input_output_type_first_and_last_stage():
    @agent(model=FakeModel(["x"]))
    def a(x: str) -> FactSheet:
        """A."""
        return x  # type: ignore[return-value]

    @agent(model=FakeModel([{"json": {"text": "s"}}]))
    def b(sheet: FactSheet) -> Summary:
        """B."""
        return str(sheet)  # type: ignore[return-value]

    pipeline = pipe(a, b)
    assert pipeline.input_type is str
    assert pipeline.output_type is Summary


def test_aggregate_rejects_zero_branches():
    with pytest.raises(CompositionTypeError):
        aggregate()


def test_aggregate_input_type_common_or_any_when_disagreeing():
    @agent(model=FakeModel(["x"]))
    def a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["y"]))
    def b(x: str) -> str:
        """B."""
        return x

    same_input = aggregate(one=a, two=b)
    assert same_input.input_type is str
    assert same_input.output_type is dict

    @agent(model=FakeModel(["z"]))
    def c(n: int) -> str:
        """C."""
        return str(n)

    disagreeing = aggregate(one=a, two=c)
    assert disagreeing.input_type is Any


# --- end-to-end pipeline behavior -----------------------------------------------


def test_pipe_two_agents_chained_output_and_trace():
    @agent(model=FakeModel([{"json": {"facts": ["quantum supremacy"]}}]))
    def researcher(topic: str) -> FactSheet:
        """Research the topic."""
        return topic  # type: ignore[return-value]

    @agent(model=FakeModel(["A great blog post."]))
    def copywriter(sheet: FactSheet) -> str:
        """Write a blog post."""
        return str(sheet)

    pipeline = pipe(researcher, copywriter)
    run = pipeline.run("quantum computing")

    assert run.output == "A great blog post."
    assert run.status == "completed"
    assert run.messages == []

    pipe_span = next(s for s in run.trace.spans if s.kind == "pipe")
    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    assert len(agent_spans) == 2
    assert all(s.parent_span_id == pipe_span.span_id for s in agent_spans)

    # each agent's FakeModel default usage is 10 input / 20 output tokens
    assert run.usage.input_tokens == 20
    assert run.usage.output_tokens == 40


def test_pipe_plain_function_stage_gets_task_span_with_io():
    @agent(model=FakeModel(["draft text"]))
    def drafter(topic: str) -> str:
        """Draft."""
        return topic

    def shout(text: str) -> str:
        return text.upper()

    @agent(model=FakeModel(["final polish"]))
    def polisher(text: str) -> str:
        """Polish."""
        return text

    pipeline = pipe(drafter, shout, polisher)
    run = pipeline.run("topic")
    assert run.output == "final polish"

    task_span = next(s for s in run.trace.spans if s.kind == "task")
    assert task_span.name == "shout"
    assert task_span.input == "draft text"
    assert task_span.output == "DRAFT TEXT"


def test_pipe_exception_propagates_unchanged_and_records_error():
    @agent(model=FakeModel(["ok"]))
    def a(x: str) -> str:
        """A."""
        return x

    def boom(x: str) -> str:
        raise ValueError("stage failed")

    pipeline = pipe(a, boom)
    with pytest.raises(ValueError, match="stage failed"):
        pipeline.run("go")


def test_nested_pipe_in_pipe():
    @agent(model=FakeModel(["step1 out"]))
    def a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["step2 out"]))
    def b(x: str) -> str:
        """B."""
        return x

    inner = pipe(a, b)

    @agent(model=FakeModel(["step3 out"]))
    def c(x: str) -> str:
        """C."""
        return x

    outer = pipe(inner, c)
    run = outer.run("start")
    assert run.output == "step3 out"

    pipe_spans = [s for s in run.trace.spans if s.kind == "pipe"]
    assert len(pipe_spans) == 2
    outer_pipe_span = next(s for s in pipe_spans if s.parent_span_id is None)
    inner_pipe_span = next(s for s in pipe_spans if s.span_id != outer_pipe_span.span_id)
    assert inner_pipe_span.parent_span_id == outer_pipe_span.span_id

    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    assert len(agent_spans) == 3
    assert all(s.parent_span_id == inner_pipe_span.span_id for s in agent_spans[:2])


# --- aggregate() -----------------------------------------------------------------


def test_aggregate_dict_result_and_declaration_order():
    @agent(model=FakeModel(["sec result"]))
    def security(code: str) -> str:
        """Security audit."""
        return code

    @agent(model=FakeModel(["perf result"]))
    def performance(code: str) -> str:
        """Perf audit."""
        return code

    audit = aggregate(sec=security, perf=performance)
    result = audit("some code")
    assert result == {"sec": "sec result", "perf": "perf result"}
    assert list(result.keys()) == ["sec", "perf"]


def test_aggregate_genuine_parallelism():
    def slow_a(x: str) -> str:
        time.sleep(0.15)
        return "a"

    def slow_b(x: str) -> str:
        time.sleep(0.15)
        return "b"

    audit = aggregate(one=slow_a, two=slow_b)
    start = time.monotonic()
    run = audit.run("x")
    elapsed = time.monotonic() - start
    assert run.output == {"one": "a", "two": "b"}
    assert elapsed < 0.25, f"branches did not run concurrently ({elapsed:.3f}s)"


def test_aggregate_deterministic_first_error_after_all_settle():
    # "a" is declared *first* but finishes *last* among the failing branches
    # (and "b" -- declared second -- fails first, chronologically) -- proving
    # the raised exception is picked by declaration order, not arrival order.
    finished: list[str] = []

    def branch_a(x):
        time.sleep(0.15)
        finished.append("a")
        raise ValueError("a failed")

    def branch_b(x):
        time.sleep(0.05)
        finished.append("b")
        raise RuntimeError("b failed")

    def branch_c(x):
        time.sleep(0.1)
        finished.append("c")
        return "c ok"

    audit = aggregate(a=branch_a, b=branch_b, c=branch_c)
    with pytest.raises(ValueError, match="a failed"):
        audit("x")

    assert set(finished) == {"a", "b", "c"}  # every branch ran to completion
    assert finished.index("b") < finished.index("a")  # b failed first, chronologically


def test_aggregate_spans_parent_correctly():
    @agent(model=FakeModel(["sec"]))
    def security(code: str) -> str:
        """Sec."""
        return code

    @agent(model=FakeModel(["perf"]))
    def performance(code: str) -> str:
        """Perf."""
        return code

    audit = aggregate(sec=security, perf=performance)
    run = audit.run("code")

    agg_span = next(s for s in run.trace.spans if s.kind == "aggregate")
    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    assert len(agent_spans) == 2
    assert all(s.parent_span_id == agg_span.span_id for s in agent_spans)


# --- compose.map -------------------------------------------------------------------


def test_map_preserves_order():
    def double(x: int) -> int:
        return x * 2

    results = compose.map(double, [1, 2, 3, 4])
    assert results == [2, 4, 6, 8]


def test_map_runs_in_parallel():
    def slow(x: int) -> int:
        time.sleep(0.15)
        return x

    start = time.monotonic()
    results = compose.map(slow, [1, 2, 3])
    elapsed = time.monotonic() - start
    assert results == [1, 2, 3]
    assert elapsed < 0.3, f"map items did not run concurrently ({elapsed:.3f}s)"


def test_map_error_at_index_raised_deterministically():
    # index 1 fails *slowly*; index 3 fails *fast* (no sleep) -- proving the
    # raised exception is picked by index, not by which one raises first.
    def flaky(item: tuple[int, str]) -> str:
        index, label = item
        if index == 1:
            time.sleep(0.1)
            raise ValueError("bad item at index 1")
        if index == 3:
            raise ValueError("bad item at index 3")
        time.sleep(0.05)
        return label

    items = [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")]
    with pytest.raises(ValueError, match="bad item at index 1"):
        compose.map(flaky, items)


def test_map_item_task_spans():
    def fn(x: int) -> int:
        return x + 1

    with tracing.span("flow", "root"):
        compose.map(fn, [10, 20])
        trace = tracing.current_trace()

    assert trace is not None
    task_spans = sorted((s for s in trace.spans if s.kind == "task"), key=lambda s: s.name)
    assert [s.name for s in task_spans] == ["fn[0]", "fn[1]"]


def test_map_wrapper_span_is_aggregate_kind():
    def fn(x: int) -> int:
        return x

    with tracing.span("flow", "root"):
        compose.map(fn, [1])
        trace = tracing.current_trace()

    assert trace is not None
    agg_span = next(s for s in trace.spans if s.kind == "aggregate")
    assert agg_span.name == "map(fn)"


# --- streaming: pipeline / aggregate ------------------------------------------------


def test_pipeline_stream_yields_both_agents_text_deltas_and_terminal_run_finished():
    @agent(model=FakeModel(["hello from a"]))
    def a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["hello from b"]))
    def b(x: str) -> str:
        """B."""
        return x

    pipeline = pipe(a, b)
    run_stream = pipeline.stream("go")
    events_list = list(run_stream)

    by_span: dict[str | None, str] = defaultdict(str)
    for e in events_list:
        if e.kind == "text_delta":
            by_span[e.span_id] += e.text or ""
    assert set(by_span.values()) == {"hello from a", "hello from b"}

    assert events_list[-1].kind == "run_finished"
    assert events_list[-1].data == {"status": "completed"}

    run = run_stream.run
    assert run.output == "hello from b"


def test_pipeline_stream_run_property_works_without_draining():
    @agent(model=FakeModel(["out a"]))
    def a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["out b"]))
    def b(x: str) -> str:
        """B."""
        return x

    pipeline = pipe(a, b)
    run_stream = pipeline.stream("go")
    assert run_stream.run.output == "out b"


def test_aggregate_stream_yields_branches_and_terminal_run_finished():
    @agent(model=FakeModel(["sec out"]))
    def sec(x: str) -> str:
        """Sec."""
        return x

    @agent(model=FakeModel(["perf out"]))
    def perf(x: str) -> str:
        """Perf."""
        return x

    audit = aggregate(sec=sec, perf=perf)
    run_stream = audit.stream("code")
    events_list = list(run_stream)
    assert events_list[-1].kind == "run_finished"
    assert events_list[-1].data == {"status": "completed"}
    assert run_stream.run.output == {"sec": "sec out", "perf": "perf out"}


# --- exports -----------------------------------------------------------------------


def test_pipe_aggregate_map_budget_exported_from_top_level_package():
    assert compose.pipe is pipe
    assert compose.aggregate is aggregate
    assert callable(compose.map)
    assert compose.Budget is not None


def test_pipeline_stream_emits_exactly_one_run_finished():
    """Nested runs must not emit run_finished — only the trace root does."""
    from composeai.testing import FakeModel

    @agent(model=FakeModel(script=["alpha"]))
    def first(text: str) -> str:
        """First."""
        return text

    @agent(model=FakeModel(script=["beta"]))
    def second(text: str) -> str:
        """Second."""
        return text

    pipeline = pipe(first, second)
    events_seen = list(pipeline.stream("go"))
    finished = [e for e in events_seen if e.kind == "run_finished"]
    assert len(finished) == 1
    assert events_seen[-1].kind == "run_finished"
    assert finished[0].data == {"status": "completed"}


def test_map_span_names_use_task_name_not_repr():
    """Regression: compose.map over a @task must render `map(fetch)`, not a repr."""
    from composeai import task

    @task
    def fetch(u: str) -> str:
        return u

    with tracing.span("flow", "outer"):
        trace = tracing.current_trace()
        compose.map(fetch, ["a", "b"])
    assert trace is not None
    names = [s.name for s in trace.spans if s.kind == "aggregate"]
    assert names == ["map(fetch)"]


def test_map_timeout_per_item_raises():
    import time as _time

    from composeai.errors import TaskTimeoutError

    def task9_slow(x: int) -> int:
        if x == 1:
            _time.sleep(10)
        return x

    with pytest.raises(TaskTimeoutError):
        compose.map(task9_slow, [0, 1], timeout_per_item=0.2)


def test_map_collect_returns_per_item_results():
    from composeai import MapResult

    def task9_flaky(x: int) -> int:
        if x == 1:
            raise ValueError("boom")
        return x * 2

    results = compose.map(task9_flaky, [0, 1, 2], on_error="collect")
    assert [r.ok for r in results] == [True, False, True]
    assert results[0] == MapResult(ok=True, value=0)
    assert results[2].value == 4
    assert results[1].error_type == "ValueError"
    assert "boom" in results[1].error


def test_map_collect_captures_timeouts_too():
    import time as _time

    def task9_slow_collect(x: int) -> int:
        if x == 1:
            _time.sleep(10)
        return x

    results = compose.map(
        task9_slow_collect, [0, 1], timeout_per_item=0.2, on_error="collect"
    )
    assert results[0].ok is True
    assert results[1].ok is False
    assert results[1].error_type == "TaskTimeoutError"


def test_map_invalid_on_error_rejected():
    from composeai.errors import ConfigError

    with pytest.raises(ConfigError):
        compose.map(lambda x: x, [1], on_error="ignore")


def test_aggregate_timeout_per_branch_raises():
    import time as _time

    from composeai.errors import TaskTimeoutError

    def fast(x: int) -> int:
        return x + 1

    def hung(x: int) -> int:
        _time.sleep(10)
        return x

    agg = compose.aggregate(timeout_per_branch=0.2, fast=fast, hung=hung)
    with pytest.raises(TaskTimeoutError):
        agg(1)


def test_aggregate_no_timeout_unchanged():
    agg = compose.aggregate(a=lambda x: x + 1, b=lambda x: x * 2)
    assert agg(3) == {"a": 4, "b": 6}


def test_aggregate_rejects_positional_stage():
    with pytest.raises(TypeError):
        compose.aggregate(lambda x: x)  # type: ignore[misc]  # positional args are not branches


def test_aggregate_rejects_non_numeric_timeout():
    from composeai.errors import ConfigError

    with pytest.raises(ConfigError):
        compose.aggregate(timeout_per_branch=lambda x: x, a=lambda x: x)  # type: ignore[arg-type]
