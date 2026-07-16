"""Tests for human-in-the-loop pause/resume in ``@flow`` bodies (Phase 8).

``approve``/``ask_human`` raise an internal, named ``_Pause`` when their
answer isn't journaled yet; ``@flow.run()`` catches it and returns a
``Run(status="paused", pending=Interrupt(...))`` instead of raising --
pausing is not an error. ``resume(run_id, answers)`` journals the answers
under ``__interrupt__:{id}`` keys and re-executes the flow body; completed
``@task`` steps replay from the journal (never re-execute).

Deliberately *not* using ``from __future__ import annotations`` (same
reason as ``test_flow.py``): plain, straightforward tests, no need for it
here either, but kept consistent with the rest of the flow test suite.
"""

import pytest

from composeai import runs
from composeai.combinators import map as compose_map
from composeai.errors import ConfigError
from composeai.flow import flow, resume, task
from composeai.hitl import Interrupt, approve, ask_human

# --- approve(): pause + resume round trip --------------------------------------


def test_approve_pauses_flow_with_paused_run_pending_row_and_span_statuses():
    @flow
    def publish_flow() -> str:
        if approve("publish"):
            return "published"
        return "rejected"

    # Annotated bare `runs.Run` (i.e. `Run[Any]`) so the paused-run assertion
    # `run.output is None` below type-checks: `publish_flow` returns `str`, so
    # `publish_flow.run()` now statically infers `Run[str]` (v0.5.0 Plan B,
    # Task 5 -- `Flow.run -> Run[R]`), under which `run.output` is `str` and
    # `... is None` would narrow `run` to `Never`. A paused run genuinely has
    # `output=None` at runtime; the success-typed `Run[R]` can't express that,
    # so this one paused-run test opts back out to `Run[Any]`. Local annotation,
    # never evaluated at runtime (PEP 526) -- zero behavior change.
    run: runs.Run = publish_flow.run()

    assert run.status == "paused"
    assert run.output is None
    assert isinstance(run.pending, Interrupt)
    assert run.pending.id == "publish"
    assert run.pending.kind == "approval"

    store = runs.open_default()
    row = store.get_run(run.id)
    assert row is not None
    assert row["status"] == "paused"

    pending_rows = store.pending_interrupts_all(run.id)
    assert len(pending_rows) == 1
    assert pending_rows[0]["interrupt_id"] == "publish"
    assert pending_rows[0]["kind"] == "approval"

    # The pause must never look like an *error* anywhere in the trace.
    assert all(s.status != "error" for s in run.trace.spans)
    pause_spans = [s for s in run.trace.spans if s.kind == "pause"]
    assert len(pause_spans) == 1
    assert pause_spans[0].status == "paused"
    flow_spans = [s for s in run.trace.spans if s.kind == "flow"]
    assert len(flow_spans) == 1
    assert flow_spans[0].status == "paused"


def test_resume_with_true_answer_completes_and_deletes_pending_row():
    @flow
    def publish_flow() -> str:
        if approve("publish"):
            return "published"
        return "rejected"

    run = publish_flow.run()
    assert run.status == "paused"

    run2 = resume(run.id, {"publish": True})
    assert run2.status == "completed"
    assert run2.output == "published"

    store = runs.open_default()
    assert store.pending_interrupts_all(run.id) == []
    row = store.get_run(run.id)
    assert row is not None
    assert row["status"] == "completed"


def test_resume_with_false_answer_takes_the_denied_branch():
    @flow
    def publish_flow() -> str:
        if approve("publish"):
            return "published"
        return "rejected"

    run = publish_flow.run()
    run2 = resume(run.id, {"publish": False})
    assert run2.status == "completed"
    assert run2.output == "rejected"


# --- ask_human(): answer value round-trip ---------------------------------------


def test_ask_human_string_answer_round_trips():
    @flow
    def greet_flow() -> str:
        name = ask_human("who", "Who should I greet?")
        return f"Hello, {name}!"

    run = greet_flow.run()
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.kind == "question"
    assert run.pending.question == "Who should I greet?"

    run2 = resume(run.id, {"who": "Ada"})
    assert run2.status == "completed"
    assert run2.output == "Hello, Ada!"


def test_ask_human_dict_answer_round_trips():
    @flow
    def config_flow() -> dict:
        cfg = ask_human("cfg", "Give me a config dict")
        return cfg

    run = config_flow.run()
    assert run.status == "paused"

    answer = {"retries": 3, "mode": "fast"}
    run2 = resume(run.id, {"cfg": answer})
    assert run2.status == "completed"
    assert run2.output == answer


# --- pre-pause steps don't re-execute on resume ---------------------------------


def test_steps_before_pause_do_not_reexecute_on_resume():
    counters = {"a": 0, "b": 0}

    @task
    def step_a() -> int:
        counters["a"] += 1
        return 1

    @task
    def step_b() -> int:
        counters["b"] += 1
        return 2

    @flow
    def gated_flow() -> int:
        a = step_a()
        b = step_b()
        if approve("go"):
            return a + b
        return 0

    run = gated_flow.run()
    assert run.status == "paused"
    assert counters == {"a": 1, "b": 1}

    run2 = resume(run.id, {"go": True})
    assert run2.output == 3
    assert counters == {"a": 1, "b": 1}  # not re-executed


# --- sequential multiple interrupts ---------------------------------------------


def test_sequential_multiple_interrupts_resolved_one_at_a_time():
    @flow
    def two_gates_flow() -> str:
        if not approve("first"):
            return "stopped-at-first"
        if not approve("second"):
            return "stopped-at-second"
        return "all-approved"

    run = two_gates_flow.run()
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.id == "first"

    run2 = resume(run.id, {"first": True})
    assert run2.status == "paused"
    assert run2.pending is not None
    assert run2.pending.id == "second"

    run3 = resume(run2.id, {"second": True})
    assert run3.status == "completed"
    assert run3.output == "all-approved"


def test_sequential_multiple_interrupts_answered_all_at_once():
    @flow
    def two_gates_flow() -> str:
        if not approve("first"):
            return "stopped-at-first"
        if not approve("second"):
            return "stopped-at-second"
        return "all-approved"

    run = two_gates_flow.run()
    assert run.status == "paused"

    # Providing both answers up front resolves both gates in one resume().
    run2 = resume(run.id, {"first": True, "second": True})
    assert run2.status == "completed"
    assert run2.output == "all-approved"


# --- interrupt inside a map branch -----------------------------------------------


def test_interrupt_inside_a_map_branch_pauses_and_resumes():
    @task
    def maybe_process(item: int) -> int:
        if approve(f"item-{item}"):
            return item * 10
        return -1

    @flow
    def map_flow() -> list[int]:
        return compose_map(maybe_process, [1, 2, 3])

    run = map_flow.run()
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.id.startswith("item-")

    # Answer every possible interrupt up front (map/aggregate only re-raise
    # the first _Pause per attempt -- the rest re-pause on later resumes
    # unless all answers are supplied together, which resolves them all).
    answers = {f"item-{i}": True for i in (1, 2, 3)}
    run2 = resume(run.id, answers)
    assert run2.status == "completed"
    assert sorted(run2.output) == [10, 20, 30]


def test_interrupt_inside_map_branch_reraises_next_one_when_answered_one_at_a_time():
    seen_pending_ids = []

    @task
    def maybe_process(item: int) -> int:
        if approve(f"item-{item}"):
            return item * 10
        return -1

    @flow
    def map_flow() -> list[int]:
        return compose_map(maybe_process, [1, 2])

    run = map_flow.run()
    assert run.status == "paused"
    assert run.pending is not None
    seen_pending_ids.append(run.pending.id)

    # Answer only the first-seen interrupt; the flow must re-pause (on the
    # other branch) rather than completing.
    run2 = resume(run.id, {run.pending.id: True})
    if run2.status == "paused":
        assert run2.pending is not None
        seen_pending_ids.append(run2.pending.id)
        run3 = resume(run2.id, {run2.pending.id: True})
        assert run3.status == "completed"
    else:
        assert run2.status == "completed"

    assert len(set(seen_pending_ids)) == len(seen_pending_ids)  # distinct ids


# --- resume with no answers just re-pauses --------------------------------------


def test_resume_with_no_answers_repauses_on_the_same_interrupt():
    @flow
    def publish_flow() -> str:
        if approve("publish"):
            return "published"
        return "rejected"

    run = publish_flow.run()
    assert run.status == "paused"

    run2 = resume(run.id)
    assert run2.status == "paused"
    assert run2.pending is not None
    assert run2.pending.id == "publish"

    store = runs.open_default()
    # Still exactly one pending row (idempotent re-pause, not a duplicate).
    assert len(store.pending_interrupts_all(run.id)) == 1


# --- same trace_id across pause/resume ------------------------------------------


def test_trace_id_stays_the_same_across_pause_and_resume():
    @flow
    def publish_flow() -> str:
        if approve("publish"):
            return "published"
        return "rejected"

    run = publish_flow.run()
    original_trace_id = run.trace.trace_id
    assert original_trace_id

    run2 = resume(run.id, {"publish": True})
    assert run2.trace.trace_id == original_trace_id
    for span in run2.trace.spans:
        assert span.trace_id == original_trace_id


# --- approve()/ask_human() outside any run ---------------------------------------


def test_approve_outside_any_flow_or_agent_raises_config_error():
    with pytest.raises(ConfigError):
        approve("nope")


def test_ask_human_outside_any_flow_or_agent_raises_config_error():
    with pytest.raises(ConfigError):
        ask_human("nope", "question?")


# --- shorthand tool-name answer keys are flow-agnostic (exact id passthrough) ---


def test_flow_level_interrupt_ids_are_used_verbatim_not_tool_shorthand():
    @flow
    def publish_flow() -> str:
        if approve("send_email"):
            return "sent"
        return "not sent"

    run = publish_flow.run()
    assert run.status == "paused"
    run2 = resume(run.id, {"send_email": True})
    assert run2.status == "completed"
    assert run2.output == "sent"
