"""Tests for nested ``Pipeline``/``Aggregate`` run/trace adoption (v0.5.0
Plan A, Tasks 1-2, plus a follow-up closing the final review's finding I1).

Design under test: a ``pipe()``/``aggregate()`` result CALLED (``__call__``)
while a span or run context is already ambient (a ``@flow`` body, a bare
``@task`` body, ...) must join the enclosing trace/run as a nested step --
the same shape ``_ainvoke_stage``'s Pipeline/Aggregate-as-a-stage branches
have always used -- instead of always minting its own independent ``runs``
row via ``_run_top`` (the cross-trace/cross-run linkage bug; see
``plans/superpowers/research/2026-07-16-typing/trace-linkage-investigation.md``).
``await inner.arun(x)``/``inner.astream(x)`` join the SAME way -- this is
NOT the asymmetry ``Flow`` has (see below); only the SYNC ``.run()``/
``.stream()`` calls are unaffected -- they always start their own
independent run, mirroring ``Flow.run()``'s own semantics when called from
inside another flow (``Flow.arun()`` adopts; only ``Flow.run()`` doesn't --
docs/async.md's "run()/arun() nested asymmetry" section). The original Task
1-2 implementation left ``Pipeline.arun``/``Aggregate.arun``/``.astream``
unconditionally routing through ``_arun_top`` -- independent of ambient
context, unlike ``Flow.arun``/``Task.arun``/``AgentFunction.arun`` -- citing
this asymmetry as the reason; that citation was wrong (``Flow``'s asymmetry
is ``.run()``-only, its ``arun`` already adopts), and the gap it left open on
the async path is what the final review's finding I1 caught and this file's
``*_async`` tests below pin closed.

New file (rather than adding to the already-874-line ``test_combinators.py``)
per the task brief's "implementer's choice" -- this also gives Task 3 of the
same plan (budget/usage/pause correctness batch) a dedicated, uncluttered
home.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from composeai import runs, tracing
from composeai.agentfn import agent
from composeai.cli import _rebuild_trace
from composeai.combinators import aggregate, pipe
from composeai.errors import BudgetExceededError
from composeai.flow import flow, resume, task
from composeai.hitl import approve
from composeai.runs import Budget
from composeai.testing import FakeModel

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_nested_pipe_call_joins_enclosing_flow_trace():
    @agent(model=FakeModel(["stage-a-out"]))
    def nc_stage_a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["stage-b-out"]))
    def nc_stage_b(x: str) -> str:
        """B."""
        return x

    the_pipe = pipe(nc_stage_a, nc_stage_b)

    @flow
    def nc_my_flow(x: str) -> str:
        return the_pipe(x)

    run = nc_my_flow.run("start")
    assert run.output == "stage-b-out"

    store = runs.open_default()

    # Exactly one runs row -- the flow's. No separate "pipe"-kind row was
    # minted for the nested `the_pipe(x)` call.
    pipe_rows = [
        r for r in store.list_runs(kind="pipe", limit=200) if r["name"] == the_pipe._name
    ]
    assert pipe_rows == []
    flow_rows = [r for r in store.list_runs(kind="flow", limit=200) if r["name"] == "nc_my_flow"]
    assert len(flow_rows) == 1
    assert flow_rows[0]["run_id"] == run.id

    # All pipe/agent/llm spans share the flow's own trace_id and run_id.
    flow_root = next(s for s in run.trace.spans if s.kind == "flow")
    pipe_span = next(s for s in run.trace.spans if s.kind == "pipe")
    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    llm_spans = [s for s in run.trace.spans if s.kind == "llm"]
    assert len(agent_spans) == 2
    assert len(llm_spans) == 2
    assert pipe_span.trace_id == flow_root.trace_id == run.trace.trace_id
    assert all(s.trace_id == run.trace.trace_id for s in agent_spans + llm_spans)

    # The pipe span's parent is the flow root span.
    assert pipe_span.parent_span_id == flow_root.span_id

    # `compose trace`-equivalent rendering shows the pipe nested under the
    # flow root, not as an invisible or dangling-parented subtree.
    rendered = tracing.render_trace(run.trace, color=False)
    lines = rendered.splitlines()
    flow_line_idx = next(i for i, line in enumerate(lines) if "nc_my_flow" in line)
    pipe_line_idx = next(i for i, line in enumerate(lines) if the_pipe._name in line)
    assert pipe_line_idx > flow_line_idx
    # Indented further than the flow root line -- i.e. a child, not a sibling.
    flow_indent = len(lines[flow_line_idx]) - len(lines[flow_line_idx].lstrip())
    pipe_indent = len(lines[pipe_line_idx]) - len(lines[pipe_line_idx].lstrip())
    assert pipe_indent > flow_indent


def test_nested_pipe_call_inside_bare_task_joins_ambient_trace():
    @agent(model=FakeModel(["stage-a-out2"]))
    def nc_bare_stage_a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["stage-b-out2"]))
    def nc_bare_stage_b(x: str) -> str:
        """B."""
        return x

    the_pipe = pipe(nc_bare_stage_a, nc_bare_stage_b)
    captured: dict[str, tracing.Trace | None] = {"trace": None}

    @task
    def nc_task_wraps_pipe(x: str) -> str:
        captured["trace"] = tracing.current_trace()
        return the_pipe(x)

    # No enclosing @flow at all -- current_run_context() is None the whole
    # time; only tracing.current_span() (the bare task's own span) is set.
    assert runs.current_run_context() is None
    result = nc_task_wraps_pipe("start")
    assert result == "stage-b-out2"

    trace = captured["trace"]
    assert trace is not None
    task_span = next(s for s in trace.spans if s.kind == "task" and s.name == "nc_task_wraps_pipe")
    pipe_span = next(s for s in trace.spans if s.kind == "pipe")

    assert pipe_span.trace_id == task_span.trace_id
    assert pipe_span.parent_span_id == task_span.span_id

    # No new runs row was created for the nested pipe call (there's no
    # durable run at all here -- a bare @task outside any @flow never mints
    # one, and the nested pipe must not mint one either).
    store = runs.open_default()
    pipe_rows = [
        r for r in store.list_runs(kind="pipe", limit=200) if r["name"] == the_pipe._name
    ]
    assert pipe_rows == []


def test_top_level_pipe_call_unchanged():
    @agent(model=FakeModel(["top-a-out"]))
    def nc_top_a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["top-b-out"]))
    def nc_top_b(x: str) -> str:
        """B."""
        return x

    p = pipe(nc_top_a, nc_top_b)

    # No ambient span/run context at all at the point of the call.
    assert tracing.current_span() is None
    assert runs.current_run_context() is None

    result = p("start")
    assert result == "top-b-out"

    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="pipe", limit=200) if r["name"] == p._name]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"

    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    try:
        trace = _rebuild_trace(conn, row["trace_id"])
    finally:
        conn.close()

    roots = trace.roots()
    assert len(roots) == 1
    assert roots[0].kind == "pipe"
    rendered = tracing.render_trace(trace, color=False)
    # line 0 is the trace-level header ("trace <id> -- status -- [...]");
    # line 1 is the pipe's own root node.
    assert p._name in rendered.splitlines()[1]


# --- Aggregate nested __call__ adoption (Task 2) ------------------------------


def test_nested_aggregate_call_joins_enclosing_flow_trace():
    @agent(model=FakeModel(["alpha-out"]))
    def nc_agg_alpha(x: str) -> str:
        """Alpha."""
        return x

    @agent(model=FakeModel(["beta-out"]))
    def nc_agg_beta(x: str) -> str:
        """Beta."""
        return x

    the_agg = aggregate(nc1alpha=nc_agg_alpha, nc1beta=nc_agg_beta)

    @flow
    def nc_agg_my_flow(x: str) -> dict[str, str]:
        return the_agg(x)

    run = nc_agg_my_flow.run("start")
    assert run.output == {"nc1alpha": "alpha-out", "nc1beta": "beta-out"}

    store = runs.open_default()

    # Exactly one runs row -- the flow's. No separate "aggregate"-kind row
    # was minted for the nested `the_agg(x)` call.
    agg_rows = [
        r for r in store.list_runs(kind="aggregate", limit=200) if r["name"] == the_agg._name
    ]
    assert agg_rows == []
    flow_rows = [
        r for r in store.list_runs(kind="flow", limit=200) if r["name"] == "nc_agg_my_flow"
    ]
    assert len(flow_rows) == 1
    assert flow_rows[0]["run_id"] == run.id

    # All aggregate/agent/llm spans share the flow's own trace_id and run_id.
    flow_root = next(s for s in run.trace.spans if s.kind == "flow")
    agg_span = next(s for s in run.trace.spans if s.kind == "aggregate")
    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    llm_spans = [s for s in run.trace.spans if s.kind == "llm"]
    assert len(agent_spans) == 2
    assert len(llm_spans) == 2
    assert agg_span.trace_id == flow_root.trace_id == run.trace.trace_id
    assert all(s.trace_id == run.trace.trace_id for s in agent_spans + llm_spans)

    # The aggregate span's parent is the flow root span, and both branches'
    # agent spans are parented under the aggregate span -- the investigation's
    # "worse than pipe" pin: today this renders ZERO children under the flow
    # root at all; after this fix it renders the aggregate node AND both
    # branch subtrees.
    assert agg_span.parent_span_id == flow_root.span_id
    assert all(s.parent_span_id == agg_span.span_id for s in agent_spans)

    rendered = tracing.render_trace(run.trace, color=False)
    lines = rendered.splitlines()
    flow_line_idx = next(i for i, line in enumerate(lines) if "nc_agg_my_flow" in line)
    agg_line_idx = next(i for i, line in enumerate(lines) if the_agg._name in line)
    alpha_line_idx = next(i for i, line in enumerate(lines) if "nc_agg_alpha" in line)
    beta_line_idx = next(i for i, line in enumerate(lines) if "nc_agg_beta" in line)
    assert flow_line_idx < agg_line_idx < alpha_line_idx
    assert agg_line_idx < beta_line_idx

    def indent_of(idx: int) -> int:
        return len(lines[idx]) - len(lines[idx].lstrip())

    # Each is indented further than its parent -- i.e. children, not siblings.
    assert indent_of(agg_line_idx) > indent_of(flow_line_idx)
    assert indent_of(alpha_line_idx) > indent_of(agg_line_idx)
    assert indent_of(beta_line_idx) > indent_of(agg_line_idx)


def test_nested_aggregate_call_inside_bare_task_joins_ambient_trace():
    @agent(model=FakeModel(["alpha-out2"]))
    def nc_bare_agg_alpha(x: str) -> str:
        """Alpha."""
        return x

    @agent(model=FakeModel(["beta-out2"]))
    def nc_bare_agg_beta(x: str) -> str:
        """Beta."""
        return x

    the_agg = aggregate(nc2alpha=nc_bare_agg_alpha, nc2beta=nc_bare_agg_beta)
    captured: dict[str, tracing.Trace | None] = {"trace": None}

    @task
    def nc_task_wraps_agg(x: str) -> dict[str, str]:
        captured["trace"] = tracing.current_trace()
        return the_agg(x)

    # No enclosing @flow at all -- current_run_context() is None the whole
    # time; only tracing.current_span() (the bare task's own span) is set.
    assert runs.current_run_context() is None
    result = nc_task_wraps_agg("start")
    assert result == {"nc2alpha": "alpha-out2", "nc2beta": "beta-out2"}

    trace = captured["trace"]
    assert trace is not None
    task_span = next(s for s in trace.spans if s.kind == "task" and s.name == "nc_task_wraps_agg")
    agg_span = next(s for s in trace.spans if s.kind == "aggregate")

    assert agg_span.trace_id == task_span.trace_id
    assert agg_span.parent_span_id == task_span.span_id

    # No new runs row was created for the nested aggregate call (there's no
    # durable run at all here -- a bare @task outside any @flow never mints
    # one, and the nested aggregate must not mint one either).
    store = runs.open_default()
    agg_rows = [
        r for r in store.list_runs(kind="aggregate", limit=200) if r["name"] == the_agg._name
    ]
    assert agg_rows == []


def test_top_level_aggregate_call_unchanged():
    @agent(model=FakeModel(["top-alpha-out"]))
    def nc_top_agg_alpha(x: str) -> str:
        """Alpha."""
        return x

    @agent(model=FakeModel(["top-beta-out"]))
    def nc_top_agg_beta(x: str) -> str:
        """Beta."""
        return x

    agg = aggregate(nc3alpha=nc_top_agg_alpha, nc3beta=nc_top_agg_beta)

    # No ambient span/run context at all at the point of the call.
    assert tracing.current_span() is None
    assert runs.current_run_context() is None

    result = agg("start")
    assert result == {"nc3alpha": "top-alpha-out", "nc3beta": "top-beta-out"}

    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="aggregate", limit=200) if r["name"] == agg._name]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"

    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    try:
        trace = _rebuild_trace(conn, row["trace_id"])
    finally:
        conn.close()

    roots = trace.roots()
    assert len(roots) == 1
    assert roots[0].kind == "aggregate"
    rendered = tracing.render_trace(trace, color=False)
    # line 0 is the trace-level header ("trace <id> -- status -- [...]");
    # line 1 is the aggregate's own root node.
    assert agg._name in rendered.splitlines()[1]


def test_nested_aggregate_call_branch_journal_scoping_deterministic_across_resume():
    """Task brief Step 1: the per-branch journal scope reservation discipline
    (``Aggregate._arun_branches``'s ``ctx.reserve_scope_segment("aggregate")``
    dance, unchanged by this task -- see combinators.py) must still produce
    deterministic, per-branch-qualified journal keys -- and support crash/
    resume without re-executing already-journaled branch steps -- when the
    aggregate is entered via ``agg(x)`` (``__call__``) directly, not just via
    ``.run()`` or as a stage of an outer combinator (that path is already
    covered by
    ``test_aggregate_branches_in_flow_get_deterministic_per_branch_scope``
    in test_flow.py).
    """
    calls: list[str] = []

    @task
    def nc_agg_journal_task_a() -> str:
        calls.append("a")
        return "a-result"

    @task
    def nc_agg_journal_task_b() -> str:
        calls.append("b")
        return "b-result"

    agg = aggregate(
        nc4alpha=lambda x: nc_agg_journal_task_a(), nc4beta=lambda x: nc_agg_journal_task_b()
    )

    should_fail = {"value": True}

    @task
    def nc_agg_journal_after() -> str:
        if should_fail["value"]:
            raise RuntimeError("boom")
        return "after-done"

    @flow
    def nc_agg_journal_flow(x: str) -> dict[str, str]:
        result = agg(x)
        nc_agg_journal_after()
        return result

    with pytest.raises(RuntimeError, match="boom"):
        nc_agg_journal_flow.run("start")
    assert set(calls) == {"a", "b"}

    store = runs.open_default()
    rows = [
        r for r in store.list_runs(kind="flow", limit=50) if r["name"] == "nc_agg_journal_flow"
    ]
    run_row = rows[0]
    assert run_row["status"] == "failed"

    # Deterministic, per-branch-qualified journal keys -- reserved serially
    # in declaration order on the engine coroutine before either branch's
    # `@task` body actually ran, exactly as when the aggregate is used as a
    # stage of an outer combinator.
    keys = set(store.journal_all(run_row["run_id"]).keys())
    assert keys == {
        "aggregate#1/nc4alpha/nc_agg_journal_task_a#1",
        "aggregate#1/nc4beta/nc_agg_journal_task_b#1",
    }

    # No separate "aggregate"-kind runs row was minted for the nested call.
    agg_rows = [r for r in store.list_runs(kind="aggregate", limit=200) if r["name"] == agg._name]
    assert agg_rows == []

    calls.clear()
    should_fail["value"] = False
    run2 = resume(run_row["run_id"])
    assert run2.output == {"nc4alpha": "a-result", "nc4beta": "b-result"}
    # Both branch steps replayed from the journal -- neither @task re-executed.
    assert calls == []


# --- Task 3: budget/usage/pause correctness batch -----------------------------
#
# Ported from the empirical repros in
# plans/superpowers/research/2026-07-16-typing/trace-bug-repro/
# (repro_budget2.py, repro2.py) and the §3 sibling-cases table + budget-leak
# entry of trace-linkage-investigation.md. All four pass UNMODIFIED against
# the Tasks 1-2 code landed above -- Option A's nested-adoption fix (no
# second `Trace` object, no second `runs` row, ambient `RunContext` stays
# visible) closes the budget/usage leaks as a side effect, exactly as the
# investigation's "Fix options" section predicted. No src change was needed
# for this task; see the module-level report for confirmation these were
# verified empirically (not just asserted) against the staged diff.


def test_nested_pipe_budget_enforced_cumulatively():
    """Port of ``repro_budget2.py``: a flow spends 30 tok via a plain agent
    BEFORE calling a pipe whose two stages spend 30 tok each. With
    ``Budget(tokens=45)``, true cumulative spend right after the pipe's
    first stage (``stage_a``) is already 60 tok (30 pre-pipe + 30 stage_a)
    > 45, so ``BudgetExceededError`` must fire there -- ``stage_b``'s
    ``FakeModel`` must record ZERO requests, and the error's reported
    ``used`` must count the full 60, not just the pipe-local 30.

    Before Tasks 1-2, the nested pipe call ran inside its own, separate
    ``Trace`` object (``_run_top``'s unconditional ``tracing.use_trace()``),
    so ``check_budgets()``'s ``current_trace().rollup_usage(root_span)``
    could only see pipe-local spend (30 < 45) right after ``stage_a`` --
    letting one extra, unbudgeted LLM call (``stage_b``) slip through
    before the pipe-local rollup finally reached 60 > 45 afterward (the
    investigation's budget-leak entry). This passes unmodified against the
    Task 1-2 code: nested adoption means there is no second ``Trace``
    object at all, so ``check_budgets()`` already sees the flow's own root
    span's full subtree (pre-pipe spend + pipe spend) the moment it's
    called from inside ``stage_a``.
    """
    model_pre = FakeModel(["plain"] * 10)
    model_a = FakeModel(["A"] * 10)
    model_b = FakeModel(["B"] * 10)

    @agent(model=model_pre, name="nc3_budget_plain_agent")
    def nc3_budget_plain_agent(x: str) -> str:
        """Plain."""
        return f"p:{x}"

    @agent(model=model_a, name="nc3_budget_stage_a")
    def nc3_budget_stage_a(x: str) -> str:
        """A."""
        return f"a:{x}"

    @agent(model=model_b, name="nc3_budget_stage_b")
    def nc3_budget_stage_b(x: str) -> str:
        """B."""
        return f"b:{x}"

    the_pipe = pipe(nc3_budget_stage_a, nc3_budget_stage_b)

    @flow
    def nc3_budget_flow(x: str) -> dict[str, str]:
        agent_result = nc3_budget_plain_agent(x)
        pipe_result = the_pipe(x)
        return {"agent_result": agent_result, "pipe_result": pipe_result}

    with pytest.raises(BudgetExceededError) as excinfo:
        nc3_budget_flow.run("in", budget=Budget(tokens=45))

    assert "used=60" in str(excinfo.value)
    assert len(model_pre.requests) == 1
    assert len(model_a.requests) == 1
    assert len(model_b.requests) == 0


def test_nested_pipe_usage_rolls_up():
    """Port of ``repro2.py``: a flow calls a plain agent (30 tok) THEN a
    pipe (60 tok total, 30 tok per stage) directly. ``run.usage`` must
    total 90 tok and ``run.trace`` must contain the full pipe subtree.

    Before Tasks 1-2, the nested pipe call built its spans against a
    completely separate, freshly-minted ``Trace`` object, so the flow's own
    in-memory ``Trace``/``Run.usage`` never saw the pipe's 60 tok at all
    (``run.usage`` == 30, only 3 spans total in ``run.trace``). This passes
    unmodified against the Task 1-2 code: the nested ``__call__`` adoption
    path never calls ``tracing.use_trace()`` -- it just opens a ``pipe``
    span nested under whatever span/trace is already ambient -- so there is
    only ever one ``Trace`` object for the whole flow run.
    """
    model_a = FakeModel(["A"])
    model_b = FakeModel(["B"])
    model_agent = FakeModel(["plain"])

    @agent(model=model_a, name="nc3_usage_stage_a")
    def nc3_usage_stage_a(x: str) -> str:
        """A."""
        return f"a:{x}"

    @agent(model=model_b, name="nc3_usage_stage_b")
    def nc3_usage_stage_b(x: str) -> str:
        """B."""
        return f"b:{x}"

    @agent(model=model_agent, name="nc3_usage_plain_agent")
    def nc3_usage_plain_agent(x: str) -> str:
        """Plain."""
        return f"p:{x}"

    the_pipe = pipe(nc3_usage_stage_a, nc3_usage_stage_b)

    @flow
    def nc3_usage_flow(x: str) -> dict[str, str]:
        pipe_result = the_pipe(x)
        agent_result = nc3_usage_plain_agent(x)
        return {"pipe_result": pipe_result, "agent_result": agent_result}

    run = nc3_usage_flow.run("in")

    assert run.usage.input_tokens + run.usage.output_tokens == 90
    trace_totals = run.trace.total_usage()
    assert trace_totals.input_tokens + trace_totals.output_tokens == 90

    kinds = [s.kind for s in run.trace.spans]
    assert kinds.count("llm") == 3
    pipe_span = next(s for s in run.trace.spans if s.kind == "pipe")
    flow_root = next(s for s in run.trace.spans if s.kind == "flow")
    assert pipe_span.trace_id == flow_root.trace_id == run.trace.trace_id
    assert pipe_span.parent_span_id == flow_root.span_id


def test_flow_pauses_inside_nested_pipe_and_resumes():
    """A pipe stage's ``approve()`` call pauses the ENCLOSING flow run when
    the pipe is called directly (``the_pipe(x)``, Task 1's nested-adoption
    path) rather than as a stage of an outer combinator (that already-legal
    shape is ``test_pipe_nested_as_a_flow_stage_can_pause_and_resume_normally``
    in ``test_flow.py``). The pre-pause stage (``nc3_pause_stage_a``, an
    ``@agent`` backed by ``FakeModel``) is journaled as one whole flow step
    (``agentfn._run_agent_journaled`` -- triggered by the ambient
    ``current_run_context()`` the nested pipe call now correctly keeps
    visible): on ``resume()``, that step replays from the journal without
    calling the model again -- pinned here by ``FakeModel.requests`` staying
    at length 1 across the pause/resume boundary. The post-pause stage (a
    plain callable, not whole-step journaled -- only its ``approve()`` call
    is) re-executes on resume with the now-journaled answer.
    """
    model_a = FakeModel(["stage-a-out"])

    @agent(model=model_a, name="nc3_pause_stage_a")
    def nc3_pause_stage_a(x: str) -> str:
        """A."""
        return x

    def nc3_pause_stage_b(x: str) -> str:
        if not approve("nc3_pause_go"):
            return "denied:" + x
        return "sent:" + x

    the_pipe = pipe(nc3_pause_stage_a, nc3_pause_stage_b)

    @flow
    def nc3_pause_flow(x: str) -> str:
        return the_pipe(x)

    run = nc3_pause_flow.run("hello")
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.id == "nc3_pause_go"
    assert len(model_a.requests) == 1

    run2 = resume(run.id, {"nc3_pause_go": True})
    assert run2.status == "completed"
    assert run2.output == "sent:stage-a-out"
    # The pre-pause stage's journaled step replayed from the store -- the
    # model was NOT called again.
    assert len(model_a.requests) == 1
    assert run2.trace.trace_id == run.trace.trace_id


def _run_subprocess_fixture(script_name: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FIXTURES_DIR / script_name)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_flow_pauses_inside_nested_pipe_and_resumes_cross_process(tmp_path):
    """Cross-process twin of
    ``test_flow_pauses_inside_nested_pipe_and_resumes`` -- reuses the shape
    of ``tests/test_hitl_subprocess.py``'s two-process pause/resume fixture
    pair (``subprocess_hitl_flow_defs.py``/``run_a.py``/``run_b.py``, itself
    the shape the task brief pointed at via ``test_async_surface.py``'s
    subprocess-driven gate): a brand-new process A pauses on the nested
    pipe's ``approve()`` stage and exits cleanly (status "paused" is not an
    error, same as any other Python script); a brand-new process B (no
    shared memory/import state with A) resumes by run id with the answer.

    A ``FakeModel`` instance can't survive across a process boundary, so
    non-re-execution of the pre-pause stage (``subproc_nested_pipe_stage_a``)
    is pinned by an external, file-based side-effect counter instead of
    request counts -- see
    ``tests/fixtures/subprocess_nested_pipe_hitl_flow_defs.py``.
    """
    compose_dir = tmp_path / "compose-store"
    run_id_file = tmp_path / "run_id.txt"
    counters_file = tmp_path / "counters.txt"
    pause_result_file = tmp_path / "pause_result.json"
    result_file = tmp_path / "result.json"

    base_env = dict(os.environ)
    base_env.update(
        COMPOSE_DIR=str(compose_dir),
        RUN_ID_FILE=str(run_id_file),
        COUNTERS_FILE=str(counters_file),
        PAUSE_RESULT_FILE=str(pause_result_file),
        RESULT_FILE=str(result_file),
    )

    # --- Process A: run until the nested pipe's approve() pauses it --------
    proc_a = _run_subprocess_fixture("subprocess_nested_pipe_hitl_run_a.py", base_env)
    assert proc_a.returncode == 0, (
        f"process A must exit cleanly on pause (pausing is not an error)\n"
        f"stdout: {proc_a.stdout}\nstderr: {proc_a.stderr}"
    )
    assert run_id_file.exists(), "flow.run() must write its run_id before exiting"
    run_id = run_id_file.read_text().strip()

    pause_result = json.loads(pause_result_file.read_text())
    assert pause_result["status"] == "paused"
    assert pause_result["pending_id"] == "nested_pipe_go"
    original_trace_id = pause_result["trace_id"]
    assert original_trace_id

    counters_after_a = counters_file.read_text().splitlines()
    assert counters_after_a == ["stage_a", "stage_b_attempt"]

    store = runs.RunStore(compose_dir / "runs.db")
    row_after_a = store.get_run(run_id)
    assert row_after_a is not None
    assert row_after_a["status"] == "paused"
    assert row_after_a["trace_id"] == original_trace_id

    # --- Process B: brand-new process, resumes with the approval answer ----
    proc_b = _run_subprocess_fixture("subprocess_nested_pipe_hitl_run_b.py", base_env)
    assert proc_b.returncode == 0, (
        f"resume process failed\nstdout: {proc_b.stdout}\nstderr: {proc_b.stderr}"
    )
    assert result_file.exists()
    result = json.loads(result_file.read_text())

    assert result["status"] == "completed"
    assert result["output"] == "sent:stage-a-out"
    assert result["run_id"] == run_id
    assert result["trace_id"] == original_trace_id  # one continuous trace

    # The pre-pause, LLM-backed stage (journaled as a whole flow step) must
    # NOT have re-executed on resume; the post-pause plain-callable stage
    # (not whole-step journaled -- only its `approve()` call is) does
    # re-execute, exactly as the same-process test above establishes.
    counters_after_b = counters_file.read_text().splitlines()
    assert counters_after_b.count("stage_a") == 1
    assert counters_after_b == [
        "stage_a",
        "stage_b_attempt",
        "stage_b_attempt",
        "stage_b_sent",
    ]

    store2 = runs.RunStore(compose_dir / "runs.db")
    row_after_b = store2.get_run(run_id)
    assert row_after_b is not None
    assert row_after_b["status"] == "completed"
    assert row_after_b["trace_id"] == original_trace_id


def test_nested_pipe_explicit_run_still_mints_independent_run():
    """The documented asymmetry (v0.5.0 Plan A -- docs land in Task 4): a
    ``the_pipe(x)`` __call__ inside a ``@flow`` body joins the enclosing run
    (Tasks 1-2), but an EXPLICIT ``the_pipe.run(x)`` call from the same
    place still mints its own, independent ``runs`` row -- mirroring
    ``Flow.run()``'s own semantics when called from inside another flow.
    Calling sync ``.run()``/``.stream()`` is always an explicit "start a
    new run" request, regardless of ambient context; ``__call__``,
    ``.arun()``, and ``.astream()`` adopt the ambient trace/run.
    """
    model_a = FakeModel(["a-out"])
    model_b = FakeModel(["b-out"])

    @agent(model=model_a, name="nc3_asym_a")
    def nc3_asym_a(x: str) -> str:
        """A."""
        return x

    @agent(model=model_b, name="nc3_asym_b")
    def nc3_asym_b(x: str) -> str:
        """B."""
        return x

    the_pipe = pipe(nc3_asym_a, nc3_asym_b)

    @flow
    def nc3_asym_flow(x: str) -> str:
        return the_pipe.run(x).output

    run = nc3_asym_flow.run("hi")
    assert run.output == "b-out"

    store = runs.open_default()
    pipe_rows = [r for r in store.list_runs(kind="pipe", limit=200) if r["name"] == the_pipe._name]
    assert len(pipe_rows) == 1
    assert pipe_rows[0]["status"] == "completed"

    flow_rows = [
        r for r in store.list_runs(kind="flow", limit=200) if r["name"] == "nc3_asym_flow"
    ]
    assert len(flow_rows) == 1
    # The pipe's own row carries its OWN trace_id, distinct from the flow's
    # -- an independent run, not a nested step.
    assert pipe_rows[0]["trace_id"] != flow_rows[0]["trace_id"]


# --- v0.5.0 Plan A follow-up (final-review finding I1): async-body arun() ------
#
# `Pipeline.arun`/`Aggregate.arun` (and `.astream`) never ctx-checked at all --
# unlike `Flow.arun`/`Task.arun`/`AgentFunction.arun`, which all adopt an
# ambient run/span context -- so an async flow body's `await the_pipe.arun(x)`
# still hit the pre-Plan-A bug triad (budget leak past an already-exceeded
# cumulative spend, usage undercount on the enclosing run, an orphan
# `kind="pipe"` runs row whose own trace rendered header-only), even though
# the sync sugar `the_pipe(x)` was already fixed by Tasks 1-2. Async twins of
# T1's join test and T3's budget/usage tests below, `await`ing `.arun()`
# instead of calling the sugar, driven through an `async def` @flow body
# (itself driven via the ordinary sync `.run()` facade, which bridges into an
# async flow body same as every other async-flow test in this suite).


def test_nested_pipe_arun_joins_enclosing_flow_trace_async():
    """Async twin of ``test_nested_pipe_call_joins_enclosing_flow_trace``
    (finding I1): an async flow body's ``await the_pipe.arun(x)`` must join
    the enclosing run/trace exactly like the sync ``the_pipe(x)`` sugar does
    -- one ``runs`` row (the flow's), the pipe span parented under the
    flow's own root span in the SAME trace, not a second, independent
    ``kind="pipe"`` row with its own trace_id/run_id.
    """

    @agent(model=FakeModel(["stage-a-out"]))
    def nc4_stage_a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["stage-b-out"]))
    def nc4_stage_b(x: str) -> str:
        """B."""
        return x

    the_pipe = pipe(nc4_stage_a, nc4_stage_b)

    @flow
    async def nc4_my_flow(x: str) -> str:
        r = await the_pipe.arun(x)
        return r.output

    run = nc4_my_flow.run("start")
    assert run.output == "stage-b-out"

    store = runs.open_default()

    # Exactly one runs row -- the flow's. No separate "pipe"-kind row was
    # minted for the nested `await the_pipe.arun(x)` call.
    pipe_rows = [
        r for r in store.list_runs(kind="pipe", limit=200) if r["name"] == the_pipe._name
    ]
    assert pipe_rows == []
    flow_rows = [r for r in store.list_runs(kind="flow", limit=200) if r["name"] == "nc4_my_flow"]
    assert len(flow_rows) == 1
    assert flow_rows[0]["run_id"] == run.id

    # All pipe/agent spans share the flow's own trace_id and run_id.
    flow_root = next(s for s in run.trace.spans if s.kind == "flow")
    pipe_span = next(s for s in run.trace.spans if s.kind == "pipe")
    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    assert len(agent_spans) == 2
    assert pipe_span.trace_id == flow_root.trace_id == run.trace.trace_id
    assert all(s.trace_id == run.trace.trace_id for s in agent_spans)

    # The pipe span's parent is the flow root span -- not a dangling parent
    # from some other trace.
    assert pipe_span.parent_span_id == flow_root.span_id


def test_nested_pipe_arun_budget_enforced_cumulatively_async():
    """Async twin of ``test_nested_pipe_budget_enforced_cumulatively``
    (finding I1): a flow spends 30 tok via ``await plain_agent.arun(x)``
    BEFORE ``await the_pipe.arun(x)`` whose two stages spend 30 tok each.
    With ``Budget(tokens=45)``, true cumulative spend right after the pipe's
    first stage is already 60 tok > 45, so ``BudgetExceededError`` must fire
    there -- ``stage_b``'s ``FakeModel`` must record ZERO requests, and the
    error's reported ``used`` must count the full 60, not just a pipe-local
    30.

    Before this fix, ``await the_pipe.arun(x)`` routed unconditionally
    through ``_arun_top``, which always opens its own fresh ``Trace``
    (``tracing.use_trace()``) -- so ``check_budgets()`` could only see
    pipe-local spend (30 < 45) right after stage_a, letting stage_b's
    unbudgeted extra call slip through (empirically confirmed alive by the
    final review's probe_arun_in_async_flow.py, the exact scenario this
    ports). This passes once ``arun`` ctx-checks and adopts: no second
    ``Trace`` object, so ``check_budgets()`` already sees the flow's own
    root span's full subtree (pre-pipe spend + pipe spend) from inside
    stage_a.
    """
    model_pre = FakeModel(["plain"] * 10)
    model_a = FakeModel(["A"] * 10)
    model_b = FakeModel(["B"] * 10)

    @agent(model=model_pre, name="nc4_budget_plain_agent")
    def nc4_budget_plain_agent(x: str) -> str:
        """Plain."""
        return f"p:{x}"

    @agent(model=model_a, name="nc4_budget_stage_a")
    def nc4_budget_stage_a(x: str) -> str:
        """A."""
        return f"a:{x}"

    @agent(model=model_b, name="nc4_budget_stage_b")
    def nc4_budget_stage_b(x: str) -> str:
        """B."""
        return f"b:{x}"

    the_pipe = pipe(nc4_budget_stage_a, nc4_budget_stage_b)

    @flow
    async def nc4_budget_flow(x: str) -> dict[str, str]:
        agent_result = (await nc4_budget_plain_agent.arun(x)).output
        pipe_result = (await the_pipe.arun(x)).output
        return {"agent_result": agent_result, "pipe_result": pipe_result}

    with pytest.raises(BudgetExceededError) as excinfo:
        nc4_budget_flow.run("in", budget=Budget(tokens=45))

    assert "used=60" in str(excinfo.value)
    assert len(model_pre.requests) == 1
    assert len(model_a.requests) == 1
    assert len(model_b.requests) == 0


def test_nested_pipe_arun_usage_rolls_up_async():
    """Async twin of ``test_nested_pipe_usage_rolls_up`` (finding I1): a
    flow calls a plain agent (30 tok) THEN a pipe (60 tok total) both via
    ``.arun()``. ``run.usage`` must total 90 tok and ``run.trace`` must
    contain the full pipe subtree.

    Before this fix, ``await the_pipe.arun(x)``'s nested call built its
    spans against a completely separate, freshly-minted ``Trace``, so the
    flow's own in-memory ``Trace``/``Run.usage`` never saw the pipe's 60 tok
    at all (``run.usage`` == 30 -- empirically confirmed alive by the final
    review's probe_arun_in_async_flow.py). This passes once ``arun``
    ctx-checks and adopts: the nested path never calls
    ``tracing.use_trace()`` -- it just opens a ``pipe`` span nested under
    whatever span/trace is already ambient -- so there is only ever one
    ``Trace`` object for the whole flow run.
    """
    model_a = FakeModel(["A"])
    model_b = FakeModel(["B"])
    model_agent = FakeModel(["plain"])

    @agent(model=model_a, name="nc4_usage_stage_a")
    def nc4_usage_stage_a(x: str) -> str:
        """A."""
        return f"a:{x}"

    @agent(model=model_b, name="nc4_usage_stage_b")
    def nc4_usage_stage_b(x: str) -> str:
        """B."""
        return f"b:{x}"

    @agent(model=model_agent, name="nc4_usage_plain_agent")
    def nc4_usage_plain_agent(x: str) -> str:
        """Plain."""
        return f"p:{x}"

    the_pipe = pipe(nc4_usage_stage_a, nc4_usage_stage_b)

    @flow
    async def nc4_usage_flow(x: str) -> dict[str, str]:
        pipe_result = (await the_pipe.arun(x)).output
        agent_result = (await nc4_usage_plain_agent.arun(x)).output
        return {"pipe_result": pipe_result, "agent_result": agent_result}

    run = nc4_usage_flow.run("in")

    assert run.usage.input_tokens + run.usage.output_tokens == 90
    trace_totals = run.trace.total_usage()
    assert trace_totals.input_tokens + trace_totals.output_tokens == 90

    kinds = [s.kind for s in run.trace.spans]
    assert kinds.count("llm") == 3
    pipe_span = next(s for s in run.trace.spans if s.kind == "pipe")
    flow_root = next(s for s in run.trace.spans if s.kind == "flow")
    assert pipe_span.trace_id == flow_root.trace_id == run.trace.trace_id
    assert pipe_span.parent_span_id == flow_root.span_id
