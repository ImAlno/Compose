"""Public async surface (v0.4.0 Plan B): arun/astream and friends."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest
from pydantic import BaseModel

from composeai import agent, amap, prompt
from composeai.combinators import aggregate, pipe
from composeai.testing import FakeModel


def test_arun_returns_run_on_callers_loop():
    model = FakeModel(["hello"])

    @agent(model=model, name="asurf_basic")
    def basic(q: str) -> str:
        """Answer."""
        return prompt(q)

    async def drive():
        run = await basic.arun("hi")
        return run

    run = asyncio.run(drive())
    assert run.status == "completed"
    assert run.output == "hello"


def test_arun_never_touches_runtime_thread(monkeypatch):
    """arun must run the engine on the caller's loop — run_sync unreachable."""
    from composeai import _runtime

    def _forbidden(coro):
        coro.close()
        raise AssertionError("run_sync called from the async surface")

    monkeypatch.setattr(_runtime, "run_sync", _forbidden)
    model = FakeModel(["ok"])

    @agent(model=model, name="asurf_noloop")
    def noloop(q: str) -> str:
        """Answer."""
        return prompt(q)

    assert asyncio.run(noloop.arun("x")).output == "ok"


def test_astream_yields_events_and_run():
    model = FakeModel(["streamed answer"])

    @agent(model=model, name="asurf_stream")
    def streamer(q: str) -> str:
        """Answer."""
        return prompt(q)

    async def drive():
        stream = streamer.astream("go")
        kinds = []
        async for event in stream:
            kinds.append(event.kind)
        run = await stream.run()
        return kinds, run

    kinds, run = asyncio.run(drive())
    assert "text_delta" in kinds
    assert kinds[-1] == "run_finished"
    assert run.output == "streamed answer"


def test_arun_creates_durable_row_like_run():
    from composeai.runs import open_default

    model = FakeModel(["persisted"])

    @agent(model=model, name="asurf_durable")
    def durable(q: str) -> str:
        """Answer."""
        return prompt(q)

    run = asyncio.run(durable.arun("x"))
    row = open_default().get_run(run.id)
    assert row is not None
    assert row["kind"] == "agent" and row["status"] == "completed"


def test_concurrent_aruns_on_one_loop():
    model_a = FakeModel(["alpha"])
    model_b = FakeModel(["beta"])

    @agent(model=model_a, name="asurf_conc_a")
    def agent_a(q: str) -> str:
        """Answer."""
        return prompt(q)

    @agent(model=model_b, name="asurf_conc_b")
    def agent_b(q: str) -> str:
        """Answer."""
        return prompt(q)

    async def drive():
        run_a, run_b = await asyncio.gather(agent_a.arun("1"), agent_b.arun("2"))
        return run_a.output, run_b.output

    assert asyncio.run(drive()) == ("alpha", "beta")


# --- Task 5: combinators async surface (amap, Pipeline/Aggregate arun/astream) --


def test_amap_native_async_stages():
    """An ``async def`` stage is awaited natively (see ``_dispatch.run_stage``'s
    ``iscoroutinefunction`` branch) and input order is preserved."""
    calls: list[int] = []

    async def stage(x: int) -> int:
        calls.append(x)
        await asyncio.sleep(0)
        return x * 2

    async def drive():
        return await amap(stage, [1, 2, 3])

    result = asyncio.run(drive())
    assert result == [2, 4, 6]
    assert calls == [1, 2, 3]


def test_amap_on_error_collect_parity():
    """The same flaky fn through sync ``map`` and ``amap`` produces identical
    ``MapResult`` fields (on_error='collect')."""
    import composeai as compose

    def flaky(x: int) -> int:
        if x == 1:
            raise ValueError("boom")
        return x * 2

    sync_results = compose.map(flaky, [0, 1, 2], on_error="collect")

    async def drive():
        return await amap(flaky, [0, 1, 2], on_error="collect")

    async_results = asyncio.run(drive())

    def _fields(results):
        return [(r.ok, r.value, r.error, r.error_type) for r in results]

    assert _fields(sync_results) == _fields(async_results)


def test_pipeline_arun():
    @agent(model=FakeModel(["step1 out"]), name="asurf_pipe_arun_a")
    def a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["step2 out"]), name="asurf_pipe_arun_b")
    def b(x: str) -> str:
        """B."""
        return x

    pipeline = pipe(a, b)

    async def drive():
        return await pipeline.arun("go")

    run = asyncio.run(drive())

    @agent(model=FakeModel(["step1 out"]), name="asurf_pipe_arun_a2")
    def a2(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["step2 out"]), name="asurf_pipe_arun_b2")
    def b2(x: str) -> str:
        """B."""
        return x

    sync_run = pipe(a2, b2).run("go")

    assert run.status == "completed"
    assert run.output == sync_run.output == "step2 out"


def test_aggregate_arun_timeout_per_branch():
    from composeai.errors import TaskTimeoutError

    def fast(x: int) -> int:
        return x + 1

    def hung(x: int) -> int:
        time.sleep(10)
        return x

    agg = aggregate(timeout_per_branch=0.2, fast=fast, hung=hung)

    async def drive():
        return await agg.arun(1)

    start = time.monotonic()
    with pytest.raises(TaskTimeoutError):
        asyncio.run(drive())
    assert time.monotonic() - start < 2.0


def test_pipe_astream_events():
    @agent(model=FakeModel(["hello from a"]), name="asurf_pipe_astream_a")
    def a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["hello from b"]), name="asurf_pipe_astream_b")
    def b(x: str) -> str:
        """B."""
        return x

    pipeline = pipe(a, b)

    async def drive():
        stream = pipeline.astream("go")
        kinds = []
        async for event in stream:
            kinds.append(event.kind)
        run = await stream.run()
        return kinds, run

    kinds, run = asyncio.run(drive())
    assert kinds[-1] == "run_finished"
    assert run.output == "hello from b"


def test_amap_exported():
    from composeai import amap as amap_top_level

    assert amap_top_level is amap


# --- Task 6: async bodies -- @tool, @agent prompt-builder, @task ------------


def test_async_agent_body_both_surfaces():
    """An ``async def`` @agent body works through BOTH the sync facade
    (``agent_fn(...)``) and the async facade (``await agent_fn.arun(...)``)
    -- the engine underneath is always async (see
    ``agentfn._abuild_conversation``'s docstring), so both facades exercise
    the identical code path for a coroutine body."""

    async def _abody(q: str):
        await asyncio.sleep(0)
        return prompt(f"Q: {q}")

    @agent(model=FakeModel(["sync answer"]), name="asurf_async_body_sync")
    async def researcher_sync(q: str) -> str:
        """You are a researcher."""
        return await _abody(q)

    assert researcher_sync("x") == "sync answer"

    @agent(model=FakeModel(["async answer"]), name="asurf_async_body_async")
    async def researcher_async(q: str) -> str:
        """You are a researcher."""
        return await _abody(q)

    async def drive():
        return await researcher_async.arun("x")

    run = asyncio.run(drive())
    assert run.status == "completed"
    assert run.output == "async answer"


def test_async_task_fn_from_sync_flow_body():
    """An ``async def`` @task called from a plain SYNC @flow body: journals
    like any other task, and does not re-execute on resume (replay)."""
    from composeai.flow import flow, resume, task
    from composeai.hitl import approve

    calls: list[int] = []

    @task(name="asurf_async_task_sync_flow")
    async def afetch(x: int) -> int:
        calls.append(x)
        await asyncio.sleep(0)
        return x * 2

    @flow(name="asurf_async_task_flow")
    def async_task_flow(x: int) -> int:
        first = afetch(x)
        if not approve("continue"):
            return -1
        second = afetch(first)
        return second

    run = async_task_flow.run(5)
    assert run.status == "paused"
    assert calls == [5]

    resumed = resume(run.id, {"continue": True})
    assert resumed.status == "completed"
    assert resumed.output == 20
    # afetch(5) replayed from the journal on resume -- its body did not run again
    assert calls == [5, 10]


def test_task_arun_journaled_via_manual_run_context(tmp_path):
    """``Task.arun``'s journaled path, exercised directly against a manual
    :class:`~composeai.runs.RunContext` pushed under ``asyncio.run`` --
    mirrors ``tests/test_dispatch.py``'s zombie-guard test setup. A real
    async ``@flow`` body driving this is T7's job (async flow bodies aren't
    implemented yet); this covers ``Task.arun``'s own journal/replay
    contract in isolation in the meantime."""
    from composeai import runs
    from composeai.flow import task

    calls: list[int] = []

    @task(name="asurf_arun_journaled")
    async def afetch(x: int) -> int:
        calls.append(x)
        await asyncio.sleep(0)
        return x * 2

    store = runs.RunStore(tmp_path / "runs.db")

    async def first_attempt():
        ctx = runs.RunContext(run_id="r-arun-1", store=store)
        with runs.use_run_context(ctx):
            return await afetch.arun(5)

    result1 = asyncio.run(first_attempt())
    assert result1 == 10
    assert calls == [5]

    async def resumed_attempt():
        preloaded = store.journal_all("r-arun-1")
        ctx = runs.RunContext(run_id="r-arun-1", store=store, preloaded=preloaded)
        with runs.use_run_context(ctx):
            return await afetch.arun(5)

    result2 = asyncio.run(resumed_attempt())
    assert result2 == 10
    assert calls == [5]  # replayed -- afetch's body did not run a second time


# --- Task 7: Flow.arun, aresume, async @flow bodies, anow/arandom -----------


def test_flow_arun_smoke():
    """An ``async def`` @flow body, driven through ``Flow.arun`` on the
    caller's own loop: awaits ``Task.arun``, ``agent.arun``, and ``anow()``,
    completes, and journals under the expected keys."""
    from composeai.flow import anow, flow, task
    from composeai.runs import open_default

    model = FakeModel(["async flow answer"])

    @task(name="asurf_arun_smoke_task")
    def double(x: int) -> int:
        return x * 2

    @agent(model=model, name="asurf_arun_smoke_agent")
    def responder(q: str) -> str:
        """Answer."""
        return prompt(q)

    @flow(name="asurf_arun_smoke_flow")
    async def async_flow(x: int) -> str:
        doubled = await double.arun(x)
        agent_run = await responder.arun(f"double is {doubled}")
        await anow()
        return agent_run.output

    async def drive():
        return await async_flow.arun(5)

    run = asyncio.run(drive())
    assert run.status == "completed"
    assert run.output == "async flow answer"

    journal = open_default().journal_all(run.id)
    assert "asurf_arun_smoke_task#1" in journal
    assert "asurf_arun_smoke_agent#1" in journal
    assert "now#1" in journal


def test_async_flow_pause_and_aresume():
    """An ``async def`` @flow body calling sync ``approve()`` pauses;
    ``await aresume(...)`` with the answer completes -- and ``anow()``'s
    value REPLAYS identically across the paused and resumed attempts."""
    from composeai.flow import anow, aresume, flow
    from composeai.hitl import approve

    seen: list = []

    @flow(name="asurf_async_pause_flow")
    async def async_pause_flow() -> str:
        seen.append(await anow())
        if not approve("gate"):
            return "stopped"
        return "done"

    run = asyncio.run(async_pause_flow.arun())
    assert run.status == "paused"
    first_now = seen[0]

    resumed = asyncio.run(aresume(run.id, {"gate": True}))
    assert resumed.status == "completed"
    assert resumed.output == "done"
    # the resumed attempt re-executed the flow body: anow() must REPLAY
    assert seen[1] == first_now


def test_aresume_budget_override():
    """``Flow.arun`` dies on ``BudgetExceededError``; ``aresume`` with a
    raised budget completes -- mirrors
    ``test_resume_with_budget_override_allows_recovery``'s sync semantics
    with a fresh async-bodied flow/agent pair."""
    import json as _json

    from composeai.errors import BudgetExceededError
    from composeai.flow import aresume, flow
    from composeai.messages import Usage
    from composeai.runs import Budget, open_default

    model = FakeModel(
        ["first", "second", "second-retry"],
        usage=Usage(input_tokens=5, output_tokens=5),
    )

    @agent(model=model, name="asurf_aresume_budget_agent")
    def spender(prompt: str) -> str:
        return prompt

    @flow(name="asurf_aresume_budget_flow")
    async def capped_flow() -> str:
        await spender.arun("one")
        await spender.arun("two")
        return "done"

    async def drive_run():
        return await capped_flow.arun(budget=Budget(tokens=15))

    with pytest.raises(BudgetExceededError):
        asyncio.run(drive_run())

    store = open_default()
    row = store.list_runs(kind="flow", limit=1)[0]
    run_id = row["run_id"]

    # A budget-killed run is recoverable with a raised cap; completed steps
    # replay free, and the override counts prior real spend.
    run = asyncio.run(aresume(run_id, budget=Budget(tokens=60)))
    assert run.status == "completed"
    assert run.output == "done"

    # The override persisted -- a future resume sees the new budget.
    stored = store.get_run(run_id)
    assert stored is not None
    assert _json.loads(stored["budget_json"]) == {"usd": None, "tokens": 60}


def test_anow_key_compat_with_now():
    """A sync ``@flow`` body using ``now()`` and an async ``@flow`` body
    using ``anow()`` journal under the SAME key name (``now#1``) -- so a
    flow body converted sync<->async replays the other's recorded value."""
    from composeai.flow import anow, flow, now
    from composeai.runs import open_default

    @flow(name="asurf_now_key_sync_flow")
    def sync_now_flow() -> str:
        now()
        return "sync-done"

    sync_run = sync_now_flow.run()
    assert sync_run.status == "completed"

    @flow(name="asurf_now_key_async_flow")
    async def async_now_flow() -> str:
        await anow()
        return "async-done"

    async def drive():
        return await async_now_flow.arun()

    async_run = asyncio.run(drive())
    assert async_run.status == "completed"

    store = open_default()
    assert "now#1" in store.journal_all(sync_run.id)
    assert "now#1" in store.journal_all(async_run.id)


def test_async_exports():
    from composeai import anow, arandom, aresume

    assert callable(anow)
    assert callable(arandom)
    assert callable(aresume)


# --- Task 8: async integration suites + nested-flow ctx-routing for arun ----


class _AsurfRepairPoint(BaseModel):
    # Module-level, not local to the test function: under this file's
    # `from __future__ import annotations`, a model class local to a test
    # function leaves pydantic unable to rebuild its own core schema (the
    # same reason test_agent.py's `_Task6Point` -- this test's sync sibling
    # -- is module-level too, and that whole file deliberately skips the
    # future-annotations import).
    x: int
    y: int


def test_arun_structured_output_with_repair():
    """``max_repairs`` via ``arun`` -- async twin of
    ``test_repair_turn_recovers_from_invalid_output`` (test_agent.py)."""

    model = FakeModel(
        [
            {"json": {"x": "bad", "y": 2}},
            {"json": {"x": 1, "y": 2}},
        ]
    )

    @agent(model=model, max_repairs=1, name="asurf_t8_repair_agent")
    def repairer(question: str) -> _AsurfRepairPoint:
        """Answer."""
        return prompt(question)

    async def drive():
        return await repairer.arun("go")

    run = asyncio.run(drive())
    assert run.status == "completed"
    assert run.output == _AsurfRepairPoint(x=1, y=2)
    # the second request carried the validation error back to the model
    assert len(model.requests) == 2
    assert "did not match the required output schema" in model.requests[1].messages[-1].text


def test_arun_budget_cumulative_across_aresume():
    """Async twin of ``test_budget_accumulates_across_resume_attempts``
    (test_flow.py), ported onto ``Flow.arun``/``aresume``: budget
    enforcement must stay cumulative across an async pause/resume -- the
    resumed attempt's spend adds to the paused attempt's, not a fresh
    window."""
    from composeai.errors import BudgetExceededError
    from composeai.flow import aresume, flow
    from composeai.hitl import approve
    from composeai.messages import Usage
    from composeai.runs import Budget

    model = FakeModel(["first", "second"], usage=Usage(input_tokens=5, output_tokens=5))

    @agent(model=model, name="asurf_t8_budget_agent")
    def spender(prompt: str) -> str:
        return prompt

    @flow(name="asurf_t8_budget_flow")
    async def two_step() -> str:
        await spender.arun("one")  # 10 tokens -- under the 15 cap
        if not approve("continue"):
            return "stopped"
        await spender.arun("two")  # 10 more -- lifetime total 20 > 15
        return "done"

    async def drive_run():
        return await two_step.arun(budget=Budget(tokens=15))

    run = asyncio.run(drive_run())
    assert run.status == "paused"

    # Before Task 7's cumulative-budget fix, a resumed attempt started a
    # fresh in-memory trace, so its own 10 tokens stayed under the cap and
    # the run completed -- every aresume silently granted a fresh window.
    async def drive_resume():
        return await aresume(run.id, {"continue": True})

    with pytest.raises(BudgetExceededError):
        asyncio.run(drive_resume())


def test_arun_hitl_tool_approval_pause_aresume():
    """Async twin of
    ``test_agent_nested_in_flow_pauses_flow_and_resume_completes_via_nested_state``
    (test_hitl_agent.py): a ``requires_approval`` local ``@tool`` inside an
    ``@agent`` inside an async ``@flow``, driven through ``arun``; pause;
    ``aresume`` with a bare tool-name answer key (shorthand resolution --
    see ``composeai.runs.resolve_answer_key``)."""
    from composeai.flow import aresume, flow
    from composeai.tools import tool

    @tool(requires_approval=True)
    def asurf_t8_dangerous_tool() -> str:
        """Needs approval."""
        return "done"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "asurf_t8_dangerous_tool", "arguments": {}, "id": "call_1"}]},
            "agent-final",
        ]
    )

    @agent(
        model=model,
        tools=[asurf_t8_dangerous_tool],
        max_turns=5,
        name="asurf_t8_hitl_inner_agent",
    )
    def inner_agent() -> str:
        """Inner agent."""
        return "go"

    @flow(name="asurf_t8_hitl_outer_flow")
    async def outer_flow() -> str:
        agent_run = await inner_agent.arun()
        return f"flow-got: {agent_run.output}"

    run = asyncio.run(outer_flow.arun())
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.id == "tool:asurf_t8_dangerous_tool:call_1"

    async def drive_resume():
        return await aresume(run.id, {"asurf_t8_dangerous_tool": True})

    resumed = asyncio.run(drive_resume())
    assert resumed.status == "completed"
    assert resumed.output == "flow-got: agent-final"


def test_astream_pipe_event_parity():
    """``pipe.astream``'s event-kind sequence matches ``.stream()``'s, on
    fresh equivalent FakeModels for each side -- event ordering must not
    depend on which engine (background-thread sync facade vs. native async)
    drives the pipeline."""

    @agent(model=FakeModel(["sync-a"]), name="asurf_t8_parity_sync_a")
    def sync_a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["sync-b"]), name="asurf_t8_parity_sync_b")
    def sync_b(x: str) -> str:
        """B."""
        return x

    sync_stream = pipe(sync_a, sync_b).stream("go")
    sync_kinds = [e.kind for e in sync_stream]
    sync_run = sync_stream.run

    @agent(model=FakeModel(["sync-a"]), name="asurf_t8_parity_async_a")
    def async_a(x: str) -> str:
        """A."""
        return x

    @agent(model=FakeModel(["sync-b"]), name="asurf_t8_parity_async_b")
    def async_b(x: str) -> str:
        """B."""
        return x

    async_pipeline = pipe(async_a, async_b)

    async def drive():
        stream = async_pipeline.astream("go")
        kinds = [event.kind async for event in stream]
        run = await stream.run()
        return kinds, run

    async_kinds, async_run = asyncio.run(drive())

    assert async_kinds == sync_kinds
    assert async_run.output == sync_run.output == "sync-b"


def test_nested_flow_arun_journals_as_step():
    """Async twin of test_flow.py's
    ``test_nested_flow_call_journals_as_one_step_and_replays_on_resume``:
    an async outer ``@flow`` body awaits ``inner.arun(...)`` -- must
    journal as ONE step of the outer run (never mint a second durable run
    row for the inner flow), and replay -- not re-execute -- the inner
    body's on resume."""
    from composeai.flow import aresume, flow
    from composeai.runs import open_default

    inner_calls = {"n": 0}

    @flow(name="asurf_t8_nested_inner")
    async def inner_flow() -> int:
        inner_calls["n"] += 1
        return 5

    should_fail = {"value": True}

    @flow(name="asurf_t8_nested_outer")
    async def outer_flow() -> int:
        inner_run = await inner_flow.arun()
        if should_fail["value"]:
            raise RuntimeError("boom")
        return inner_run.output + 100

    async def drive_first():
        return await outer_flow.arun()

    with pytest.raises(RuntimeError):
        asyncio.run(drive_first())
    assert inner_calls["n"] == 1

    store = open_default()
    outer_rows = [
        r for r in store.list_runs(kind="flow", limit=50) if r["name"] == "asurf_t8_nested_outer"
    ]
    assert len(outer_rows) == 1  # ONE runs-table row for the outer run
    run_row = outer_rows[0]
    assert run_row["status"] == "failed"

    inner_rows = [
        r for r in store.list_runs(kind="flow", limit=50) if r["name"] == "asurf_t8_nested_inner"
    ]
    assert inner_rows == []  # the nested arun() call never minted its own durable row

    should_fail["value"] = False

    async def drive_resume():
        return await aresume(run_row["run_id"])

    resumed = asyncio.run(drive_resume())
    assert resumed.output == 105
    # inner_flow must NOT re-execute on resume -- it replays from a single
    # journaled step, exactly like the sync nested-flow-call test.
    assert inner_calls["n"] == 1

    journal = store.journal_all(run_row["run_id"])
    assert "asurf_t8_nested_inner#1" in journal


def test_full_surface_under_one_asyncio_run():
    """The 'user loop' acceptance test (Task 8): one ``asyncio.run`` driving
    a ``gather`` of two agent ``arun``s, an ``amap``, and a flow ``arun``
    that pauses + ``aresume`` completes it -- all on the SAME running loop,
    never touching the composeai runtime thread."""
    from composeai.flow import aresume, flow
    from composeai.hitl import approve

    model_a = FakeModel(["alpha-full"])
    model_b = FakeModel(["beta-full"])

    @agent(model=model_a, name="asurf_t8_full_agent_a")
    def agent_a(q: str) -> str:
        """Answer."""
        return prompt(q)

    @agent(model=model_b, name="asurf_t8_full_agent_b")
    def agent_b(q: str) -> str:
        """Answer."""
        return prompt(q)

    async def double(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    @flow(name="asurf_t8_full_flow")
    async def gated_flow() -> str:
        if not approve("go"):
            return "stopped"
        return "flow-done"

    async def drive():
        run_a, run_b, mapped, flow_run = await asyncio.gather(
            agent_a.arun("1"), agent_b.arun("2"), amap(double, [1, 2, 3]), gated_flow.arun()
        )
        assert flow_run.status == "paused"
        resumed = await aresume(flow_run.id, {"go": True})
        return run_a.output, run_b.output, mapped, resumed.output

    out_a, out_b, mapped, flow_output = asyncio.run(drive())
    assert out_a == "alpha-full"
    assert out_b == "beta-full"
    assert mapped == [2, 4, 6]
    assert flow_output == "flow-done"


# --- Task 9: thread census + perf ----------------------------------------


_LONG_LIVED_THREAD_NAMES = frozenset({"composeai-runtime", "composeai-store"})


def _leaked_composeai_thread_names() -> set[str]:
    """Async-surface twin of ``test_engine_async``'s helper of the same
    name -- see that module's docstring for the full rationale
    (``composeai-runtime``/``composeai-store`` are process-wide singletons
    started lazily and never closed; ``composeai-mcp-*`` bridge threads are
    likewise long-lived per server, not per-call). Duplicated locally
    rather than imported -- it's private, file-scoped test tooling, not a
    shared public helper.
    """
    names = {t.name for t in threading.enumerate() if t.name.startswith("composeai-")}
    return {
        name
        for name in names
        if name not in _LONG_LIVED_THREAD_NAMES and not name.startswith("composeai-mcp-")
    }


def test_thread_census_after_mixed_sync_and_async_use():
    """Mirrors ``test_engine_async``'s Plan A Task 10 sync-only hygiene
    test, but interleaves the sync and async facades in one process -- an
    ``@agent`` ``.run()`` AND ``.arun()``, a sync ``map()`` AND async
    ``amap()``, a sync ``@flow`` ``.run()`` AND async ``@flow`` ``.arun()``
    -- proving the two facades coexist without leaking any per-call thread
    beyond the long-lived runtime/store/mcp singletons (see
    :func:`_leaked_composeai_thread_names`)."""
    from composeai import map as compose_map
    from composeai.flow import flow

    model_sync = FakeModel(["sync-hi"])
    model_async = FakeModel(["async-hi"])

    @agent(model=model_sync, name="asurf_t9_census_sync_agent")
    def sync_greeter(name: str) -> str:
        """Greet."""
        return f"Greet {name}"

    @agent(model=model_async, name="asurf_t9_census_async_agent")
    def async_greeter(name: str) -> str:
        """Greet."""
        return f"Greet {name}"

    sync_greeter.run("Ann")
    asyncio.run(async_greeter.arun("Bo"))

    def double(x: int) -> int:
        return x * 2

    compose_map(double, [1, 2, 3])
    asyncio.run(amap(double, [1, 2, 3]))

    @flow(name="asurf_t9_census_sync_flow")
    def sync_body() -> int:
        return double(21)

    @flow(name="asurf_t9_census_async_flow")
    async def async_body() -> int:
        return double(21)

    sync_body.run()
    asyncio.run(async_body.arun())

    # `composeai-stage` threads resolve their result a moment before the
    # underlying OS thread actually exits (see test_engine_async's own
    # hygiene test) -- poll briefly rather than asserting instantaneously.
    deadline = time.monotonic() + 2.0
    leaked = _leaked_composeai_thread_names()
    while leaked and time.monotonic() < deadline:
        time.sleep(0.02)
        leaked = _leaked_composeai_thread_names()

    assert not leaked, f"leaked composeai thread name(s) still alive: {sorted(leaked)}"


def test_async_surface_never_starts_runtime_thread_in_subprocess(tmp_path):
    """A fresh process that touches ONLY the async surface -- one
    ``@agent`` ``arun``, one ``amap``, one ``@flow`` ``arun``, all driven by
    a single ``asyncio.run`` -- must never lazily start the
    ``composeai-runtime`` thread: the whole point of the async surface (see
    :func:`test_arun_never_touches_runtime_thread`, which proves this
    in-process by making ``_runtime.run_sync`` raise if called) is that it
    never falls back to the sync facade's runtime-loop bridge. A real
    subprocess is the only way to prove the thread was never started at
    all -- an in-process assertion can't rule out some EARLIER test in the
    same session having already started that process-wide singleton.
    ``composeai-store`` MAY be present: a trace-root ``arun``'s durable row
    write goes through the store's own dedicated writer thread (never the
    runtime loop) -- see ``composeai.runs.arun_standalone_agent``.
    """
    code = textwrap.dedent(
        """
        import asyncio
        import threading

        from composeai import agent, amap, prompt
        from composeai.flow import flow
        from composeai.testing import FakeModel

        model = FakeModel(['census-ok'])

        @agent(model=model, name='asurf_t9_subproc_agent')
        def greeter(name: str) -> str:
            return prompt(f'Greet {name}')

        async def double(x: int) -> int:
            return x * 2

        @flow(name='asurf_t9_subproc_flow')
        async def body() -> int:
            return await double(21)

        async def main():
            run = await greeter.arun('Ann')
            mapped = await amap(double, [1, 2, 3])
            frun = await body.arun()
            return run.output, mapped, frun.output

        result = asyncio.run(main())
        assert result == ('census-ok', [2, 4, 6], 42), result

        names = sorted(t.name for t in threading.enumerate())
        print('THREADS:' + ','.join(names))
        """
    )
    env = dict(os.environ, COMPOSE_DIR=str(tmp_path / "compose-store"))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=env
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    thread_lines = [line for line in result.stdout.splitlines() if line.startswith("THREADS:")]
    assert thread_lines, f"subprocess never printed its thread census: {result.stdout!r}"
    census = thread_lines[0].removeprefix("THREADS:")
    names = census.split(",") if census else []

    assert "composeai-runtime" not in names, (
        f"async surface lazily started the runtime thread: {names}"
    )


def test_arun_facade_overhead_is_small_and_cheaper_than_sync():
    """Perf tripwire twin of ``test_engine_async``'s Plan A Task 10 sync
    overhead test: ``arun`` does one FEWER hop than the sync facade -- no
    ``_runtime.run_sync`` bridge onto the separate runtime thread's loop,
    just the caller's own already-running loop straight through (see
    :func:`test_arun_never_touches_runtime_thread`). Measures both facades
    here, same process, same FakeModel-driven single-turn agent shape, same
    50-iteration/1-warmup protocol as the sync measurement, so the two
    numbers are directly comparable -- only a generous <5ms tripwire is
    asserted (see the Task 9 report for both measured numbers; ``arun`` is
    expected, not required, to come in lower).
    """
    model_sync = FakeModel(["ok"] * 51)
    model_async = FakeModel(["ok"] * 51)

    @agent(model=model_sync, max_turns=3, name="t9_perf_sync_greeter")
    def sync_greeter(name: str) -> str:
        """You are a friendly greeter."""
        return f"Greet {name}"

    @agent(model=model_async, max_turns=3, name="t9_perf_async_greeter")
    def async_greeter(name: str) -> str:
        """You are a friendly greeter."""
        return f"Greet {name}"

    iterations = 50

    sync_greeter("warmup")  # untimed: pays one-time runtime/store startup costs
    sync_start = time.perf_counter()
    for _ in range(iterations):
        sync_greeter("x")
    sync_elapsed = time.perf_counter() - sync_start
    sync_mean_ms = (sync_elapsed / iterations) * 1000

    async def drive() -> float:
        await async_greeter.arun("warmup")  # untimed, same reasoning as the sync warmup
        start = time.perf_counter()
        for _ in range(iterations):
            await async_greeter.arun("x")
        return time.perf_counter() - start

    async_elapsed = asyncio.run(drive())
    async_mean_ms = (async_elapsed / iterations) * 1000

    print(
        f"\narun overhead: {async_mean_ms:.4f} ms/call vs sync facade "
        f"{sync_mean_ms:.4f} ms/call, over {iterations} calls each "
        f"(arun {async_elapsed:.4f}s total, sync {sync_elapsed:.4f}s total)"
    )
    assert async_mean_ms < 5.0


# --- v0.4.0 Plan B fix wave -------------------------------------------------


def test_nested_async_bodied_flow_via_sync_sugar_journals_and_replays():
    """A SYNC outer @flow body calling an async-bodied inner @flow through
    the sync sugar (``inner(...)``) -- not ``await inner.arun(...)`` -- must
    still journal as ONE step of the outer run and replay (not re-execute)
    the inner body on resume, exactly like a sync-bodied nested flow call
    (test_flow.py's
    ``test_nested_flow_call_journals_as_one_step_and_replays_on_resume``).

    Before the fix, ``_run_flow_journaled``'s miss path called
    ``flow_obj._fn(*args, **kwargs)`` directly with no
    ``iscoroutinefunction`` check -- for an ``async def`` inner body that
    returns an un-awaited coroutine object instead of the inner flow's
    actual result, which the journal then failed to serialize
    (``SerializationError``), alongside a "coroutine was never awaited"
    warning -- nowhere near the real bug.
    """
    from composeai.flow import flow, resume, task
    from composeai.runs import open_default

    inner_calls: list[int] = []

    @flow(name="asurf_fix1_inner_flow")
    async def inner_flow() -> int:
        await asyncio.sleep(0)
        inner_calls.append(1)
        return 5

    should_fail = {"value": True}

    @task(name="asurf_fix1_risky")
    def risky() -> int:
        if should_fail["value"]:
            raise RuntimeError("boom")
        return 100

    @flow(name="asurf_fix1_outer_flow")
    def outer_flow() -> int:
        a = inner_flow()  # sync sugar call of an async-bodied inner flow
        b = risky()
        return a + b

    with pytest.raises(RuntimeError):
        outer_flow.run()
    assert inner_calls == [1]

    store = open_default()
    rows = [
        r for r in store.list_runs(kind="flow", limit=50) if r["name"] == "asurf_fix1_outer_flow"
    ]
    run_row = rows[0]
    assert run_row["status"] == "failed"

    should_fail["value"] = False
    run2 = resume(run_row["run_id"])
    assert run2.output == 105
    # inner_flow must NOT re-execute on resume -- it replays from a single
    # journaled step, exactly like a sync-bodied nested flow call.
    assert inner_calls == [1]

    journal = store.journal_all(run_row["run_id"])
    assert "asurf_fix1_inner_flow#1" in journal


def test_nested_flow_arun_rejects_budget():
    """Nested ``Flow.arun(budget=...)`` (an ambient ``RunContext`` -- i.e.
    ``await inner.arun(budget=...)`` from inside another ``@flow`` body)
    must raise ``ConfigError`` rather than silently ignore the budget --
    the enclosing run's own budget governs a nested flow step (see
    ``_run_flow_journaled``'s docstring); before this fix ``budget`` simply
    had no effect at all on this path."""
    from composeai.errors import ConfigError
    from composeai.flow import flow
    from composeai.runs import Budget

    @flow(name="asurf_fix4_inner")
    async def inner_flow() -> int:
        return 1

    @flow(name="asurf_fix4_outer")
    async def outer_flow() -> int:
        await inner_flow.arun(budget=Budget(tokens=10))
        return 1

    with pytest.raises(ConfigError, match="budget"):
        asyncio.run(outer_flow.arun())
