"""Internal-API tests for the async agent engine (Task 7, v0.4.0 Plan A).

``composeai.agentfn``'s loop internals were converted to async
(``_arun_agent_uncached``/``_aperform_turn``/``_acall_llm``/``_ainvoke_model``/
``_aprocess_tool_use``/``_aexecute_one_tool``), with the existing sync names
(``_run_agent_uncached`` in particular) becoming facades over
``composeai._runtime.run_sync``. These tests exercise a few internal-API
properties the reviewer can't get from black-box behavior alone; the REAL
acceptance is the untouched pre-existing suite (THE GATE) passing unchanged
now that every ``@agent`` call is routed through the async engine.
"""

from __future__ import annotations

import inspect
import os
import signal
import threading
import time

import pytest

from composeai import agentfn, runs
from composeai import combinators as _combinators
from composeai.agentfn import agent
from composeai.flow import flow, resume, task
from composeai.hitl import approve
from composeai.messages import Message, StopReason, Usage
from composeai.models.base import ModelRequest, ModelResponse
from composeai.testing import FakeModel
from composeai.tools import tool


def test_arun_agent_uncached_is_coroutine_function():
    """`_run_agent_uncached` is a sync facade; the actual engine it drives
    via `_runtime.run_sync` is a real coroutine function."""
    assert inspect.iscoroutinefunction(agentfn._arun_agent_uncached)
    assert not inspect.iscoroutinefunction(agentfn._run_agent_uncached)


def test_async_engine_drives_fake_model_agent():
    """Smoke test: `.run(...)` (the sync facade chain, ending in
    `_runtime.run_sync(_arun_agent_uncached(...))`) still returns the right
    output for a plain, single-turn `FakeModel` agent. Trivially true if the
    rest of the suite passes -- kept as a minimal, isolated reproduction.
    """
    model = FakeModel(["Hello there."])

    @agent(model=model, max_turns=3, name="async_smoke_greeter")
    def greeter(name: str) -> str:
        """You are a friendly greeter."""
        return f"Greet {name}"

    run = greeter.run("Ann")
    assert run.output == "Hello there."
    assert run.status == "completed"


def test_nested_sync_agent_call_from_tool_body_no_deadlock():
    """A sync ``@tool`` body that itself drives another ``FakeModel`` agent
    synchronously must not deadlock the runtime loop.

    The outer agent's turn runs as a Task on the runtime loop; its tool call
    is dispatched off-loop via ``_dispatch.run_stage``, onto its own
    dedicated worker thread (``_dispatch._run_sync_on_own_thread`` --
    deliberately NOT ``asyncio.to_thread``, whose shared, bounded default
    executor could starve this exact nesting instead); the tool body's own
    ``inner_agent(...)`` call reaches the sync facade ``_run_agent_uncached``
    -- and so ``_runtime.run_sync`` -- from THAT dedicated worker thread, not
    the runtime thread itself. Per the conversion contract's rule 7, this is
    legal: the runtime loop is merely *awaiting* the outer tool call's
    dedicated-worker-thread future (not itself blocked running code), so it
    is free to pick up and run the inner agent's fresh submission
    concurrently. If this ever regressed into an actual deadlock (e.g.
    `run_sync` blocking the loop instead of hopping back onto it), this call
    would hang forever -- bounded here by a SIGALRM so that regression shows
    up as a fast, readable test failure instead of an opaque hang.
    """
    inner_model = FakeModel(["inner result"])

    @agent(model=inner_model, max_turns=2, name="nested_inner_agent")
    def inner_agent(topic: str) -> str:
        """Inner agent, called synchronously from a tool body."""
        return f"Inner: {topic}"

    @tool
    def call_inner() -> str:
        """Call the inner agent synchronously from inside a tool body."""
        return inner_agent("nested")

    outer_model = FakeModel(
        [
            {"tool_calls": [{"name": "call_inner", "arguments": {}}]},
            "All done.",
        ]
    )

    @agent(model=outer_model, tools=[call_inner], max_turns=5, name="nested_outer_agent")
    def outer_agent(topic: str) -> str:
        """Outer agent."""
        return f"Outer: {topic}"

    def _on_alarm(signum: int, frame: object) -> None:
        raise TimeoutError(
            "nested sync agent-call-from-tool-body did not complete within 30s -- "
            "likely a runtime-loop deadlock regression"
        )

    has_alarm = hasattr(signal, "SIGALRM")
    previous_handler = None
    if has_alarm:
        previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(30)
    try:
        run = outer_agent.run("x")
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    assert run.output == "All done."
    assert run.status == "completed"


def test_model_without_acomplete_still_works():
    """A minimal, sync-only ``Model`` (no ``acomplete``/``stream``/``astream``,
    just ``complete()``) must still run correctly through the async engine --
    exercising the discovery rule's fallback path
    (``await asyncio.to_thread(model.complete, request)`` in
    ``_ainvoke_model``), not just ``FakeModel``'s native async methods.
    """

    class SyncOnlyModel:
        """Duck-types ``composeai.models.base.Model`` -- `complete()` only."""

        def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                message=Message.assistant("sync-only response"),
                stop_reason=StopReason.END_TURN,
                raw_stop_reason="end_turn",
                usage=Usage(),
                model_id=request.model,
            )

    assert getattr(SyncOnlyModel, "acomplete", None) is None
    assert getattr(SyncOnlyModel, "astream", None) is None
    assert getattr(SyncOnlyModel, "stream", None) is None

    @agent(model=SyncOnlyModel(), max_turns=2, name="sync_only_model_agent")
    def asker(question: str) -> str:
        """Ask something."""
        return question

    run = asker.run("hello?")
    assert run.output == "sync-only response"
    assert run.status == "completed"


# --- Task 8: combinators engine -> async core, sync facades -------------------
#
# composeai.combinators's Pipeline/Aggregate/map internals were converted to
# async (`_ainvoke_stage`, `Pipeline._arun_stages`, `Aggregate._arun_branches`,
# `_amap`), with the existing sync names (`Pipeline._run_stages`,
# `Aggregate._run_branches`, `map`) becoming facades over
# `composeai._runtime.run_sync`. As with Task 7, the REAL acceptance is the
# untouched pre-existing suite (THE GATE) -- these tests exercise a couple of
# internal-API properties (the async cores really are coroutine functions) and
# the one genuinely NEW capability an async engine enables: an `async def`
# plain callable as a map()/aggregate() stage, driven entirely through the
# existing sync public API.
#
# Map-in-flow journaling parity (item 3 of the task's internal-test list):
# `tests/test_flow.py::test_map_in_flow_journal_keys_match_input_order_not_completion_order`
# already drives `compose.map` over a real `@task` inside a `@flow`, asserts
# the per-item journal keys ("map#1[i]/process#1", input order not completion
# order), and resumes to confirm zero re-execution -- it exercises the public
# `compose.map`/`resume()` surface, which is now unconditionally backed by
# this task's new async engine (there is no separate old code path left to
# fall back to). Re-driving the same scenario here would be a duplicate of
# that test, not a different one, so it's intentionally not repeated -- this
# module's own async-native-stage tests below cover what that test does not
# (an `async def` stage, which `test_map_in_flow_...` does not use).

def test_ainvoke_stage_and_async_cores_are_coroutine_functions():
    """The new async twins are real coroutine functions; the sync names they
    back are not."""
    assert inspect.iscoroutinefunction(_combinators._ainvoke_stage)
    assert inspect.iscoroutinefunction(_combinators._amap)
    assert not inspect.iscoroutinefunction(_combinators.map)

    from composeai.combinators import Aggregate, Pipeline

    assert inspect.iscoroutinefunction(Pipeline._arun_stages)
    assert not inspect.iscoroutinefunction(Pipeline._run_stages)
    assert inspect.iscoroutinefunction(Aggregate._arun_branches)
    assert not inspect.iscoroutinefunction(Aggregate._run_branches)


def test_map_async_def_stage_preserves_order():
    """NEW capability (internal-only for now -- public exposure is Plan B):
    an ``async def`` plain callable as a ``map()`` stage runs natively on
    the composeai runtime loop (no thread hop -- `_dispatch.run_stage`'s
    coroutine-function branch) instead of `asyncio.to_thread`. Driven
    entirely through the existing sync `compose.map()` facade; item order
    is preserved in the output regardless of each item's own await point.
    """
    import asyncio as _asyncio

    from composeai import map as compose_map

    async def double(x: int) -> int:
        # A real await point (not just an async def with no suspension) --
        # items with a longer sleep must still land at their own index.
        await _asyncio.sleep(0.05 if x == 1 else 0.0)
        return x * 2

    assert compose_map(double, [1, 2, 3, 4, 5]) == [2, 4, 6, 8, 10]


def test_aggregate_async_def_branch_stage():
    """An ``async def`` branch stage in ``aggregate()`` runs correctly
    alongside a plain sync branch, driven through the sync `agg(x)`/
    `.run(x)` facade."""
    import asyncio as _asyncio

    from composeai import aggregate

    async def branch_a(x: int) -> int:
        await _asyncio.sleep(0)
        return x + 1

    def branch_b(x: int) -> int:
        return x * 2

    agg = aggregate(a=branch_a, b=branch_b)
    assert agg(5) == {"a": 6, "b": 10}


# --- Critical fix: shared-executor starvation deadlock -----------------------
#
# `_dispatch.run_stage`'s sync branches used to hop onto the loop's default
# `asyncio.to_thread` executor (`min(32, cpu+4)` threads shared process-wide).
# A sync stage that itself calls a sync facade (`compose.map`, `agent.run`,
# ...) blocked its executor slot while the nested work needed more slots of
# that SAME shared pool -- at fan-out >= pool size, every slot was pinned by
# an outer call waiting on an inner call that could never get scheduled: a
# permanent, silent deadlock. The fix (`_dispatch._run_sync_on_own_thread`)
# gives every in-flight sync stage its own dedicated daemon thread instead,
# so nesting can never starve a shared bound that no longer exists.


def test_nested_sync_fanout_does_not_starve():
    """A sync `map()` stage whose own body calls a nested sync `map()` must
    not deadlock even when the outer fan-out exceeds the event loop's default
    `asyncio.to_thread` executor size (`min(32, cpu+4)`).

    Mirrors the reviewer's `starve.py` repro shape: before the thread-per-
    stage fix, this hung forever (every outer worker pinned a slot in the
    shared default executor, starving the nested `map()` calls of a slot to
    run on). Bounded by a SIGALRM so a regression back into that deadlock
    shows up as a fast, readable failure instead of a hang.
    """
    from composeai import map as compose_map

    def inner(x: int) -> int:
        return x + 1

    def outer(x: int) -> int:
        # Sync stage nesting a sync facade call -- the shape that starved
        # the old shared `asyncio.to_thread` executor.
        return compose_map(inner, [x])[0]

    fan_out = min(32, (os.cpu_count() or 8) + 4) + 8  # exceed default executor size

    def _on_alarm(signum: int, frame: object) -> None:
        raise TimeoutError(
            "nested sync map() fan-out did not complete within 30s -- "
            "likely a shared-executor starvation regression"
        )

    has_alarm = hasattr(signal, "SIGALRM")
    previous_handler = None
    if has_alarm:
        previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(30)
    try:
        out = compose_map(outer, list(range(fan_out)))
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    assert out == [x + 1 for x in range(fan_out)]


def test_map_max_workers_zero_raises():
    """`map(..., max_workers=0)` must raise immediately (matching the old
    `ThreadPoolExecutor(max_workers=0)`'s eager `ValueError`), not hang --
    `max_workers=0` used to mean "an `asyncio.Semaphore(0)`", which no task
    could ever acquire."""
    from composeai import map as compose_map

    def fn(x: int) -> int:
        return x

    with pytest.raises(ValueError):
        compose_map(fn, [1, 2, 3], max_workers=0)


# --- Task 9: flow engine + resume -> async core, sync facades ----------------
#
# composeai.flow's engine (`_execute_flow`) was converted to async
# (`_aexecute_flow`), with `Flow.run()`/`resume()` becoming facades over
# `composeai._runtime.run_sync(_aexecute_flow(...))` -- the last conversion
# task (Task 9) of v0.4.0 Plan A. The flow BODY itself stays a plain sync
# user function (dispatched onto its own dedicated thread via
# `_dispatch.run_stage`, same bridge Task 8 built), so every `@task`/
# `@agent`/`now()`/`random()`/`approve()` call made from inside it still
# reaches the existing sync facades unchanged -- `Task._run_journaled`/
# `_execute` and `_run_flow_journaled` (nested flow calls made directly, not
# via a stage combinator) are NOT converted in this plan and stay exactly as
# they were. As with Tasks 7/8, the REAL acceptance is the untouched
# pre-existing suite (THE GATE, including `test_resume_subprocess.py`'s
# two-process crash/resume durability test) passing unchanged; these tests
# exercise a few internal-API/coherence properties black-box behavior alone
# doesn't pin down.


def test_aexecute_flow_is_coroutine_function():
    """`_aexecute_flow` is a real coroutine function; `Flow.run`/`resume`
    (the sync facades that drive it via `_runtime.run_sync`) are not --
    mirrors Task 7/8's identical parity checks for the agent/combinators
    engines."""
    # Imported directly from the `composeai.flow` submodule (not via
    # `import composeai.flow as ...`/`composeai.flow`) -- `composeai/__init__.py`
    # re-exports the `flow` DECORATOR under the name `composeai.flow`,
    # shadowing the submodule attribute of the same name once the package is
    # fully imported; `from composeai.flow import ...` reaches the actual
    # submodule regardless (it's resolved via `sys.modules`, not attribute
    # lookup on the `composeai` package object).
    from composeai.flow import Flow, Task, _aexecute_flow, _run_flow_journaled

    assert inspect.iscoroutinefunction(_aexecute_flow)
    assert not inspect.iscoroutinefunction(Flow.run)
    assert not inspect.iscoroutinefunction(resume)
    # Task._run_journaled/_execute and _run_flow_journaled are explicitly
    # NOT converted in this plan (they run on the flow body's own thread,
    # where the existing sync path already works) -- pinned here so a
    # future change can't silently convert them without this test noticing.
    assert not inspect.iscoroutinefunction(Task._run_journaled)
    assert not inspect.iscoroutinefunction(Task._execute)
    assert not inspect.iscoroutinefunction(_run_flow_journaled)


def test_flow_facade_smoke_through_async_engine():
    """Plain smoke test: a `@flow` calling a `@task` still works end to end
    now that `Flow.run()` dispatches through `_runtime.run_sync(_aexecute_flow(...))`
    instead of calling a sync engine directly."""

    @task
    def add_one(x: int) -> int:
        return x + 1

    @flow(name="t9_add_one_flow")
    def add_one_flow(x: int) -> int:
        return add_one(x)

    run = add_one_flow.run(4)
    assert run.output == 5
    assert run.status == "completed"


def test_flow_pause_resume_via_async_facade():
    """A `@flow` pausing on `approve()` and resuming, driven entirely
    through the Task 9 facade chain (`_runtime.run_sync(_aexecute_flow(...))`
    on both the initial run and the resume). `tests/test_hitl_flow.py`
    covers this round trip in much greater depth; kept minimal here as this
    task's own smoke test for the pause-propagates-through-`run_stage`
    contract (rule 1: a `composeai.hitl._Pause` raised in the flow body is a
    `BaseException` that must propagate through `_dispatch.run_stage`'s
    to-own-thread bridge verbatim, resolved on the awaiting engine side).
    """

    @flow(name="t9_gate_flow")
    def gate_flow() -> str:
        if approve("go"):
            return "went"
        return "stopped"

    run = gate_flow.run()
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.id == "go"

    run2 = resume(run.id, {"go": True})
    assert run2.status == "completed"
    assert run2.output == "went"


def test_nested_flow_direct_call_journals_as_one_step_via_async_engine():
    """A direct nested ``@flow`` call (the callee invoked as a plain
    function from inside the caller's body, NOT as a stage of
    ``pipe``/``aggregate``/``map``) still journals as one step of the outer
    flow's own run and does not re-execute on resume, now that the outer
    flow's engine is `_aexecute_flow` (async) -- `_run_flow_journaled`
    itself is unconverted/sync (it runs on the outer flow BODY's own
    thread, same as before Task 9). The async-engine-facing counterpart of
    `tests/test_flow.py::test_nested_flow_call_journals_as_one_step_and_replays_on_resume`.
    """
    inner_calls = {"n": 0}

    @task
    def inner_task() -> int:
        inner_calls["n"] += 1
        return 7

    @flow(name="t9_inner_flow")
    def inner_flow() -> int:
        return inner_task()

    @flow(name="t9_outer_flow")
    def outer_flow() -> int:
        return inner_flow() + 1

    run = outer_flow.run()
    assert run.output == 8
    assert inner_calls["n"] == 1

    store = runs.open_default()
    journal = store.journal_all(run.id)
    # One journaled step for the whole nested flow call (keyed by its
    # registered name) -- its own nested @task call is journaled too, but
    # scoped underneath that one step's key, not as a sibling of the outer
    # flow's own steps.
    assert "t9_inner_flow#1" in journal
    assert "t9_inner_flow#1/inner_task#1" in journal

    run2 = resume(run.id)
    assert run2.output == 8
    assert inner_calls["n"] == 1  # replayed, not re-executed


def test_flow_object_as_map_stage_inside_outer_flow_journals_as_one_step():
    """Rule-4 coherence check (v0.4.0 Plan A, Task 9): a bare ``Flow``
    object used AS A STAGE of ``compose.map()``, nested inside an outer
    ``@flow``.

    ``Flow`` is not one of ``composeai.combinators._ainvoke_stage``'s
    special-cased stage kinds (``AgentFunction``/``Pipeline``/``Aggregate``)
    -- Task 8 left it on the generic plain-callable branch, which dispatches
    ``Flow.__call__`` (a plain sync method, not a coroutine function) onto
    its own dedicated thread via ``_dispatch.run_stage``/
    ``_run_sync_on_own_thread``. On that thread, ``Flow.__call__`` sees the
    ambient ``RunContext`` (propagated via ``contextvars.copy_context()``)
    and routes to ``_run_flow_journaled`` (sync, unchanged) exactly as a
    direct nested ``@flow(...)`` call would -- so the inner flow still
    journals as ONE step per map item, scoped under that item's own
    ``map#1[i]`` segment, and replays without re-executing on resume.
    """
    from composeai import map as compose_map

    inner_calls = {"n": 0}

    @task
    def inner_step(x: int) -> int:
        inner_calls["n"] += 1
        return x + 1

    @flow(name="t9_map_inner_flow")
    def inner_flow(x: int) -> int:
        return inner_step(x)

    @flow(name="t9_map_outer_flow")
    def outer_flow() -> list[int]:
        return compose_map(inner_flow, [1, 2, 3])

    run = outer_flow.run()
    assert run.output == [2, 3, 4]
    assert inner_calls["n"] == 3

    store = runs.open_default()
    journal = store.journal_all(run.id)
    # One journaled step per map item for the nested flow call itself, plus
    # one more for its own nested @task call, both scoped under that item's
    # "map#1[i]" segment -- input order, not completion order.
    assert set(journal.keys()) == {
        "map#1[0]/t9_map_inner_flow#1",
        "map#1[0]/t9_map_inner_flow#1/inner_step#1",
        "map#1[1]/t9_map_inner_flow#1",
        "map#1[1]/t9_map_inner_flow#1/inner_step#1",
        "map#1[2]/t9_map_inner_flow#1",
        "map#1[2]/t9_map_inner_flow#1/inner_step#1",
    }

    inner_calls["n"] = 0
    run2 = resume(run.id)
    assert run2.output == [2, 3, 4]
    assert inner_calls["n"] == 0  # every item replayed, none re-executed


def test_async_def_map_stage_inside_flow_journals_by_input_order():
    """Closes Task 8's noted gap: an ``async def`` plain callable used as a
    ``compose.map()`` stage, nested inside an active ``@flow`` body, whose
    own coroutine body calls a ``@task``.

    Exercises the full ambient-context propagation chain end to end now
    that BOTH the flow engine (Task 9) and the combinators engine (Task 8)
    are async: the flow's ``RunContext`` (installed by ``_aexecute_flow``)
    must still reach a per-item ``asyncio.Task`` spawned natively by
    ``gather_settled`` for the ``async def`` stage (no thread hop for it --
    see ``_dispatch.run_stage``'s coroutine-function branch) -- through the
    flow body's own dedicated thread (``_run_sync_on_own_thread``) and
    ``compose.map()``'s own ``_runtime.run_sync`` re-entry from that thread
    into a fresh task on the SAME runtime loop. Per-item journal keys must
    still reflect INPUT order (``map#1[i]/record#1``), not completion
    order, and a resume must replay every item without re-executing any of
    them -- both already guaranteed by Task 8 for combinators alone; this
    confirms the guarantee still holds with the flow engine itself also
    async.
    """
    import asyncio

    from composeai import map as compose_map

    execution_log: list[int] = []

    @task
    def record(item: int) -> int:
        execution_log.append(item)
        return item * 10

    async def double_then_record(item: int) -> int:
        # Item 0 has the longest await -- if completion order (wrongly)
        # drove journal-key assignment, its key would be assigned last.
        await asyncio.sleep(0.05 if item == 0 else 0.0)
        return record(item)

    @flow(name="t9_async_map_flow")
    def run_map() -> list[int]:
        return compose_map(double_then_record, [0, 1, 2, 3])

    run = run_map.run()
    assert run.output == [0, 10, 20, 30]
    assert sorted(execution_log) == [0, 1, 2, 3]

    store = runs.open_default()
    journal = store.journal_all(run.id)
    assert set(journal.keys()) == {
        "map#1[0]/record#1",
        "map#1[1]/record#1",
        "map#1[2]/record#1",
        "map#1[3]/record#1",
    }

    execution_log.clear()
    run2 = resume(run.id)
    assert run2.output == [0, 10, 20, 30]
    assert execution_log == []  # every item replayed, none re-executed


# --- Task 10: thread hygiene + facade overhead sanity -------------------------
#
# The whole point of the v0.4.0 Plan A migration was "one always-running
# async engine behind a synchronous facade" -- these two tests are the final
# sanity check that the facade doesn't leak threads and doesn't add
# meaningful per-call latency now that every sync entry point hops onto the
# runtime loop and (for a trace-root run) the store's writer thread.

_LONG_LIVED_THREAD_NAMES = frozenset({"composeai-runtime", "composeai-store"})


def _leaked_composeai_thread_names() -> set[str]:
    """``composeai``-prefixed thread names alive right now that are neither
    one of the process's long-lived singletons nor an MCP bridge thread.

    ``composeai-runtime`` (the one process-wide runtime loop thread) and
    ``composeai-store`` (a ``RunStore``'s dedicated writer thread -- see
    ``composeai._storeasync``) are expected to still be alive: both are
    lazily started once and never closed until interpreter exit (or, for a
    store, until ``runs.reset_default()`` orphans it -- see that function's
    docstring), so earlier tests in this session may well have already
    started one or more of each. Because every such thread shares the exact
    same literal name, checking the *set of names* (not a count, and not
    object identity) is what makes this robust regardless of suite
    ordering or how many stores/resets earlier tests performed.
    ``composeai-mcp-*`` (a per-server MCP bridge thread -- see
    ``composeai.mcp``) is allowed for the same reason, if an MCP test ran
    earlier in this session and left its (idle, harmless) bridge thread
    registered.

    Anything else -- most notably ``composeai-stage`` (the per-call sync
    bridge thread spawned by ``composeai._dispatch.run_stage`` for every
    sync ``@agent``/``@task``/``@tool`` call) -- must NOT still be alive by
    the time a caller's blocking call has already returned: that thread
    resolves its result and then exits with nothing left to do, so a
    leftover one means a per-call thread is pinning an OS resource forever
    instead of being reclaimed.
    """
    names = {t.name for t in threading.enumerate() if t.name.startswith("composeai-")}
    return {
        name
        for name in names
        if name not in _LONG_LIVED_THREAD_NAMES and not name.startswith("composeai-mcp-")
    }


def test_thread_hygiene_no_leaked_per_call_threads():
    """After driving an ``@agent``, a ``compose.map()``, and a ``@flow``
    (each over a plain sync callable, so each actually dispatches through
    ``_dispatch.run_stage``'s ``composeai-stage`` per-call bridge thread)
    through the sync facade, no per-call thread may still be alive --  only
    ``composeai-runtime``/``composeai-store``/``composeai-mcp-*`` may be
    (see :func:`_leaked_composeai_thread_names`).
    """
    from composeai import map as compose_map

    model = FakeModel(["hi there"])

    @agent(model=model, max_turns=3, name="t10_thread_hygiene_agent")
    def greeter(name: str) -> str:
        """You are a friendly greeter."""
        return f"Greet {name}"

    greeter.run("Ann")

    @task
    def double(x: int) -> int:
        return x * 2

    compose_map(double, [1, 2, 3])

    @flow(name="t10_thread_hygiene_flow")
    def body() -> int:
        return double(21)

    body.run()

    # `composeai-stage` threads resolve their result via
    # `loop.call_soon_threadsafe` a moment before the underlying OS thread
    # actually finishes executing and exits -- the awaiting call above can
    # therefore return a few microseconds before `threading.enumerate()`
    # stops reporting that thread. Poll briefly (generously bounded well
    # above any realistic thread-teardown time) instead of asserting
    # instantaneously, to avoid a rare, spurious flake.
    deadline = time.monotonic() + 2.0
    leaked: set[str] = _leaked_composeai_thread_names()
    while leaked and time.monotonic() < deadline:
        time.sleep(0.02)
        leaked = _leaked_composeai_thread_names()

    assert not leaked, f"leaked composeai thread name(s) still alive: {sorted(leaked)}"


def test_facade_overhead_is_small_for_fake_model_agent():
    """Facade overhead tripwire: a ``FakeModel`` agent call does no real
    network I/O, so almost all of its wall-clock time is pure composeai
    overhead -- one hop onto the runtime loop (``_runtime.run_sync``), the
    ``agent``/``llm`` spans, budget/usage bookkeeping, and (this call being
    a trace-root run) one durable ``runs`` row write through the store's
    writer thread. 50 sequential calls (after one untimed warmup call, so
    one-time costs like the runtime thread's own startup don't skew the
    mean) -- asserts only a generous tripwire (< 5ms mean/call), not a tight
    performance budget; the measured number is printed so it can be tracked
    release over release (see the Task 10 report for the number this branch
    measured).
    """
    model = FakeModel(["ok"] * 51)

    @agent(model=model, max_turns=3, name="t10_overhead_greeter")
    def greeter(name: str) -> str:
        """You are a friendly greeter."""
        return f"Greet {name}"

    greeter("warmup")  # untimed: pays one-time runtime/store startup costs

    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        greeter("x")
    elapsed = time.perf_counter() - start

    mean_ms = (elapsed / iterations) * 1000
    print(f"\nfacade overhead: {mean_ms:.4f} ms/call over {iterations} calls "
          f"({elapsed:.4f}s total)")
    assert mean_ms < 5.0
