"""Tests for human-in-the-loop pause/resume in ``@agent`` runs (Phase 8).

``@tool(requires_approval=True)`` pauses the agent loop before executing an
unanswered approval-gated call; ``ask_human`` inside a tool body pauses the
same way. Both standalone agent runs and agents nested in a ``@flow`` share
the same ``resume(run_id, answers)`` entry point and the same
``__interrupt__:`` journal convention.

Deliberately *not* using ``from __future__ import annotations`` (same
reason as ``test_agent.py``).
"""

import pytest

from composeai import runs
from composeai.agentfn import agent
from composeai.errors import ConfigError
from composeai.flow import _FLOW_REGISTRY, flow, resume
from composeai.hitl import Interrupt
from composeai.messages import ToolResultPart
from composeai.testing import FakeModel
from composeai.tools import tool


@pytest.fixture(autouse=True)
def _clear_flow_registry():
    yield
    _FLOW_REGISTRY.clear()


# --- approval-gated tool pauses a standalone agent -------------------------------


def test_approval_tool_pauses_standalone_agent_sibling_executed_once():
    counters = {"safe": 0}

    @tool
    def safe_tool() -> str:
        """A safe tool that doesn't need approval."""
        counters["safe"] += 1
        return "safe-done"

    @tool(requires_approval=True)
    def dangerous_tool() -> str:
        """A dangerous tool that needs approval."""
        return "dangerous-done"

    model = FakeModel(
        [
            {
                "tool_calls": [
                    {"name": "safe_tool", "arguments": {}, "id": "call_safe"},
                    {"name": "dangerous_tool", "arguments": {}, "id": "call_dangerous"},
                ]
            },
            "final",
        ]
    )

    @agent(model=model, tools=[safe_tool, dangerous_tool], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"
    assert isinstance(run.pending, Interrupt)
    assert run.pending.id == "tool:dangerous_tool:call_dangerous"
    assert run.pending.kind == "approval"
    assert counters["safe"] == 1

    store = runs.open_default()
    row = store.get_run(run.id)
    assert row is not None
    assert row["kind"] == "agent"
    assert row["status"] == "paused"
    # `run.id` of a paused standalone agent must equal its durable row id.
    assert run.id == row["run_id"]

    # No answer yet: resume must re-pause without re-running the safe sibling.
    run2 = resume(run.id)
    assert run2.status == "paused"
    assert counters["safe"] == 1


def test_approved_tool_executes_and_completes():
    executed = {"n": 0}

    @tool(requires_approval=True)
    def dangerous_tool() -> str:
        """Needs approval."""
        executed["n"] += 1
        return "did it"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "dangerous_tool", "arguments": {}, "id": "call_1"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[dangerous_tool], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"

    run2 = resume(run.id, {"tool:dangerous_tool:call_1": True})
    assert run2.status == "completed"
    assert run2.output == "final"
    assert executed["n"] == 1


def test_denied_tool_reaches_model_as_denied_error():
    @tool(requires_approval=True)
    def dangerous_tool() -> str:
        """Needs approval."""
        return "should not run"

    seen_messages = []

    def second_turn(request):
        seen_messages.append(request.messages[-1])
        return "final"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "dangerous_tool", "arguments": {}, "id": "call_1"}]},
            second_turn,
        ]
    )

    @agent(model=model, tools=[dangerous_tool], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"

    run2 = resume(run.id, {"tool:dangerous_tool:call_1": False})
    assert run2.status == "completed"
    assert run2.output == "final"

    tool_result_message = seen_messages[0]
    result_parts = [p for p in tool_result_message.parts if isinstance(p, ToolResultPart)]
    assert len(result_parts) == 1
    assert result_parts[0].is_error is True
    assert result_parts[0].content == "denied by user"


def test_partial_results_from_completed_calls_are_not_reexecuted_after_resume():
    counters = {"safe": 0}

    @tool
    def safe_tool() -> str:
        """Safe."""
        counters["safe"] += 1
        return "safe-done"

    @tool(requires_approval=True)
    def dangerous_tool() -> str:
        """Needs approval."""
        return "dangerous-done"

    model = FakeModel(
        [
            {
                "tool_calls": [
                    {"name": "safe_tool", "arguments": {}, "id": "call_safe"},
                    {"name": "dangerous_tool", "arguments": {}, "id": "call_dangerous"},
                ]
            },
            "final",
        ]
    )

    @agent(model=model, tools=[safe_tool, dangerous_tool], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"
    assert counters["safe"] == 1

    run2 = resume(run.id, {"tool:dangerous_tool:call_dangerous": True})
    assert run2.status == "completed"
    assert counters["safe"] == 1  # not re-executed


# --- ask_human inside a tool body -------------------------------------------------


def test_ask_human_inside_tool_body_reexecutes_tool_and_returns_answer():
    from composeai.hitl import ask_human

    calls = {"n": 0}

    @tool
    def needs_input() -> str:
        """Calls ask_human."""
        calls["n"] += 1
        name = ask_human("who", "who are you?")
        return f"hello {name}"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "needs_input", "arguments": {}, "id": "call_1"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[needs_input], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.kind == "question"
    assert run.pending.id == "who"
    assert calls["n"] == 1

    run2 = resume(run.id, {"who": "Ada"})
    assert run2.status == "completed"
    assert run2.output == "final"
    assert calls["n"] == 2  # re-executed from the top (documented contract)


# --- shorthand answer keys ---------------------------------------------------------


def test_shorthand_tool_name_answer_key_resolves_when_unambiguous():
    @tool(requires_approval=True)
    def send_email() -> str:
        """Needs approval."""
        return "sent"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "send_email", "arguments": {}, "id": "call_123"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[send_email], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"

    run2 = resume(run.id, {"send_email": True})
    assert run2.status == "completed"
    assert run2.output == "final"


def test_shorthand_tool_name_answer_key_ambiguous_raises_config_error():
    @tool(requires_approval=True)
    def send_email(to: str) -> str:
        """Needs approval.

        Args:
            to: recipient
        """
        return f"sent to {to}"

    model = FakeModel(
        [
            {
                "tool_calls": [
                    {"name": "send_email", "arguments": {"to": "a"}, "id": "call_a"},
                    {"name": "send_email", "arguments": {"to": "b"}, "id": "call_b"},
                ]
            },
            "final",
        ]
    )

    @agent(model=model, tools=[send_email], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"

    store = runs.open_default()
    pending_ids = {row["interrupt_id"] for row in store.pending_interrupts_all(run.id)}
    assert pending_ids == {"tool:send_email:call_a", "tool:send_email:call_b"}

    with pytest.raises(ConfigError):
        resume(run.id, {"send_email": True})

    # Full ids still work, unambiguously.
    run2 = resume(run.id, {"tool:send_email:call_a": True, "tool:send_email:call_b": True})
    assert run2.status == "completed"


def test_shorthand_tool_name_answer_key_no_match_raises_config_error():
    """Regression: `resume(run_id, {"send_email": True})` while the run is
    paused on a DIFFERENT tool used to silently journal the answer verbatim
    under `__interrupt__:send_email` -- a key that could never be looked up
    (real approval interrupts are always the full `tool:{name}:{call_id}`
    form), so the answer was permanently useless with no error and no
    warning. It now raises ConfigError listing what's actually pending."""

    @tool(requires_approval=True)
    def dangerous_tool() -> str:
        """Needs approval."""
        return "danger-done"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "dangerous_tool", "arguments": {}, "id": "call_1"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[dangerous_tool], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"

    with pytest.raises(ConfigError, match="send_email"):
        resume(run.id, {"send_email": True})

    # The real interrupt must still be answerable normally afterward.
    run2 = resume(run.id, {"tool:dangerous_tool:call_1": True})
    assert run2.status == "completed"


# --- agent nested in a flow --------------------------------------------------------


def test_agent_nested_in_flow_pauses_flow_and_resume_completes_via_nested_state():
    @tool(requires_approval=True)
    def dangerous_tool() -> str:
        """Needs approval."""
        return "done"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "dangerous_tool", "arguments": {}, "id": "call_1"}]},
            "agent-final",
        ]
    )

    @agent(model=model, tools=[dangerous_tool], max_turns=5)
    def inner_agent() -> str:
        """Inner agent."""
        return "go"

    @flow
    def outer_flow() -> str:
        result = inner_agent()
        return f"flow-got: {result}"

    run = outer_flow.run()
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.id == "tool:dangerous_tool:call_1"

    store = runs.open_default()
    row = store.get_run(run.id)
    assert row is not None
    assert row["kind"] == "flow"
    assert row["status"] == "paused"

    run2 = resume(run.id, {"tool:dangerous_tool:call_1": True})
    assert run2.status == "completed"
    assert run2.output == "flow-got: agent-final"


# --- snapshot rows written per turn -------------------------------------------------


def test_agent_state_snapshot_row_written_after_tool_batch():
    @tool
    def step_tool() -> str:
        """A plain step tool."""
        return "ok"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "step_tool", "arguments": {}, "id": "call_1"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[step_tool], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "completed"

    store = runs.open_default()
    row = store.agent_state_get(run.id, "")
    assert row is not None
    assert row["turn"] == 1
    assert row["messages_json"]


# --- agent registry: duplicate names -------------------------------------------------


def test_duplicate_agent_name_raises_config_error():
    model = FakeModel(["hi"])

    @agent(model=model)
    def dupe_agent_name(x: str) -> str:
        """One."""
        return x

    with pytest.raises(ConfigError):

        @agent(model=model)
        def dupe_agent_name(x: str) -> str:  # noqa: F811
            """Two."""
            return x


# --- capstone fix wave A regressions -------------------------------------------


def test_pause_persists_snapshot_and_pending_interrupts_in_one_atomic_call(monkeypatch):
    """Regression: `_pause_for_unanswered_approval` used to persist its
    interrupt (and mark the run "paused") immediately -- one commit per
    unanswered call, well before the turn's `agent_state` snapshot was
    written afterward, once, at the end. A crash in that window orphaned
    the interrupt with no snapshot to safely resume it against. Now every
    write for one pause goes through a single `RunStore.persist_pause`
    call; the old two-step `persist_pending_interrupt` path is never used
    for this case."""
    calls: list[str] = []
    original_persist_pause = runs.RunStore.persist_pause
    original_persist_pending = runs.persist_pending_interrupt

    def spy_persist_pause(self, **kwargs):
        calls.append("persist_pause")
        return original_persist_pause(self, **kwargs)

    def spy_persist_pending(store, run_id, interrupt):
        calls.append("persist_pending_interrupt")
        return original_persist_pending(store, run_id, interrupt)

    monkeypatch.setattr(runs.RunStore, "persist_pause", spy_persist_pause)
    monkeypatch.setattr(runs, "persist_pending_interrupt", spy_persist_pending)

    @tool(requires_approval=True)
    def gate_1() -> str:
        """Needs approval."""
        return "1-done"

    @tool(requires_approval=True)
    def gate_2() -> str:
        """Needs approval."""
        return "2-done"

    model = FakeModel(
        [
            {
                "tool_calls": [
                    {"name": "gate_1", "arguments": {}, "id": "call_1"},
                    {"name": "gate_2", "arguments": {}, "id": "call_2"},
                ]
            },
            "final",
        ]
    )

    @agent(model=model, tools=[gate_1, gate_2], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"

    # Both interrupts (gate_1 AND gate_2, both unanswered) are still visible
    # via the pending_interrupts table, all written by the ONE persist_pause
    # call for this batch.
    store = runs.open_default()
    pending = {p["interrupt_id"] for p in store.pending_interrupts_all(run.id)}
    assert pending == {"tool:gate_1:call_1", "tool:gate_2:call_2"}

    # `persist_pause` (inside `_process_tool_use`, atomically writing both
    # interrupts + the snapshot + status) fires first; the one, single
    # `persist_pending_interrupt` call afterward is `settle_agent_run`'s own
    # *outer* top-level pause handling (idempotent -- re-writing the row for
    # the one _Pause that propagates all the way up) -- a separate,
    # legitimate mechanism, not the redundant per-call persistence this
    # fix removed (which would have shown up as TWO
    # `persist_pending_interrupt` calls, one per unanswered gate, before
    # `persist_pause` ever ran).
    assert calls == ["persist_pause", "persist_pending_interrupt"]


def test_approval_batch_keeps_scanning_after_an_approved_calls_own_body_pauses():
    """Regression: the `except _Pause as exc: ... break` inside the
    approval-calls scanning loop used to abandon the whole batch the moment
    an *already-approved* call's own body paused (e.g. a nested
    `ask_human()`) -- silently skipping every subsequent call in the batch,
    even independent ones with their own already-supplied answer. The fix
    changes that `break` to `continue`, matching the sibling "unanswered
    gate" branch two lines above and this function's own documented "keep
    scanning the rest" contract.
    """
    from composeai.hitl import ask_human

    executed = {"c": 0}

    @tool(requires_approval=True)
    def tool_a() -> str:
        """Needs approval; its own body pauses via ask_human()."""
        return ask_human("inner_ask", "proceed?")

    @tool(requires_approval=True)
    def tool_c() -> str:
        """Needs approval; independent of tool_a, and comes after it in the batch."""
        executed["c"] += 1
        return "c-done"

    model = FakeModel(
        [
            {
                "tool_calls": [
                    {"name": "tool_a", "arguments": {}, "id": "call_a"},
                    {"name": "tool_c", "arguments": {}, "id": "call_c"},
                ]
            },
            "final",
        ]
    )

    @agent(model=model, tools=[tool_a, tool_c], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.id == "tool:tool_a:call_a"

    # Answer BOTH gates: tool_a's own approval gate, and tool_c's. tool_a's
    # body itself then pauses again (ask_human) -- but tool_c, listed after
    # it, must still be examined and actually executed *this same attempt*.
    run2 = resume(run.id, {"tool_a": True, "tool_c": True})
    assert run2.status == "paused"
    assert run2.pending is not None
    assert run2.pending.id == "inner_ask"
    assert executed["c"] == 1


def test_parallel_tool_calls_get_deterministic_journal_scope_by_call_id():
    """Regression: nested @task calls inside concurrently-dispatched tool
    bodies used to race `ctx.next_key()` directly -- completion order, not
    the model's tool-call order, decided the keys. Each tool call now runs
    under its own `tool:{call_id}` journal scope."""
    import threading
    import time

    from composeai.flow import flow, task

    execution_log: list[str] = []
    lock = threading.Lock()

    @task
    def record(item: tuple) -> str:
        duration, cid = item
        time.sleep(duration)
        with lock:
            execution_log.append(cid)
        return cid

    @tool
    def tool_x() -> str:
        """Slow tool X."""
        return record((0.05, "x"))

    @tool
    def tool_y() -> str:
        """Fast tool Y."""
        return record((0.0, "y"))

    model = FakeModel(
        [
            {
                "tool_calls": [
                    {"name": "tool_x", "arguments": {}, "id": "call_x"},
                    {"name": "tool_y", "arguments": {}, "id": "call_y"},
                ]
            },
            "final",
        ]
    )

    @agent(model=model, tools=[tool_x, tool_y], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    @flow
    def outer() -> str:
        return runner()

    run = outer.run()
    assert run.output == "final"
    # y (no sleep) finishes before x (50ms sleep) -- proving real, out-of-
    # order concurrency, not just sequential dispatch.
    assert execution_log == ["y", "x"]

    store = runs.open_default()
    journal = store.journal_all(run.id)
    record_keys = {k for k in journal if "record" in k}
    assert record_keys == {"tool:call_x/record#1", "tool:call_y/record#1"}

    # Resume (nothing left to do) must replay both, never re-execute.
    execution_log.clear()
    run2 = resume(run.id)
    assert run2.output == "final"
    assert execution_log == []
