"""Inline approver= for requires_approval tools (0.7.0)."""

import json

from composeai import runs
from composeai.agentfn import agent
from composeai.hitl import Interrupt
from composeai.messages import ToolResultPart
from composeai.testing import FakeModel
from composeai.tools import tool


def _gated_setup():
    executed = {"n": 0}

    @tool(requires_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        executed["n"] += 1
        return "did it"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "dangerous", "arguments": {}, "id": "call_1"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[dangerous], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    return runner, model, executed


def test_approver_true_executes_inline_without_pause():
    runner, model, executed = _gated_setup()
    seen: list[Interrupt] = []

    def approver(interrupt: Interrupt) -> bool:
        seen.append(interrupt)
        return True

    run = runner.run(approver=approver)
    assert run.status == "completed"
    assert run.output == "final"
    assert executed["n"] == 1
    assert seen[0].id == "tool:dangerous:call_1"
    assert seen[0].kind == "approval"
    assert seen[0].payload == {"tool": "dangerous", "arguments": {}}


def test_approver_false_denies_and_model_sees_denial():
    runner, model, executed = _gated_setup()

    run = runner.run(approver=lambda interrupt: False)
    assert run.status == "completed"
    assert run.output == "final"
    assert executed["n"] == 0

    denial = model.requests[1].messages[-1].parts[0]
    assert isinstance(denial, ToolResultPart)
    assert denial.is_error is True
    assert denial.content == "denied by user"


def test_no_approver_still_pauses():
    runner, model, executed = _gated_setup()
    run = runner.run()
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.id == "tool:dangerous:call_1"
    assert executed["n"] == 0


def test_approver_decision_is_journaled():
    runner, model, executed = _gated_setup()
    run = runner.run(approver=lambda interrupt: True)
    assert run.status == "completed"

    store = runs.open_default()
    raw = store.journal_get(run.id, "__interrupt__:tool:dangerous:call_1")
    assert raw is not None
    assert json.loads(raw) is True


def test_approver_not_consulted_for_ungated_tools():
    @tool
    def safe() -> str:
        """Safe."""
        return "ok"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "safe", "arguments": {}, "id": "call_1"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[safe], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    seen: list[Interrupt] = []
    run = runner.run(approver=lambda i: (seen.append(i), True)[1])
    assert run.status == "completed"
    assert seen == []
