"""Inline approver= for requires_approval tools (0.7.0)."""

import json

import pytest

from composeai import runs
from composeai.agentfn import agent
from composeai.hitl import ApprovalReply, Interrupt
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


def test_approval_reply_construct_defaults_frozen_and_exported():
    from pydantic import ValidationError

    import composeai
    from composeai import ApprovalReply as TopLevelApprovalReply
    from composeai.hitl import ApprovalReply

    # Same object exported top-level and from hitl (mirrors Interrupt).
    assert TopLevelApprovalReply is ApprovalReply
    assert "ApprovalReply" in composeai.__all__

    # Construction + defaults: message is optional and defaults to None.
    reply = ApprovalReply(allow=False)
    assert reply.allow is False
    assert reply.message is None

    with_msg = ApprovalReply(allow=False, message="do X instead")
    assert with_msg.allow is False
    assert with_msg.message == "do X instead"

    # Frozen (mirrors Interrupt) -- field assignment raises.
    with pytest.raises(ValidationError):
        reply.allow = True


def test_approver_reply_message_reaches_model():
    runner, model, executed = _gated_setup()

    run = runner.run(
        approver=lambda interrupt: ApprovalReply(allow=False, message="do X instead")
    )
    assert run.status == "completed"
    assert run.output == "final"
    assert executed["n"] == 0

    denial = model.requests[1].messages[-1].parts[0]
    assert isinstance(denial, ToolResultPart)
    assert denial.is_error is True
    assert denial.content == "do X instead"


def test_approver_reply_allow_true_executes_inline():
    runner, model, executed = _gated_setup()
    run = runner.run(approver=lambda interrupt: ApprovalReply(allow=True))
    assert run.status == "completed"
    assert run.output == "final"
    assert executed["n"] == 1


def test_approver_reply_deny_without_message_is_default_denial():
    runner, model, executed = _gated_setup()
    run = runner.run(approver=lambda interrupt: ApprovalReply(allow=False))
    assert run.status == "completed"
    assert executed["n"] == 0

    denial = model.requests[1].messages[-1].parts[0]
    assert isinstance(denial, ToolResultPart)
    assert denial.content == "denied by user"


def test_deny_tool_call_uses_message_when_provided_else_default():
    from composeai.agentfn import _deny_tool_call
    from composeai.messages import ToolCallPart, ToolResultPart

    call = ToolCallPart(id="call_1", name="dangerous", arguments={})

    # Default (no message / explicit None) -> byte-identical "denied by user".
    default = _deny_tool_call(call)
    assert isinstance(default, ToolResultPart)
    assert default.is_error is True
    assert default.tool_call_id == "call_1"
    assert default.content == "denied by user"
    assert _deny_tool_call(call, None).content == "denied by user"

    # A supplied message becomes the ToolResultPart content.
    with_msg = _deny_tool_call(call, "do X instead")
    assert with_msg.is_error is True
    assert with_msg.tool_call_id == "call_1"
    assert with_msg.content == "do X instead"
