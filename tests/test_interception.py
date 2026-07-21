from __future__ import annotations

import composeai
from composeai.interception import BeforeTool, ToolInterceptor
from composeai.messages import ToolResultPart


def test_before_tool_defaults_and_frozen():
    from pydantic import ValidationError

    bt = BeforeTool()
    assert bt.action == "proceed"
    assert bt.arguments is None
    assert bt.message is None

    deny = BeforeTool(action="deny", message="nope")
    assert deny.action == "deny" and deny.message == "nope"

    import pytest

    with pytest.raises(ValidationError):
        bt.action = "deny"  # frozen


def test_exported_top_level():
    assert composeai.BeforeTool is BeforeTool
    assert composeai.ToolInterceptor is ToolInterceptor
    assert "BeforeTool" in composeai.__all__
    assert "ToolInterceptor" in composeai.__all__


def test_run_accepts_tool_interceptor_kwarg_noop_when_none():
    from composeai.agentfn import agent
    from composeai.testing import FakeModel

    @agent(model=FakeModel(["done"]), max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    # None interceptor is accepted everywhere and changes nothing.
    run = runner.run(tool_interceptor=None)
    assert run.status == "completed" and run.output == "done"


def test_flow_rejects_tool_interceptor():
    import pytest

    from composeai.errors import ConfigError
    from composeai.flow import flow, resume

    @flow
    def f() -> str:
        return "x"

    r = f.run()
    with pytest.raises(ConfigError):
        resume(r.id, {}, tool_interceptor=object())  # type: ignore[arg-type]


class _Interceptor:
    def __init__(self, before=None, after=None):
        self._before, self._after = before, after
        self.before_calls: list[str] = []
        self.after_calls: list[str] = []

    def before(self, call):
        self.before_calls.append(call.name)
        return self._before(call) if self._before else None

    def after(self, call, result):
        self.after_calls.append(call.name)
        return self._after(call, result) if self._after else None


def _gated_runner():
    from composeai.agentfn import agent
    from composeai.testing import FakeModel
    from composeai.tools import tool

    executed = {"n": 0, "args": None}

    @tool(requires_approval=True)
    def dangerous(x: int = 1) -> str:
        """Needs approval."""
        executed["n"] += 1
        executed["args"] = x
        return f"did it x={x}"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "dangerous", "arguments": {"x": 1}, "id": "call_1"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[dangerous], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    return runner, model, executed


def test_before_deny_blocks_before_approval():
    runner, model, executed = _gated_runner()
    approver_calls = {"n": 0}

    def approver(interrupt):
        approver_calls["n"] += 1
        return True

    icept = _Interceptor(before=lambda call: BeforeTool(action="deny", message="hook says no"))
    run = runner.run(approver=approver, tool_interceptor=icept)
    assert run.status == "completed"
    assert executed["n"] == 0           # tool never ran
    assert approver_calls["n"] == 0     # approver never consulted (deny before approval)
    denial = model.requests[1].messages[-1].parts[0]
    assert isinstance(denial, ToolResultPart) and denial.is_error is True
    assert denial.content == "hook says no"


def test_before_modify_args_reaches_tool():
    runner, model, executed = _gated_runner()
    icept = _Interceptor(before=lambda call: BeforeTool(arguments={"x": 42}))
    run = runner.run(approver=lambda i: True, tool_interceptor=icept)
    assert run.status == "completed"
    assert executed["n"] == 1 and executed["args"] == 42   # modified args reached the tool


def test_after_modifies_result():
    runner, model, executed = _gated_runner()

    def after(call, result):
        return ToolResultPart(tool_call_id=result.tool_call_id, content="REWRITTEN")

    icept = _Interceptor(after=after)
    run = runner.run(approver=lambda i: True, tool_interceptor=icept)
    assert run.status == "completed"
    result_part = model.requests[1].messages[-1].parts[0]
    assert isinstance(result_part, ToolResultPart) and result_part.content == "REWRITTEN"
    assert icept.after_calls == ["dangerous"]


def test_interceptor_fires_for_ungated_tools():
    from composeai.agentfn import agent
    from composeai.testing import FakeModel
    from composeai.tools import tool

    @tool
    def readonly() -> str:
        """Safe."""
        return "read"

    model = FakeModel(
        [{"tool_calls": [{"name": "readonly", "arguments": {}, "id": "c1"}]}, "final"]
    )

    @agent(model=model, tools=[readonly], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    icept = _Interceptor()
    runner.run(tool_interceptor=icept)
    assert icept.before_calls == ["readonly"]   # fires for ungated tools too
    assert icept.after_calls == ["readonly"]


def test_after_does_not_refire_on_resume():
    # A gated tool executes pre-pause via inline approve; on resume of a LATER
    # pending call, `after` must not re-fire for the already-executed tool.
    # (Mirror tests/test_hitl_agent.py's partial-results-not-re-executed shape:
    # test_partial_results_from_completed_calls_are_not_reexecuted_after_resume,
    # tests/test_hitl_agent.py:147-184.)
    from composeai.agentfn import agent
    from composeai.flow import resume
    from composeai.testing import FakeModel
    from composeai.tools import tool

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

    icept = _Interceptor()
    run = runner.run(tool_interceptor=icept)
    assert run.status == "paused"
    assert counters["safe"] == 1
    assert icept.after_calls == ["safe_tool"]   # fired once for the live execution

    # tool_interceptor is not journaled -- the same instance must be re-supplied
    # on resume for it to fire for the newly-answered call.
    run2 = resume(
        run.id, {"tool:dangerous_tool:call_dangerous": True}, tool_interceptor=icept
    )
    assert run2.status == "completed"
    assert counters["safe"] == 1  # not re-executed
    # `after` did not re-fire for the already-executed tool across the resume.
    assert icept.after_calls.count("safe_tool") == 1
    assert icept.after_calls == ["safe_tool", "dangerous_tool"]


def test_cassette_version_unchanged():
    from composeai.testing import _CASSETTE_VERSION

    assert _CASSETTE_VERSION == 2
