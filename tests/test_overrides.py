"""Per-call system= / model= overrides (0.7.0)."""

import asyncio
import json

from composeai.agentfn import _decode_agent_call, _encode_agent_call, agent
from composeai.flow import resume
from composeai.testing import FakeModel
from composeai.tools import tool


def test_per_call_system_override_reaches_request():
    model = FakeModel(["ok"])

    @agent(model=model)
    def helper() -> str:
        """Default system."""
        return "hi"

    helper.run(system="Override system.")
    assert model.requests[0].system == "Override system."


def test_per_call_system_default_keeps_docstring():
    model = FakeModel(["ok"])

    @agent(model=model)
    def helper() -> str:
        """Default system."""
        return "hi"

    helper.run()
    assert model.requests[0].system == "Default system."


def test_per_call_model_override_routes_to_other_model():
    default_model = FakeModel(["default answer"])
    override_model = FakeModel(["override answer"])

    @agent(model=default_model)
    def helper() -> str:
        """H."""
        return "hi"

    run = helper.run(model=override_model)
    assert run.output == "override answer"
    assert default_model.requests == []
    assert len(override_model.requests) == 1


def test_override_does_not_mutate_agent_defaults():
    model = FakeModel(["one", "two"])

    @agent(model=model)
    def helper() -> str:
        """Default system."""
        return "hi"

    helper.run(system="Temporary.")
    helper.run()
    assert model.requests[1].system == "Default system."


def test_stream_accepts_system_override():
    model = FakeModel(["streamed"])

    @agent(model=model)
    def helper() -> str:
        """Default system."""
        return "hi"

    stream = helper.stream(system="Stream system.")
    for _ in stream:
        pass
    assert stream.run.output == "streamed"
    assert model.requests[0].system == "Stream system."


def test_arun_system_override_reaches_request():
    model = FakeModel(["ok"])

    @agent(model=model, name="ovr_arun_system")
    def helper() -> str:
        """Default system."""
        return "hi"

    async def drive():
        return await helper.arun(system="Override system.")

    asyncio.run(drive())
    assert model.requests[0].system == "Override system."


def test_arun_model_override_routes_to_other_model():
    default_model = FakeModel(["default answer"])
    override_model = FakeModel(["override answer"])

    @agent(model=default_model, name="ovr_arun_model")
    def helper() -> str:
        """H."""
        return "hi"

    async def drive():
        return await helper.arun(model=override_model)

    run = asyncio.run(drive())
    assert run.output == "override answer"
    assert default_model.requests == []
    assert len(override_model.requests) == 1


def test_astream_accepts_system_override():
    model = FakeModel(["streamed"])

    @agent(model=model, name="ovr_astream_system")
    def helper() -> str:
        """Default system."""
        return "hi"

    async def drive():
        stream = helper.astream(system="Stream system.")
        async for _ in stream:
            pass
        return await stream.run()

    run = asyncio.run(drive())
    assert run.output == "streamed"
    assert model.requests[0].system == "Stream system."


def test_encode_agent_call_persists_str_overrides_only():
    payload = _encode_agent_call((), {}, system="S.", model="anthropic/claude-x")
    decoded = json.loads(payload)
    assert decoded["system"] == "S."
    assert decoded["model"] == "anthropic/claude-x"

    fake = FakeModel(["x"])
    payload2 = _encode_agent_call((), {}, system=None, model=fake)
    decoded2 = json.loads(payload2)
    assert "system" not in decoded2
    assert "model" not in decoded2


def test_decode_agent_call_is_backward_compatible():
    legacy = json.dumps({"args": [], "kwargs": {}})
    args, kwargs, system, model = _decode_agent_call(legacy)
    assert (args, kwargs) == ((), {})
    assert system is None
    assert model is None


def test_system_override_survives_resume():
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
        """Decoration system."""
        return "go"

    run = runner.run(system="Override system.")
    assert run.status == "paused"
    assert model.requests[0].system == "Override system."

    run2 = resume(run.id, {"tool:dangerous:call_1": True})
    assert run2.status == "completed"
    assert run2.output == "final"
    assert model.requests[1].system == "Override system."
