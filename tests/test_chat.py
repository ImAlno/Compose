"""compose.chat sessions (0.7.0)."""

import pytest

from composeai import runs
from composeai.agentfn import agent
from composeai.chat import chat
from composeai.errors import ConfigError
from composeai.testing import FakeModel, reset_registries
from composeai.tools import tool


def _buddy(model):
    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        raise AssertionError("agent body must not run for chat sends")

    return buddy


def test_chat_send_accumulates_history():
    model = FakeModel(["first reply", "second reply"])
    c = chat(_buddy(model))

    r1 = c.send("hello")
    assert r1.status == "completed"
    assert r1.output == "first reply"
    assert [m.role for m in c.messages] == ["user", "assistant"]

    r2 = c.send("and again")
    assert r2.output == "second reply"
    seen = model.requests[1].messages
    assert [m.role for m in seen] == ["user", "assistant", "user"]
    assert seen[0].text == "hello"
    assert seen[1].text == "first reply"
    assert seen[2].text == "and again"
    assert [m.role for m in c.messages] == ["user", "assistant", "user", "assistant"]


def test_chat_send_bypasses_agent_body():
    model = FakeModel(["ok"])
    c = chat(_buddy(model))
    run = c.send("hi")  # would raise AssertionError if the body ran
    assert run.output == "ok"


def test_chat_uses_agent_system_and_accepts_override():
    model = FakeModel(["a", "b"])

    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    c.send("one")
    assert model.requests[0].system == "You are Buddy."

    c2 = chat(buddy, system="You are Grumpy.")
    c2.send("two")
    assert model.requests[1].system == "You are Grumpy."


def test_chat_persists_history_after_send():
    model = FakeModel(["persisted reply"])

    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    c.send("hello")

    store = runs.open_default()
    row = store.chat_get(c.id)
    assert row is not None
    assert row["agent_name"] == "buddy"
    assert "persisted reply" in row["messages_json"]


def test_chat_messages_property_returns_copy():
    model = FakeModel(["x"])

    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    c.send("hi")
    snapshot = c.messages
    snapshot.clear()
    assert len(c.messages) == 2


def test_load_chat_restores_history_and_continues():
    model = FakeModel(["first", "second"])

    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    c.send("hello")

    from composeai.chat import load_chat

    c2 = load_chat(c.id)
    assert c2.id == c.id
    assert [m.text for m in c2.messages] == ["hello", "first"]

    c2.send("more")
    assert model.requests[1].messages[0].text == "hello"
    assert len(model.requests[1].messages) == 3


def test_load_chat_unknown_id_raises():
    from composeai.chat import load_chat

    with pytest.raises(ConfigError, match="no chat found"):
        load_chat("nope")


def test_load_chat_unregistered_agent_raises():
    model = FakeModel(["x"])

    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    c.send("hi")
    chat_id = c.id

    reset_registries()

    from composeai.chat import load_chat

    with pytest.raises(ConfigError, match="not\\s+registered"):
        load_chat(chat_id)


def test_paused_send_exposes_pending_and_resume_folds_history():
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
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    run = c.send("do the thing")
    assert run.status == "paused"
    assert c.pending is not None
    assert c.pending.id == "tool:dangerous:call_1"
    assert c.messages == []  # history NOT advanced -- the user turn folds in only on completion

    run2 = c.resume({"tool:dangerous:call_1": True})
    assert run2.status == "completed"
    assert run2.output == "final"
    assert executed["n"] == 1
    assert c.pending is None
    assert c.messages[-1].text == "final"


def test_resume_after_completed_send_raises_and_preserves_history():
    model = FakeModel(["a reply"])

    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    c.send("hello")
    assert [m.role for m in c.messages] == ["user", "assistant"]

    with pytest.raises(ConfigError, match="nothing to resume"):
        c.resume()

    # History must NOT be clobbered by the guarded resume.
    assert [m.text for m in c.messages] == ["hello", "a reply"]
    store = runs.open_default()
    row = store.chat_get(c.id)
    assert row is not None
    assert "a reply" in row["messages_json"]


def test_repeated_resume_after_successful_resume_raises():
    @tool(requires_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        return "did it"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "dangerous", "arguments": {}, "id": "call_1"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[dangerous], max_turns=5)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    c.send("do the thing")
    c.resume({"tool:dangerous:call_1": True})
    assert c.messages[-1].text == "final"

    with pytest.raises(ConfigError, match="nothing to resume"):
        c.resume({"tool:dangerous:call_1": True})

    # A second resume must not wipe the folded history.
    assert c.messages[-1].text == "final"


def test_chat_stream_yields_events_and_updates_history():
    model = FakeModel(["streamed reply"])

    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    stream = c.stream("hello")

    kinds = [event.kind for event in stream]
    assert "text_delta" in kinds

    run = stream.run
    assert run.status == "completed"
    assert run.output == "streamed reply"
    assert [m.role for m in c.messages] == ["user", "assistant"]
    assert c.messages[-1].text == "streamed reply"


def test_chat_stream_absorbs_only_once():
    model = FakeModel(["once"])

    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    stream = c.stream("hi")
    for _ in stream:
        pass
    _ = stream.run
    _ = stream.run
    assert len(c.messages) == 2


def test_context_manager_survives_chat_resume():
    fired: list[int] = []

    def manager(messages, last_input_tokens):
        fired.append(len(messages))
        return messages

    @tool(requires_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        return "did it"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "dangerous", "arguments": {}, "id": "call_1"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[dangerous], max_turns=5)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy, context_manager=manager)  # NO approver -> the gated call pauses
    run = c.send("do the thing")
    assert run.status == "paused"
    fired_at_pause = len(fired)
    assert fired_at_pause == 1  # hook fired once, for the first provider call

    run2 = c.resume({"tool:dangerous:call_1": True})
    assert run2.status == "completed"
    assert run2.output == "final"
    # The hook must fire again on the post-resume provider call -- proof the chat
    # re-supplied its context_manager to the resumed run.
    assert len(fired) > fired_at_pause
    assert c.messages[-1].text == "final"


def test_approver_survives_chat_resume():
    executed = {"n": 0}

    @tool(requires_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        executed["n"] += 1
        return "did it"

    model = FakeModel(
        [
            {"tool_calls": [{"name": "dangerous", "arguments": {}, "id": "call_1"}]},
            {"tool_calls": [{"name": "dangerous", "arguments": {}, "id": "call_2"}]},
            "final",
        ]
    )

    @agent(model=model, tools=[dangerous], max_turns=6)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)  # no approver -> the first gated call pauses
    run = c.send("do the thing")
    assert run.status == "paused"
    assert c.pending is not None
    assert c.pending.id == "tool:dangerous:call_1"

    # Poke the private attr to simulate an approver configured at resume time.
    c._approver = lambda interrupt: True

    run2 = c.resume({"tool:dangerous:call_1": True})
    assert run2.status == "completed"
    assert run2.output == "final"
    # call_1 executed via the journaled answer; call_2 was approved INLINE by the
    # resume-time approver -- no second pause.
    assert executed["n"] == 2
    assert c.pending is None
    assert c.messages[-1].text == "final"


def test_load_chat_preserves_created_at():
    model = FakeModel(["first", "second"])

    @agent(model=model)
    def buddy() -> str:
        """You are Buddy."""
        return "unused"

    c = chat(buddy)
    c.send("hello")

    store = runs.open_default()
    row1 = store.chat_get(c.id)
    assert row1 is not None
    original_created = row1["created_at"]
    original_updated = row1["updated_at"]

    from composeai.chat import load_chat

    c2 = load_chat(c.id)
    c2.send("again")

    row2 = store.chat_get(c.id)
    assert row2 is not None
    assert row2["created_at"] == original_created  # unchanged across load + resend
    assert row2["updated_at"] > original_updated  # advanced by the second send
