"""context_manager= hook (0.7.0)."""

import pytest

from composeai.agentfn import agent
from composeai.errors import ConfigError
from composeai.messages import Message
from composeai.testing import FakeModel
from composeai.tools import tool


@tool
def noop() -> str:
    """Do nothing."""
    return "ok"


def test_hook_called_before_every_provider_call_with_growing_history():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}, "id": "call_1"}]},
            "done",
        ]
    )

    calls: list[tuple[int, int]] = []

    def manager(messages: list[Message], last_input_tokens: int) -> list[Message]:
        calls.append((len(messages), last_input_tokens))
        return messages

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run(context_manager=manager)
    assert run.output == "done"
    assert len(calls) == 2
    assert calls[0] == (1, 0)  # first call: [user], no prior usage
    assert calls[1][0] == 3  # [user, assistant tool_use, user tool_result]
    assert calls[1][1] == 10  # FakeModel default usage: input_tokens=10


def test_hook_return_value_is_what_the_model_sees():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}, "id": "call_1"}]},
            "done",
        ]
    )

    def keep_last_only(messages: list[Message], last_input_tokens: int) -> list[Message]:
        return messages[-1:]

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run(context_manager=keep_last_only)
    assert run.output == "done"
    assert len(model.requests[1].messages) == 1


def test_hook_bad_return_raises_config_error():
    model = FakeModel(["never reached"])

    @agent(model=model)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(ConfigError, match="context_manager must return list\\[Message\\]"):
        runner.run(context_manager=lambda messages, last: "nope")  # type: ignore[arg-type,return-value]


def test_no_hook_is_zero_change():
    model = FakeModel(["plain"])

    @agent(model=model)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.output == "plain"
