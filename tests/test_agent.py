"""Tests for ``composeai.agentfn.agent`` -- the agent run loop (Phase 3).

Deliberately *not* using ``from __future__ import annotations`` here: several
tests define a small pydantic model local to the test function and reference
it in a decorated function's return annotation, which only resolves to the
real class (rather than an inert forward-reference string) when annotations
are evaluated eagerly.
"""

import time

import pytest
from pydantic import BaseModel

from composeai.agentfn import agent
from composeai.errors import (
    AgentTimeoutError,
    ComposeError,
    ConfigError,
    MaxTurnsExceededError,
    ModelRefusalError,
    ProviderError,
)
from composeai.messages import Message, StopReason, ToolCallPart, ToolResultPart, Usage
from composeai.models.base import ModelResponse
from composeai.testing import FakeModel
from composeai.tools import tool


@tool
def noop() -> str:
    """Do nothing, just acknowledge."""
    return "ok"


# --- basic loop: text-only, multi-turn tool loop ------------------------------


def test_single_turn_text_response():
    model = FakeModel(["Hello there."])

    @agent(model=model, max_turns=3)
    def greeter(name: str) -> str:
        """You are a friendly greeter."""
        return f"Greet {name}"

    run = greeter.run("Ann")
    assert run.output == "Hello there."
    assert run.status == "completed"


def test_multi_turn_tool_loop_ends_in_text():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            "All done.",
        ]
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def researcher(topic: str) -> str:
        """You are a researcher."""
        return f"Research {topic}"

    run = researcher.run("quantum computing")
    assert run.output == "All done."
    assert run.status == "completed"


# --- parallel tool calls -------------------------------------------------------


def test_parallel_tool_calls_batched_ordered_and_concurrent():
    @tool
    def slow_a() -> str:
        """Sleep a bit then return a."""
        time.sleep(0.15)
        return "a"

    @tool
    def slow_b() -> str:
        """Sleep a bit then return b."""
        time.sleep(0.15)
        return "b"

    model = FakeModel(
        [
            {
                "tool_calls": [
                    {"name": "slow_a", "arguments": {}, "id": "call_a"},
                    {"name": "slow_b", "arguments": {}, "id": "call_b"},
                ]
            },
            "done",
        ]
    )

    @agent(model=model, tools=[slow_a, slow_b], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    start = time.monotonic()
    run = runner.run()
    elapsed = time.monotonic() - start

    assert run.output == "done"
    assert elapsed < 0.25, f"tool calls did not run concurrently ({elapsed:.3f}s)"

    second_request = model.requests[1]
    tool_result_message = second_request.messages[-1]
    assert tool_result_message.role == "user"
    result_parts = [p for p in tool_result_message.parts if isinstance(p, ToolResultPart)]
    assert [p.tool_call_id for p in result_parts] == ["call_a", "call_b"]
    assert [p.content for p in result_parts] == ["a", "b"]


# --- tool exceptions / unknown tool -------------------------------------------


def test_tool_exception_is_error_and_loop_continues_and_span_records_error():
    @tool
    def boom() -> str:
        """Explode."""
        raise ValueError("kaboom")

    model = FakeModel(
        [
            {"tool_calls": [{"name": "boom", "arguments": {}, "id": "call_1"}]},
            "recovered",
        ]
    )

    @agent(model=model, tools=[boom], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.output == "recovered"

    tool_result_message = model.requests[1].messages[-1]
    result_part = tool_result_message.parts[0]
    assert isinstance(result_part, ToolResultPart)
    assert result_part.is_error is True
    assert result_part.content == "ValueError: kaboom"

    tool_spans = [s for s in run.trace.spans if s.kind == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0].status == "error"
    assert tool_spans[0].error is not None
    assert tool_spans[0].error.type == "ValueError"
    assert "kaboom" in tool_spans[0].error.stacktrace


def test_unknown_tool_name_returns_is_error_result_and_continues():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "does_not_exist", "arguments": {}}]},
            "done",
        ]
    )

    @agent(model=model, tools=[], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.output == "done"
    tool_result_message = model.requests[1].messages[-1]
    result_part = tool_result_message.parts[0]
    assert isinstance(result_part, ToolResultPart)
    assert result_part.is_error is True
    assert result_part.content == "unknown tool"

    # Minor-findings cleanup (Phase 10): the span should record "error" too --
    # it produced an is_error result, same as a real tool-body exception would.
    tool_spans = [s for s in run.trace.spans if s.kind == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0].status == "error"
    assert tool_spans[0].error is not None


def test_tool_use_response_with_no_tool_call_parts_raises_compose_error():
    # Minor-findings cleanup (Phase 10): a TOOL_USE stop_reason with zero
    # ToolCallParts is a provider/adapter bug -- refuse to append an empty
    # tool-results message and raise instead of silently corrupting the
    # conversation.
    bad_response = ModelResponse(
        message=Message(role="assistant", parts=[]),
        stop_reason=StopReason.TOOL_USE,
        raw_stop_reason="tool_use",
        usage=Usage(),
        model_id="fake",
    )
    model = FakeModel([bad_response])

    @agent(model=model, max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(ComposeError, match="no tool call parts|no tool-call parts"):
        runner.run()


def test_requires_approval_tool_pauses_the_agent_without_an_answer():
    # Phase 8 (see tests/test_hitl_agent.py for the full pause/resume/deny
    # round trip): an unanswered `requires_approval=True` tool call pauses
    # the agent loop instead of executing -- it no longer "just executes"
    # as it did in Phase 3, before human-in-the-loop existed.
    @tool(requires_approval=True)
    def dangerous() -> str:
        """Do something requiring approval."""
        return "done anyway"

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

    run = runner.run()
    assert run.status == "paused"
    assert run.pending is not None
    assert run.pending.id == "tool:dangerous:call_1"
    assert dangerous.spec.requires_approval is True


# --- structured output ---------------------------------------------------------


def test_structured_output_via_json_dict():
    class FactSheet(BaseModel):
        core_discoveries: list[str]

    model = FakeModel([{"json": {"core_discoveries": ["a", "b"]}}])

    @agent(model=model, max_turns=3)
    def researcher(topic: str) -> FactSheet:
        """Researcher."""
        return topic  # type: ignore[return-value]

    run = researcher.run("x")
    assert run.output == FactSheet(core_discoveries=["a", "b"])


def test_structured_output_wrapped_list_str():
    model = FakeModel([{"json": {"result": ["a", "b", "c"]}}])

    @agent(model=model, max_turns=3)
    def lister(topic: str) -> list[str]:
        """Lister."""
        return topic  # type: ignore[return-value]

    run = lister.run("x")
    assert run.output == ["a", "b", "c"]


def test_structured_output_parsed_fallback_via_json_text():
    class Out(BaseModel):
        x: int

    model = FakeModel(['{"x": 1}'])

    @agent(model=model, max_turns=3)
    def runner() -> Out:
        """Runner."""
        return "go"  # type: ignore[return-value]

    run = runner.run()
    assert run.output == Out(x=1)


def test_no_structured_payload_raises_compose_error():
    class Out(BaseModel):
        x: int

    model = FakeModel(["not valid json at all"])

    @agent(model=model, max_turns=3)
    def runner() -> Out:
        """Runner."""
        return "go"  # type: ignore[return-value]

    with pytest.raises(ComposeError, match="structured payload"):
        runner.run()


def test_dict_str_float_output_schema_keeps_value_type_not_collapsed_to_empty_object():
    """Regression: seal_schema used to collapse a dict[str, V] output
    schema's additionalProperties (a schema describing the value type) into
    False, forbidding the model from ever populating any key."""

    @agent(model=FakeModel(["placeholder"]), max_turns=3)
    def counts_agent(text: str) -> dict[str, float]:
        """Counts agent."""
        return text  # type: ignore[return-value]

    schema = counts_agent._output_schema
    assert schema is not None
    assert schema["type"] == "object"
    assert schema["additionalProperties"] == {"type": "number"}

    model = FakeModel([{"json": {"a": 1.5, "b": 2.5}}])

    @agent(model=model, max_turns=3)
    def counts_agent2(text: str) -> dict[str, float]:
        """Counts agent."""
        return text  # type: ignore[return-value]

    run = counts_agent2.run("x")
    assert run.output == {"a": 1.5, "b": 2.5}


# --- max_turns / retries / fallback --------------------------------------------


def test_max_turns_exceeded_raises():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
        ]
    )

    @agent(model=model, tools=[noop], max_turns=1)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(MaxTurnsExceededError):
        runner.run()


def test_retries_then_success_records_attributes():
    attempts = {"n": 0}

    def flaky(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ProviderError(f"fail {attempts['n']}", provider="test", model=request.model)
        return "success"

    model = FakeModel([flaky, flaky, flaky])

    @agent(model=model, retries=2, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.output == "success"

    llm_spans = [s for s in run.trace.spans if s.kind == "llm"]
    assert len(llm_spans) == 1
    retries_recorded = llm_spans[0].attributes["retries"]
    assert len(retries_recorded) == 2
    assert all(r["type"] == "ProviderError" for r in retries_recorded)


def test_fallback_switch_after_retries_exhausted_and_stays_on_fallback():
    def always_fail(request):
        raise ProviderError("down", provider="primary", model=request.model)

    primary = FakeModel([always_fail, always_fail])  # 1 initial + 1 retry
    fallback = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            "done on fallback",
        ]
    )

    @agent(model=primary, fallback=fallback, retries=1, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.output == "done on fallback"

    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    assert agent_spans[0].attributes["fallback_used"] == "FakeModel"

    # subsequent turns stayed on the fallback model, not the primary.
    assert len(primary.requests) == 2
    assert len(fallback.requests) == 2


def test_retries_exhausted_without_fallback_reraises_provider_error():
    def always_fail(request):
        raise ProviderError("down", provider="primary", model=request.model)

    model = FakeModel([always_fail, always_fail])

    @agent(model=model, retries=1, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(ProviderError):
        runner.run()


# --- capstone fix wave A: StopReason.ERROR/OTHER + fallback retries ------------


def _error_response() -> ModelResponse:
    """A *successful* call (no exception) whose own stop_reason reports
    failure -- no shipped adapter emits this today, but a custom Model
    might."""
    return ModelResponse(
        message=Message.assistant(""),
        stop_reason=StopReason.ERROR,
        raw_stop_reason="server_error",
        usage=Usage(input_tokens=5, output_tokens=5),
        model_id="fake",
    )


def test_stop_reason_error_response_is_retried_like_a_provider_error():
    """Regression: StopReason.ERROR used to bypass retries entirely (it's a
    *successful* return from the model call, not an exception, so the
    except ProviderError: retry loop never engaged) and fall through to an
    unrecoverable "unhandled stop_reason" ComposeError."""
    model = FakeModel([_error_response(), _error_response(), "success"])

    @agent(model=model, retries=2, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.output == "success"

    llm_spans = [s for s in run.trace.spans if s.kind == "llm"]
    assert len(llm_spans) == 1
    retries_recorded = llm_spans[0].attributes["retries"]
    assert len(retries_recorded) == 2


def test_stop_reason_error_falls_back_after_retries_exhausted():
    """StopReason.ERROR must also trigger the fallback mechanism, the same
    as a raised ProviderError does."""
    primary = FakeModel([_error_response(), _error_response()])  # 1 initial + 1 retry
    fallback = FakeModel(["done on fallback"])

    @agent(model=primary, fallback=fallback, retries=1, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.output == "done on fallback"
    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    assert agent_spans[0].attributes["fallback_used"] == "FakeModel"


def test_stop_reason_other_raises_compose_error_naming_raw_stop_reason():
    """OTHER stays terminal (no retry/fallback -- unlike ERROR), but the
    error message now includes the provider's own raw stop-reason string
    instead of just the normalized enum member name."""
    response = ModelResponse(
        message=Message.assistant(""),
        stop_reason=StopReason.OTHER,
        raw_stop_reason="content_filter",
        usage=Usage(input_tokens=5, output_tokens=5),
        model_id="fake",
    )
    model = FakeModel([response])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(ComposeError, match="content_filter"):
        runner.run()


def test_fallback_attempt_honors_the_agents_configured_retries():
    """Regression: the fallback call was always made with a hardcoded
    retries=0 (a single, non-retried attempt), silently ignoring the
    agent's configured `retries` -- if the fallback model also had a
    transient failure, the whole turn failed immediately even though the
    user configured retries=3."""

    def primary_always_fails(request):
        raise ProviderError("primary down", provider="primary", model=request.model)

    fallback_attempts = {"n": 0}

    def fallback_flaky(request):
        fallback_attempts["n"] += 1
        if fallback_attempts["n"] < 2:
            raise ProviderError("fallback transient", provider="fallback", model=request.model)
        return "fallback succeeded after its own retry"

    primary = FakeModel([primary_always_fails, primary_always_fails])  # 1 + 1 retry
    fallback = FakeModel([fallback_flaky, fallback_flaky])

    @agent(model=primary, fallback=fallback, retries=1, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.output == "fallback succeeded after its own retry"
    assert fallback_attempts["n"] == 2  # the fallback's own attempt, then one retry


# --- refusal / max_tokens / timeout --------------------------------------------


def test_refusal_raises_model_refusal_error_with_raw_stop_reason():
    response = ModelResponse(
        message=Message.assistant("I can't help with that."),
        stop_reason=StopReason.REFUSAL,
        raw_stop_reason="refusal",
        usage=Usage(input_tokens=5, output_tokens=5),
        model_id="fake",
    )
    model = FakeModel([response])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(ModelRefusalError) as exc_info:
        runner.run()
    assert exc_info.value.raw == "refusal"


def test_max_tokens_raises_compose_error():
    response = ModelResponse(
        message=Message.assistant("partial..."),
        stop_reason=StopReason.MAX_TOKENS,
        raw_stop_reason="max_tokens",
        usage=Usage(input_tokens=5, output_tokens=5),
        model_id="fake",
    )
    model = FakeModel([response])

    @agent(model=model, max_tokens=100, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(ComposeError, match="max_tokens"):
        runner.run()


def test_timeout_raised_at_turn_boundary_not_mid_call():
    def slow_turn(request):
        time.sleep(0.05)
        return {"tool_calls": [{"name": "noop", "arguments": {}}]}

    model = FakeModel([slow_turn, slow_turn, slow_turn])

    @agent(model=model, tools=[noop], max_turns=10, timeout=0.03)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(AgentTimeoutError):
        runner.run()


# --- usage rollup / trace shape -------------------------------------------------


def test_usage_rollup_across_turns_equals_sum():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            "done",
        ],
        usage=Usage(input_tokens=7, output_tokens=3),
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.usage.input_tokens == 14
    assert run.usage.output_tokens == 6


def test_trace_shape_agent_parents_llm_and_tool_spans():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            "done",
        ]
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    assert len(agent_spans) == 1
    agent_span = agent_spans[0]

    children = run.trace.children_of(agent_span.span_id)
    child_kinds = {c.kind for c in children}
    assert "llm" in child_kinds
    assert "tool" in child_kinds
    for child in children:
        assert child.parent_span_id == agent_span.span_id


# --- system prompt / user prompt semantics --------------------------------------


def test_docstring_becomes_system_prompt():
    model = FakeModel(["done"])

    @agent(model=model, max_turns=3)
    def researcher(topic: str) -> str:
        """You are a careful researcher.

        Focus on primary sources.
        """
        return f"Research {topic}"

    researcher.run("black holes")
    assert model.requests[0].system == "You are a careful researcher.\n\nFocus on primary sources."


def test_missing_docstring_means_no_system_prompt():
    model = FakeModel(["done"])

    @agent(model=model, max_turns=3)
    def runner(x: str) -> str:
        return x

    runner.run("hi")
    assert model.requests[0].system is None


def test_body_list_of_messages_used_verbatim():
    model = FakeModel(["done"])
    conversation = [
        Message.user("first"),
        Message.assistant("ack"),
        Message.user("second"),
    ]

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """System prompt."""
        return conversation  # type: ignore[return-value]

    run = runner.run()
    assert model.requests[0].messages[: len(conversation)] == conversation
    assert run.output == "done"


def test_body_return_type_other_than_str_or_message_list_raises_config_error():
    model = FakeModel(["done"])

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return 12345  # type: ignore[return-value]

    with pytest.raises(ConfigError):
        runner.run()


# --- sugar / introspection / run.messages --------------------------------------


def test_direct_call_sugar_equals_run_output():
    model_a = FakeModel(["hello"])
    model_b = FakeModel(["hello"])

    @agent(model=model_a, max_turns=3)
    def runner_a(name: str) -> str:
        """System."""
        return f"hi {name}"

    @agent(model=model_b, max_turns=3)
    def runner_b(name: str) -> str:
        """System."""
        return f"hi {name}"

    direct_result = runner_a("Ann")
    run_result = runner_b.run("Ann").output
    assert direct_result == run_result == "hello"


def test_name_and_output_type_introspection():
    model = FakeModel(["done"])

    class Out(BaseModel):
        x: int

    @agent(model=model, max_turns=3)
    def my_agent(topic: str) -> Out:
        """System."""
        return topic  # type: ignore[return-value]

    assert my_agent.name == "my_agent"
    assert my_agent.output_type is Out


def test_run_messages_include_tool_call_and_tool_result_parts():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            "done",
        ]
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    all_parts = [part for msg in run.messages for part in msg.parts]
    assert any(isinstance(p, ToolCallPart) for p in all_parts)
    assert any(isinstance(p, ToolResultPart) for p in all_parts)


def test_prompt_helper_is_a_typed_noop():
    """compose.prompt() returns its argument unchanged and satisfies type checkers."""
    import composeai as compose
    from composeai.messages import Message

    assert compose.prompt("hello") == "hello"
    msgs = [Message.user("hi")]
    assert compose.prompt(msgs) is msgs

    @compose.agent(model=FakeModel(script=[{"json": {"result": ["a", "b"]}}]))
    def lister(topic: str) -> list[str]:
        """List things."""
        return compose.prompt(f"List: {topic}")

    assert lister("x") == ["a", "b"]
