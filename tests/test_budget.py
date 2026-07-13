"""Tests for ``Budget`` enforcement -- token/usd caps checked after every LLM call.

FakeModel is always scripted with a fixed ``usage`` so token/cost math is
exact and deterministic. No SDKs, no network.

Deliberately *not* using ``from __future__ import annotations`` (same
reason as ``test_agent.py``).
"""

import pytest

from composeai.agentfn import agent
from composeai.combinators import aggregate, pipe
from composeai.errors import BudgetExceededError, ConfigError
from composeai.messages import Usage
from composeai.runs import Budget
from composeai.testing import FakeModel
from composeai.tools import tool


@tool
def noop() -> str:
    """Do nothing, just acknowledge."""
    return "ok"


# --- Budget construction -------------------------------------------------------


def test_budget_requires_at_least_one_field():
    with pytest.raises(ConfigError):
        Budget()


def test_budget_usd_only_is_valid():
    Budget(usd=1.0)


def test_budget_tokens_only_is_valid():
    Budget(tokens=100)


def test_budget_both_fields_is_valid():
    Budget(usd=1.0, tokens=100)


# --- token budget ----------------------------------------------------------------


def test_token_budget_trips_after_the_call_that_crosses_it():
    # Each call uses 10 input + 10 output = 20 tokens; budget = 30 tokens:
    # call 1 lands at 20 (under), call 2 lands at 40 (over) -> trips after call 2.
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            "done",
        ],
        usage=Usage(input_tokens=10, output_tokens=10),
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(BudgetExceededError):
        runner.run(budget=Budget(tokens=30))


def test_token_budget_not_tripped_when_generous():
    model = FakeModel(["done"], usage=Usage(input_tokens=10, output_tokens=10))

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run(budget=Budget(tokens=1000))
    assert run.output == "done"


# --- usd budget --------------------------------------------------------------------


def test_usd_budget_trips():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            "done",
        ],
        usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.6),
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(BudgetExceededError):
        runner.run(budget=Budget(usd=1.0))


def test_usd_budget_not_tripped_when_generous():
    model = FakeModel(["done"], usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.1))

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run(budget=Budget(usd=1.0))
    assert run.output == "done"


def test_unknown_cost_usage_does_not_trip_usd_budget():
    # Default FakeModel usage leaves cost_usd unset (None, i.e. unknown) --
    # a usd budget must never trip on a call whose cost is unknown.
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            "done",
        ]
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run(budget=Budget(usd=0.0001))
    assert run.output == "done"


# --- no budget: no check -----------------------------------------------------------


def test_no_budget_never_raises_regardless_of_usage():
    model = FakeModel(
        ["done"],
        usage=Usage(input_tokens=10**9, output_tokens=10**9, cost_usd=10**9),
    )

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.output == "done"


# --- pipeline / aggregate budgets ---------------------------------------------------


def test_pipeline_budget_spans_stages():
    model_a = FakeModel(["a out"], usage=Usage(input_tokens=10, output_tokens=10))
    model_b = FakeModel(["b out"], usage=Usage(input_tokens=10, output_tokens=10))

    @agent(model=model_a, max_turns=3)
    def a(x: str) -> str:
        """A."""
        return x

    @agent(model=model_b, max_turns=3)
    def b(x: str) -> str:
        """B."""
        return x

    pipeline = pipe(a, b)
    # Neither stage alone crosses 30 tokens (20 each) but combined = 40 > 30.
    with pytest.raises(BudgetExceededError):
        pipeline.run("go", budget=Budget(tokens=30))


def test_pipeline_budget_not_tripped_when_generous():
    model_a = FakeModel(["a out"], usage=Usage(input_tokens=10, output_tokens=10))
    model_b = FakeModel(["b out"], usage=Usage(input_tokens=10, output_tokens=10))

    @agent(model=model_a, max_turns=3)
    def a(x: str) -> str:
        """A."""
        return x

    @agent(model=model_b, max_turns=3)
    def b(x: str) -> str:
        """B."""
        return x

    pipeline = pipe(a, b)
    run = pipeline.run("go", budget=Budget(tokens=1000))
    assert run.output == "b out"


def test_aggregate_budget_spans_branches():
    model_a = FakeModel(["a"], usage=Usage(input_tokens=10, output_tokens=10))
    model_b = FakeModel(["b"], usage=Usage(input_tokens=10, output_tokens=10))

    @agent(model=model_a, max_turns=3)
    def branch_a(x: str) -> str:
        """A."""
        return x

    @agent(model=model_b, max_turns=3)
    def branch_b(x: str) -> str:
        """B."""
        return x

    agg = aggregate(a=branch_a, b=branch_b)
    with pytest.raises(BudgetExceededError):
        agg.run("x", budget=Budget(tokens=30))


def test_nested_inner_tighter_budget_trips_first():
    inner_model = FakeModel(["inner out"], usage=Usage(input_tokens=10, output_tokens=10))

    @agent(model=inner_model, max_turns=3)
    def inner_agent(x: str) -> str:
        """Inner."""
        return x

    def middle_stage(x: str) -> str:
        # Explicitly runs a *tighter* budget than the enclosing pipe's.
        run = inner_agent.run(x, budget=Budget(tokens=5))
        return run.output

    outer_model = FakeModel(["outer out"], usage=Usage(input_tokens=10, output_tokens=10))

    @agent(model=outer_model, max_turns=3)
    def outer_agent(x: str) -> str:
        """Outer."""
        return x

    pipeline = pipe(outer_agent, middle_stage)
    # The outer pipe budget is generous -- it would never trip on its own
    # (total usage across the whole pipe is only 40 tokens).
    with pytest.raises(BudgetExceededError) as exc_info:
        pipeline.run("go", budget=Budget(tokens=1000))

    # The inner (tighter) budget is the one that actually tripped.
    assert "5" in str(exc_info.value)


# --- streaming + budget -------------------------------------------------------------


def test_stream_with_budget_raises_from_run_property():
    model = FakeModel(
        [
            {"tool_calls": [{"name": "noop", "arguments": {}}]},
            "done",
        ],
        usage=Usage(input_tokens=10, output_tokens=10),
    )

    @agent(model=model, tools=[noop], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run_stream = runner.stream(budget=Budget(tokens=30))
    with pytest.raises(BudgetExceededError):
        list(run_stream)


# --- error message contents ---------------------------------------------------------


def test_budget_exceeded_error_message_contents():
    model = FakeModel(["done"], usage=Usage(input_tokens=50, output_tokens=50))

    @agent(model=model, max_turns=3)
    def runner() -> str:
        """Runner."""
        return "go"

    with pytest.raises(BudgetExceededError) as exc_info:
        runner.run(budget=Budget(tokens=10))

    message = str(exc_info.value)
    assert "10" in message  # the configured budget
    assert "100" in message  # usage so far (50 + 50)
    assert "runner" in message  # which run's budget tripped
