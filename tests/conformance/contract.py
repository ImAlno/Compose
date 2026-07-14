"""Reusable ``Model``-contract assertions.

Not a test module itself (no ``test_`` prefix) -- these are building blocks
that provider-specific test files (``test_anthropic.py``, and Phase 4's
OpenAI equivalents) import and call against an already-configured
:class:`~composeai.models.base.Model` instance (a scripted ``FakeModel``,
or a real adapter wired to a stub/fake client).

Each function sends one fixed, simple :class:`ModelRequest` and asserts
the resulting :class:`ModelResponse` (or raised exception) has the shape
every adapter must produce, regardless of provider. The caller is
responsible for configuring the model so that its reply to that fixed
request matches the scenario being asserted (e.g. scripting ``FakeModel``
with a matching item, or stubbing an adapter's client to return a
matching canned response) -- what "matching" means for a given provider is
provider-specific and lives in that provider's own test file, not here.
"""

from __future__ import annotations

from composeai.errors import ProviderError
from composeai.messages import Message, StopReason, ToolCallPart, ToolResultPart
from composeai.models.base import Model, ModelRequest


def assert_text_completion(model: Model, *, model_id: str, expected_text: str) -> None:
    """(a) A plain user message round-trips to an assistant text reply."""
    request = ModelRequest(model=model_id, messages=[Message.user("hello")])
    response = model.complete(request)
    assert response.message.role == "assistant"
    assert response.message.text == expected_text
    assert response.stop_reason == StopReason.END_TURN
    assert response.model_id == model_id


def assert_tool_call(model: Model, *, model_id: str, tool_name: str) -> None:
    """(b) A tool-call response maps to one ``ToolCallPart`` with a real id."""
    request = ModelRequest(model=model_id, messages=[Message.user("call the tool")])
    response = model.complete(request)
    assert response.stop_reason == StopReason.TOOL_USE
    calls = [p for p in response.message.parts if isinstance(p, ToolCallPart)]
    assert len(calls) == 1
    assert calls[0].name == tool_name
    assert isinstance(calls[0].id, str) and calls[0].id


def assert_tool_result_batching_survives_round_trip(
    model: Model, *, model_id: str, tool_call_id: str
) -> None:
    """(c) A user message with multiple tool results for one turn is accepted as one call.

    Deeper "did the wire payload keep them in one API message" checks are
    provider-specific (they require inspecting that provider's captured
    request) and belong in that provider's own test file; this only
    asserts the provider-agnostic contract: passing such a message through
    ``complete()`` doesn't raise and produces a normal response.
    """
    request = ModelRequest(
        model=model_id,
        messages=[
            Message.assistant([ToolCallPart(id=tool_call_id, name="f", arguments={})]),
            Message.user(
                [
                    ToolResultPart(tool_call_id=tool_call_id, content="result one"),
                    ToolResultPart(tool_call_id="other-id", content="result two"),
                ]
            ),
        ],
    )
    response = model.complete(request)
    assert response.message.role == "assistant"


def assert_structured_output(model: Model, *, model_id: str, expected: dict) -> None:
    """(d) ``output_schema`` set -> ``parsed`` carries the decoded JSON dict."""
    request = ModelRequest(
        model=model_id,
        messages=[Message.user("give me json")],
        output_schema={"type": "object"},
    )
    response = model.complete(request)
    assert response.parsed == expected
    assert response.stop_reason == StopReason.END_TURN


def assert_tool_use_with_output_schema_does_not_crash(
    model: Model, *, model_id: str, tool_name: str
) -> None:
    """(d2) A tool-only turn with ``output_schema`` set must not crash.

    Regression for the "structured-output parse crash on any non-END_TURN
    stop reason" critical: an ``@agent`` combining ``tools=[...]`` with a
    non-``str`` ``output_type`` sends ``output_schema`` on *every* turn
    (``agentfn.py``'s ``_aperform_turn``), including the first turn where the
    model just calls a tool and returns no text at all. Adapters must only
    attempt to JSON-decode the response text when ``stop_reason`` is
    ``END_TURN`` -- for ``TOOL_USE`` (this scenario), ``parsed`` must stay
    ``None`` rather than raising trying to ``json.loads('')``. This mirrors
    ``agentfn.py``'s own loop, which only ever reads ``response.parsed`` on
    the ``StopReason.END_TURN`` branch (``_extract_output`` is never called
    for ``TOOL_USE``/``REFUSAL``/``MAX_TOKENS``/``OTHER``).
    """
    request = ModelRequest(
        model=model_id,
        messages=[Message.user("call the tool")],
        output_schema={"type": "object"},
    )
    response = model.complete(request)
    assert response.stop_reason == StopReason.TOOL_USE
    assert response.parsed is None


def assert_stop_reason_mapping(
    model: Model, *, model_id: str, expected: StopReason, expected_raw: str | None = None
) -> None:
    """(e) One entry of a provider's raw-stop-reason -> ``StopReason`` table."""
    request = ModelRequest(model=model_id, messages=[Message.user("hi")])
    response = model.complete(request)
    assert response.stop_reason == expected
    if expected_raw is not None:
        assert response.raw_stop_reason == expected_raw


def assert_usage_lands(
    model: Model, *, model_id: str, expected_input: int, expected_output: int
) -> None:
    """(f) Token counts from the provider land unchanged in ``Usage``."""
    request = ModelRequest(model=model_id, messages=[Message.user("hi")])
    response = model.complete(request)
    assert response.usage.input_tokens == expected_input
    assert response.usage.output_tokens == expected_output


def assert_sdk_failure_raises_provider_error(model: Model, *, model_id: str) -> None:
    """(g) A configured-to-fail model raises ``ProviderError`` from ``complete()``."""
    request = ModelRequest(model=model_id, messages=[Message.user("hi")])
    try:
        model.complete(request)
    except ProviderError:
        return
    raise AssertionError("expected ProviderError to be raised")
