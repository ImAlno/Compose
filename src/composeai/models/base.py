"""The provider-agnostic model request/response shapes and the `Model` protocol.

Every adapter (Anthropic, OpenAI, ``FakeModel``, ...) speaks this one
vocabulary: a :class:`ModelRequest` in, a :class:`ModelResponse` out. Content
and usage accounting live in :mod:`composeai.messages`; this module only adds
the request/response envelope and the tool-declaration shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from ..messages import Message, StopReason, Usage


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool declaration passed to a model.

    ``input_schema`` must already be a complete JSON schema including
    ``additionalProperties: false`` -- Phase 3's ``@tool`` decorator is
    responsible for producing that; this layer just carries it through.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = True
    requires_approval: bool = False


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Everything needed to ask a model for one completion.

    ``model`` is the bare model id (no ``"provider/"`` prefix -- the
    registry strips that before building the adapter). ``provider`` (e.g.
    ``"anthropic"``) is carried alongside it purely as request metadata --
    adapters never read it (they already know their own provider) -- so
    that anything hashing a request (:mod:`composeai.testing`'s cassette/
    cache machinery) can distinguish two providers that happen to share a
    bare model id string. ``None`` for a request built from a bare
    ``Model`` instance rather than a ``"provider/model-id"`` string (see
    ``composeai.agentfn._make_slot``), since there's no provider label to
    carry in that case.
    """

    model: str
    messages: list[Message]
    system: str | None = None
    tools: list[ToolSpec] | None = None
    output_schema: dict[str, Any] | None = None
    max_tokens: int = 16000
    temperature: float | None = None
    provider: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A model's answer to one :class:`ModelRequest`.

    ``parsed`` is only populated when ``output_schema`` was set on the
    request; it's the JSON-decoded structured output.
    """

    message: Message
    stop_reason: StopReason
    raw_stop_reason: str | None
    usage: Usage
    model_id: str
    parsed: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RawStreamEvent:
    """One item yielded by :meth:`Model.stream`, before agent-loop mapping.

    ``kind`` mirrors (most of) :class:`~composeai.events.Event.kind`, but this
    is the adapter-facing vocabulary, not the bus's: adapters know nothing
    about :mod:`composeai.events` or tracing, they just describe what came
    off the wire.

    Adapters must guarantee the final event yielded by ``stream()`` is
    always ``"response_done"``, with ``response`` set to the same
    :class:`ModelResponse` that ``complete()`` would have returned for an
    equivalent (non-streamed) call -- consuming only that one event is
    exactly equivalent to calling ``complete()``.
    """

    kind: Literal[
        "text_delta",
        "thinking_delta",
        "tool_call_started",
        "tool_args_delta",
        "tool_call_finished",
        "response_done",
    ]
    text: str | None = None
    """Delta payload for ``text_delta``/``thinking_delta``; the raw JSON
    fragment for ``tool_args_delta``. Unset for every other kind."""
    tool_call_id: str | None = None
    """Set on ``tool_call_started``/``tool_args_delta``/``tool_call_finished``."""
    tool_name: str | None = None
    """Set on ``tool_call_started``/``tool_args_delta``/``tool_call_finished``."""
    response: ModelResponse | None = None
    """Set only on ``"response_done"`` -- the complete accumulated response."""


@runtime_checkable
class Model(Protocol):
    """The protocol every adapter implements.

    ``complete()`` is the only member of this ``@runtime_checkable``
    Protocol -- deliberately, even now that streaming exists. Adapters may
    *additionally* implement ``stream(self, request: ModelRequest) ->
    Iterator[RawStreamEvent]`` (see :class:`RawStreamEvent`), but it is not
    part of the structural contract: a Protocol member is required for
    every ``isinstance(m, Model)`` check to pass, and a duck-typed ``Model``
    that only implements ``complete()`` (a minimal user-written adapter, or
    anything flowing through ``registry.resolve()``'s instance-passthrough
    path) must keep satisfying that check. Call sites that want to stream
    look for the method with ``getattr(model, "stream", None)`` and degrade
    to plain ``complete()`` when it's absent (see
    ``composeai.agentfn._ainvoke_model``).

    Async capability follows the same duck-discovered pattern, not a
    Protocol member: an adapter may *additionally* implement ``async def
    acomplete(self, request: ModelRequest) -> ModelResponse`` and/or ``def
    astream(self, request: ModelRequest) -> AsyncIterator[RawStreamEvent]``
    (an async generator, mirroring ``stream()``). The engine prefers
    ``acomplete``/``astream`` when present -- checked with
    ``getattr(model, "acomplete", None) is not None`` -- and otherwise
    falls back to running the sync methods off-thread: ``asyncio.to_thread(
    model.complete, request)`` for a single call, or iterating sync
    ``stream()`` via ``to_thread`` hops when only sync streaming is
    available.
    """

    def complete(self, request: ModelRequest) -> ModelResponse: ...
