"""``@agent``: the product's headline API -- a typed function that runs an LLM loop.

The decorated function's docstring becomes the system prompt and its body
returns the user prompt (or a full conversation); its return-type
annotation becomes the structured output type. Calling the decorated object
runs the loop to completion (or raises); ``.run(...)`` returns the full
:class:`~composeai.runs.Run` instead of just the typed output.

``.stream(...)`` runs the same loop on a background thread against a fresh,
private :class:`~composeai.events.EventBus` and returns a
:class:`~composeai.runs.RunStream` -- streaming and tracing are the same
event bus, so what you get by iterating it is exactly the span
started/finished events tracing already produces, interleaved with token
deltas from adapters that support :meth:`~composeai.models.base.Model.stream`.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from . import events, runs, tracing
from ._encoding import from_jsonable, to_jsonable
from ._ids import new_ulid
from ._schema import register_annotation_types, resolve_annotations, seal_schema
from .errors import (
    AgentTimeoutError,
    ComposeError,
    ConfigError,
    MaxTurnsExceededError,
    ModelRefusalError,
    ProviderError,
)
from .hitl import Interrupt, _Pause
from .messages import Message, StopReason, ToolCallPart, ToolResultPart, Usage
from .models import registry
from .models.base import Model, ModelRequest, ModelResponse, ToolSpec
from .runs import Budget, Run, RunStream, budget_scope, check_budgets
from .tools import Tool

# --- agent registry (Phase 8: resume() routing) ------------------------------
#
# Mirrors `composeai.flow._FLOW_REGISTRY`: `resume()` looks a paused/crashed
# standalone agent run up by name here (see `resume_standalone_agent` below).
# Duplicate names raise `ConfigError` at decoration time, same as `@flow`.
_AGENT_REGISTRY: dict[str, AgentFunction] = {}


class AgentFunction:
    """The callable object produced by ``@agent``.

    ``agent_fn(...)`` is sugar for ``agent_fn.run(...).output``; ``.run(...)``
    runs the full loop and returns a :class:`~composeai.runs.Run`. Both
    accept an optional keyword-only ``budget``: a :class:`~composeai.runs.Budget`
    enforced across every LLM call in the run (see
    :func:`~composeai.runs.check_budgets`). ``.stream(...)`` runs the same
    loop on a background thread and returns a :class:`~composeai.runs.RunStream`
    for live consumption. ``.name``, ``.input_type``, and ``.output_type``
    support introspection (e.g. by ``composeai.combinators.pipe()``).
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        model: str | Model,
        tools: Sequence[Tool],
        temperature: float | None,
        max_tokens: int,
        max_turns: int,
        retries: int,
        max_repairs: int,
        timeout: float | None,
        fallback: str | Model | None,
        cache: bool,
        name: str | None,
        replace: bool,
    ) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        # Phase 8 (human-in-the-loop): register by name so `resume()` can route
        # a paused/crashed standalone agent run back to its definition (mirrors
        # `composeai.flow.Flow`'s own registry). `replace=True` re-binds an
        # existing name instead of raising (see `agent()`'s docstring for the
        # standalone-resume caveat that comes with doing so).
        if not replace and self.name in _AGENT_REGISTRY:
            raise ConfigError(
                f"@agent name {self.name!r} is already registered -- agent names "
                "must be unique per process (needed so resume() can route a "
                "paused agent run back to its definition); pass name=... or "
                "replace=True, or rename the function"
            )
        _AGENT_REGISTRY[self.name] = self
        # Phase 7 (durable flows): register every pydantic model/dataclass/enum
        # reachable from this function's annotations so a fresh process can
        # decode journaled values referencing them (see composeai._schema).
        register_annotation_types(fn)

        hints = resolve_annotations(fn, include_extras=True)
        self.output_type: Any = hints.get("return", str)

        sig = inspect.signature(fn)
        first_param = next(iter(sig.parameters.values()), None)
        self.input_type: Any = hints.get(first_param.name, Any) if first_param is not None else Any

        self._system = _system_prompt(fn.__doc__)
        self._model = model
        self._tools = list(tools)
        self._tools_by_name = {t.name: t for t in self._tools}
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._retries = retries
        self._max_repairs = max_repairs
        self._timeout = timeout
        self._fallback = fallback
        self._cache = cache
        self._output_schema, self._wrap_result = _build_output_schema(self.output_type)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs).output

    def run(self, *args: Any, budget: Budget | None = None, **kwargs: Any) -> Run:
        return _run_agent(self, args, kwargs, budget=budget)

    def stream(self, *args: Any, budget: Budget | None = None, **kwargs: Any) -> RunStream:
        return _stream_agent(self, args, kwargs, budget=budget)


def prompt(text_or_messages: str | Sequence[Message]) -> Any:
    """Mark a value as the user prompt an ``@agent`` body is returning.

    An ``@agent`` function's return *annotation* declares the agent's output
    type (e.g. ``-> FactSheet``) while its *body* returns the user prompt --
    a ``str`` (or a full ``list[Message]`` conversation). That is the core
    ``@agent`` idiom, but to a static type checker the bare ``return
    f"..."`` looks like returning ``str`` where ``FactSheet`` was promised.
    Wrapping the prompt in this no-op (declared ``-> Any``) keeps type
    checkers satisfied and makes the intent explicit::

        @agent(model="anthropic/claude-sonnet-5")
        def researcher(topic: str) -> FactSheet:
            \"\"\"You are a meticulous researcher.\"\"\"
            return prompt(f"Build a fact sheet about: {topic}")

    Returning the bare ``str``/``list[Message]`` works identically at
    runtime; this wrapper only exists for type-checker ergonomics.
    """
    return text_or_messages


def agent(
    *,
    model: str | Model,
    tools: Sequence[Tool] = (),
    temperature: float | None = None,
    max_tokens: int = 16000,
    max_turns: int = 10,
    retries: int = 0,
    max_repairs: int = 0,
    timeout: float | None = None,
    fallback: str | Model | None = None,
    cache: bool = False,
    name: str | None = None,
    replace: bool = False,
) -> Callable[[Callable[..., Any]], AgentFunction]:
    """Decorate a function into an :class:`AgentFunction` -- a typed, runnable agent.

    ``model`` is required and keyword-only: either a ``"provider/model-id"``
    string (resolved lazily via :mod:`composeai.models.registry`) or an
    existing :class:`~composeai.models.base.Model` instance (e.g.
    ``FakeModel`` in tests). ``fallback`` is resolved lazily, only if every
    ``retries`` attempt against the primary model fails.

    Note for ``retries > 0`` combined with ``.stream(...)``: a retry re-runs
    the whole model call, so a provider error that strikes *mid-stream* --
    after some deltas were already published -- re-streams the call from the
    start, and consumers may see the same text deltas twice on the same llm
    span. This is inherent to retrying a stream (composeai never buffers
    deltas to dedupe them); UI consumers that must not double-render should
    treat a fresh delta sequence on an llm span that already produced
    deltas as a reset (or simply wait for the span's ``span_finished``
    event and render final outputs only).

    ``cache=True`` (Phase 9's dev/test kit) wraps the resolved model (and
    fallback, if any) in a :class:`~composeai.testing.CachingModel`: a
    filesystem-backed cache of ``complete()`` responses under
    ``{COMPOSE_DIR}/cache/`` keyed by the full request hash. A hit skips
    the real call and reports ``usage=Usage()`` (zero -- never re-billed)
    on the llm span, tagged ``attributes["cached"] = True``. Applies to
    non-streaming calls only; ``.stream(...)`` always calls the real model
    (see ``CachingModel``'s docstring). Meant for local development/tests
    against a real provider (repeated runs during iteration don't re-pay
    for identical calls) -- for deterministic, offline test fixtures,
    prefer a cassette (``composeai.testing.record_cassette``/
    ``replay_cassette``/the ``cassette`` pytest fixture) instead.

    ``max_repairs`` (structured-output agents only): when the model's final
    reply fails JSON parsing or schema validation, instead of raising
    immediately, append the error as a corrective user message and re-ask --
    up to ``max_repairs`` times per run. Each repair is a full-price LLM
    turn (the whole conversation is re-sent) and counts against
    ``max_turns``. Dramatically more effective than a cold re-run for small
    local models; keep the default ``0`` when you'd rather fail fast.

    ``name`` overrides the registered/routing name (default: the decorated
    function's ``__name__``) -- the same escape hatch ``@flow``/``@task``
    already have, e.g. for two agents that would otherwise share a
    function name.

    ``replace=True`` re-binds an existing agent name instead of raising
    ``ConfigError``. WARNING: standalone-agent resume has no fingerprint/
    staleness check (unlike ``@flow`` -- see ``resume_standalone_agent``),
    so a paused run resumed after a replace continues silently against the
    NEW definition. Meant for tests and deliberate rebinding, not for
    multi-tenant factories -- those want distinct ``name=``s.
    """

    def decorator(fn: Callable[..., Any]) -> AgentFunction:
        return AgentFunction(
            fn,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
            retries=retries,
            max_repairs=max_repairs,
            timeout=timeout,
            fallback=fallback,
            cache=cache,
            name=name,
            replace=replace,
        )

    return decorator


# --- system prompt / output schema -------------------------------------------


def _system_prompt(doc: str | None) -> str | None:
    if not doc or not doc.strip():
        return None
    return inspect.cleandoc(doc)


def _build_output_schema(output_type: Any) -> tuple[dict[str, Any] | None, bool]:
    """Return ``(output_schema, wrapped)`` for ``output_type``.

    ``None`` means plain text output (``str`` or a missing annotation).
    Otherwise the schema is sealed (titles stripped, ``additionalProperties:
    false`` everywhere including ``$defs``); if its root isn't a JSON object
    (e.g. ``list[str]``), it's wrapped as ``{"result": <schema>}`` and
    ``wrapped=True`` tells the loop to unwrap ``parsed["result"]`` later.
    """
    if output_type is str:
        return None, False

    adapter = TypeAdapter(output_type)
    schema = adapter.json_schema()
    seal_schema(schema)
    defs = schema.pop("$defs", None)

    if schema.get("type") == "object":
        result_schema = schema
        wrapped = False
    else:
        result_schema = {
            "type": "object",
            "properties": {"result": schema},
            "required": ["result"],
            "additionalProperties": False,
        }
        wrapped = True

    if defs:
        result_schema["$defs"] = defs
    return result_schema, wrapped


def _build_conversation(
    agent_fn: AgentFunction, call_args: tuple[Any, ...], call_kwargs: dict[str, Any]
) -> list[Message]:
    body_result = agent_fn._fn(*call_args, **call_kwargs)
    if isinstance(body_result, str):
        return [Message.user(body_result)]
    if isinstance(body_result, list) and all(isinstance(item, Message) for item in body_result):
        return list(body_result)
    raise ConfigError(
        f"@agent function {agent_fn.name!r} must return a str (the user prompt) "
        "or a list[Message] (the full conversation) from its body, got "
        f"{type(body_result)!r}"
    )


def _extract_output(agent_fn: AgentFunction, response: ModelResponse) -> Any:
    if agent_fn._output_schema is None:
        return response.message.text

    if response.parsed is not None:
        payload: Any = response.parsed
    else:
        try:
            payload = json.loads(response.message.text)
        except json.JSONDecodeError as exc:
            raise ComposeError(
                f"@agent {agent_fn.name!r}: model returned no structured payload "
                "(response.parsed was unset and the message text is not valid JSON)"
            ) from exc

    if agent_fn._wrap_result:
        if not isinstance(payload, dict) or "result" not in payload:
            raise ComposeError(
                f"@agent {agent_fn.name!r}: model returned no structured payload "
                "(expected a JSON object with a 'result' key)"
            )
        payload = payload["result"]

    adapter = TypeAdapter(agent_fn.output_type)
    try:
        return adapter.validate_python(payload)
    except ValidationError as exc:
        # Normalized so callers (and the repair turn) can catch one exception
        # type for every "the model's own output was wrong" case -- the
        # JSON-decode and missing-'result' branches above already raise
        # ComposeError; a raw pydantic error escaping here was the odd one out.
        raise ComposeError(
            f"@agent {agent_fn.name!r}: model output failed validation for "
            f"{agent_fn.output_type!r}: {exc}"
        ) from exc


# --- model resolution (primary + lazy fallback) ------------------------------


@dataclass
class _ModelSlot:
    model: Model
    bare_id: str
    provider: str | None
    label: str


def _make_slot(spec: str | Model, *, cache: bool = False) -> _ModelSlot:
    resolved = registry.resolve(spec)
    if cache:
        # Lazy import (Phase 9's dev/test kit -- see composeai.testing):
        # keeps this module's default import path from pulling in the
        # cassette/cache machinery for every agent that never sets
        # cache=True, mirroring the lazy provider-SDK imports elsewhere
        # in this codebase (composeai.models.registry/anthropic/openai).
        from .testing import CachingModel

        resolved = CachingModel(resolved)
    if isinstance(spec, str):
        provider, bare_id = registry.parse_model_string(spec)
        return _ModelSlot(model=resolved, bare_id=bare_id, provider=provider, label=spec)
    bare_id = getattr(spec, "model_id", None) or type(spec).__name__
    return _ModelSlot(model=resolved, bare_id=bare_id, provider=None, label=bare_id)


class _RunState:
    """Mutable per-run state: which model slot is currently active."""

    def __init__(self, slot: _ModelSlot) -> None:
        self.slot = slot
        self.fallback_active = False


def _invoke_model(
    slot: _ModelSlot, request: ModelRequest, streaming: bool, llm_span: tracing.Span
) -> ModelResponse:
    """Call ``slot.model`` for one request, streaming deltas onto the bus if possible.

    When ``streaming`` is true and the model has a ``stream`` method (an
    optional extension -- see :class:`~composeai.models.base.Model`), drives
    it and forwards each :class:`~composeai.models.base.RawStreamEvent` onto
    the ambient event bus as an :class:`~composeai.events.Event` (a no-op
    when there's no ambient bus), using the model's final ``response_done``
    event as the result. Otherwise (non-streaming, or a model lacking
    ``stream``) this is exactly ``slot.model.complete(request)``.
    """
    stream_fn = getattr(slot.model, "stream", None) if streaming else None
    if stream_fn is None:
        return slot.model.complete(request)

    response: ModelResponse | None = None
    for raw_event in stream_fn(request):
        if raw_event.kind == "response_done":
            response = raw_event.response
            continue
        name = raw_event.tool_name if raw_event.tool_name is not None else slot.label
        data = {"id": raw_event.tool_call_id} if raw_event.tool_call_id is not None else None
        events.emit(
            events.Event(
                kind=raw_event.kind,
                trace_id=llm_span.trace_id,
                span_id=llm_span.span_id,
                name=name,
                text=raw_event.text,
                data=data,
            )
        )
    if response is None:
        raise ComposeError(
            f"Model.stream() for {slot.label!r} ended without yielding a "
            "response_done event -- adapters must always yield exactly one, "
            "as their final event"
        )
    return response


def _call_llm(
    slot: _ModelSlot,
    system: str | None,
    messages: list[Message],
    tools: list[ToolSpec] | None,
    output_schema: dict[str, Any] | None,
    max_tokens: int,
    temperature: float | None,
    retries: int,
    streaming: bool = False,
) -> ModelResponse:
    request = ModelRequest(
        model=slot.bare_id,
        messages=list(messages),
        system=system,
        tools=tools,
        output_schema=output_schema,
        max_tokens=max_tokens,
        temperature=temperature,
        provider=slot.provider,
    )
    attributes: dict[str, Any] = {"model": slot.bare_id}
    if slot.provider is not None:
        attributes["provider"] = slot.provider
    with tracing.span("llm", slot.label, attributes=attributes) as llm_span:
        llm_span.set_input({"system": system, "messages": list(messages)})
        attempt = 0
        while True:
            try:
                response = _invoke_model(slot, request, streaming, llm_span)
            except ProviderError as exc:
                llm_span.attributes.setdefault("retries", []).append(
                    {"type": type(exc).__name__, "message": str(exc)}
                )
                if attempt < retries:
                    attempt += 1
                    continue
                raise
            if response.stop_reason == StopReason.ERROR:
                # A *successful* call (no exception -- `_invoke_model` returned
                # normally) whose own stop_reason nonetheless reports failure:
                # no adapter shipped with composeai emits this today, but a
                # custom `Model` might, and treating it as terminal would
                # silently discard the retries/fallback a user configured
                # specifically for resilience against provider failures.
                # Converting it into a `ProviderError` here reuses the exact
                # retry loop above and `_perform_turn`'s existing fallback
                # path below, uniformly -- no separate mechanism needed.
                message = (
                    f"model returned stop_reason=ERROR (raw={response.raw_stop_reason!r})"
                )
                llm_span.attributes.setdefault("retries", []).append(
                    {"type": "ProviderError", "message": message}
                )
                if attempt < retries:
                    attempt += 1
                    continue
                raise ProviderError(
                    message, provider=attributes.get("provider"), model=slot.bare_id
                )
            llm_span.usage = response.usage
            # Phase 9 (@agent(cache=True)): duck-typed, so this module never
            # needs to import composeai.testing.CachingModel just to check --
            # see that class's docstring for why usage is already zeroed on
            # the response itself by the time we get here.
            if getattr(slot.model, "last_was_hit", False):
                llm_span.attributes["cached"] = True
            check_budgets()
            llm_span.set_output(response.message)
            return response


def _perform_turn(
    agent_fn: AgentFunction,
    agent_span: tracing.Span,
    state: _RunState,
    system: str | None,
    conversation: list[Message],
    tools: list[ToolSpec] | None,
    output_schema: dict[str, Any] | None,
    streaming: bool = False,
) -> ModelResponse:
    try:
        return _call_llm(
            state.slot,
            system,
            conversation,
            tools,
            output_schema,
            agent_fn._max_tokens,
            agent_fn._temperature,
            agent_fn._retries,
            streaming,
        )
    except ProviderError:
        if agent_fn._fallback is None or state.fallback_active:
            raise
        state.slot = _make_slot(agent_fn._fallback, cache=agent_fn._cache)
        state.fallback_active = True
        agent_span.attributes["fallback_used"] = (
            agent_fn._fallback if isinstance(agent_fn._fallback, str) else state.slot.label
        )
        return _call_llm(
            state.slot,
            system,
            conversation,
            tools,
            output_schema,
            agent_fn._max_tokens,
            agent_fn._temperature,
            agent_fn._retries,
            streaming,
        )


# --- tool execution -----------------------------------------------------------


class UnknownToolError(Exception):
    """Internal control-flow signal: raised inside the ``tool`` span (see
    :func:`_execute_one_tool`) when the model calls a name this agent has no
    ``@tool`` for, then caught right there and turned into the same
    ``is_error`` :class:`ToolResultPart` as before (content ``"unknown
    tool"``, loop continues) -- the only effect of routing it through a real
    exception is that the span records status ``"error"`` (see
    ``composeai.tracing.span``), same as a genuine tool-body failure would.
    Deliberately module-private (not part of ``composeai.errors``): it never
    escapes this function, but its name/message do show up in rendered
    traces as the span's error, so both are written for that audience.
    """


def _execute_one_tool(agent_fn: AgentFunction, call: ToolCallPart) -> ToolResultPart:
    """Run one tool call, wrapped in its own ``tool`` span.

    An unknown tool name and a regular ``Exception`` raised by the tool body
    both become an ``is_error`` result (the loop continues) and both mark
    the span ``"error"`` -- only the result content differs (``"unknown
    tool"`` vs. ``f"{type}: {message}"``). A :class:`~composeai.hitl._Pause`
    (e.g. from ``ask_human()`` inside the tool body) is a
    :class:`BaseException`, so it is *not* caught here -- it propagates out
    (through the ``tool`` span, marked ``"paused"`` rather than ``"error"``
    -- see :mod:`composeai.tracing`) to whichever caller is watching for it.
    """
    tool_obj = agent_fn._tools_by_name.get(call.name)
    try:
        with tracing.span("tool", call.name, input=call.arguments) as tool_span:
            if tool_obj is None:
                raise UnknownToolError(
                    f"the model called tool {call.name!r} but this agent has no such tool"
                )
            content = tool_obj.execute(call.arguments)
            tool_span.set_output(content)
    except UnknownToolError:
        return ToolResultPart(tool_call_id=call.id, content="unknown tool", is_error=True)
    except Exception as exc:
        return ToolResultPart(
            tool_call_id=call.id,
            content=f"{type(exc).__name__}: {exc}",
            is_error=True,
        )
    return ToolResultPart(tool_call_id=call.id, content=content)


def _tool_requires_approval(agent_fn: AgentFunction, call: ToolCallPart) -> bool:
    tool_obj = agent_fn._tools_by_name.get(call.name)
    return tool_obj is not None and tool_obj.spec.requires_approval


def _approval_interrupt_id(call: ToolCallPart) -> str:
    return f"tool:{call.name}:{call.id}"


def _deny_tool_call(call: ToolCallPart) -> ToolResultPart:
    with tracing.span("tool", call.name, input=call.arguments) as tool_span:
        tool_span.set_output("denied by user")
    return ToolResultPart(tool_call_id=call.id, content="denied by user", is_error=True)


def _pause_for_unanswered_approval(call: ToolCallPart) -> _Pause:
    """Build, span-record, and return (not raise) the pause for an unanswered approval call.

    Does *not* persist anything -- ``_process_tool_use`` collects every
    pause built this way and, if the batch ends up pausing, persists all of
    them atomically together with the turn's ``agent_state`` snapshot (see
    :meth:`~composeai.runs.RunStore.persist_pause`). Previously this
    persisted its interrupt immediately, one ``pending_interrupts`` row (and
    commit) at a time, well before the snapshot needed to safely resume it
    was written -- a crash in that window orphaned the interrupt and, on
    resume, re-ran the whole turn from scratch (re-executing any
    already-completed sibling tool calls' side effects a second time).
    """
    interrupt = Interrupt(
        id=_approval_interrupt_id(call),
        kind="approval",
        payload={"tool": call.name, "arguments": call.arguments},
    )
    try:
        with tracing.span("pause", interrupt.id):
            raise _Pause(interrupt)
    except _Pause as exc:
        return exc


def _process_tool_use(
    agent_fn: AgentFunction,
    calls: list[ToolCallPart],
    conversation: list[Message],
    turn: int,
    run_id: str | None,
    scope_key: str,
    seed_results: dict[str, ToolResultPart] | None = None,
) -> Message:
    """Execute one tool-call batch, pausing if an approval-gated call is
    unanswered or any tool body raises ``_Pause`` (``ask_human``).

    1. Every non-approval-gated call not already in ``seed_results`` (a
       restored snapshot's already-completed calls) runs in parallel, same
       as before Phase 8 -- each under its own ``tool:{call_id}`` journal
       scope (see ``composeai.runs``'s scope-stack module docs), so a
       ``@task``/``@agent`` call inside a tool body gets a deterministic key
       regardless of which worker thread reaches it first.
    2. Approval-gated calls are then checked *in order*: answered True ->
       execute now (same scope); answered False -> a "denied by user"
       ``is_error`` result; unanswered -> build (not yet persist) its pause
       (see :func:`_pause_for_unanswered_approval`) and keep scanning the
       rest (independent calls may still resolve) -- this "keep scanning"
       rule also applies when an *answered* call's own body pauses (e.g.
       nested ``ask_human()``): that used to ``break`` out of the loop
       entirely, silently skipping every call after it; it now
       ``continue``s, matching the unanswered-gate case two lines above and
       this function's own contract.
    3. If anything paused, atomically persist every pending interrupt
       collected in steps 1-2 together with the ``conversation``/``turn``/
       whatever results *did* complete (so a resume doesn't redo them, and
       so a crash right after this point leaves a fully consistent,
       resumable pause -- see :meth:`~composeai.runs.RunStore.persist_pause`)
       and raise the *first* pause encountered, in call order. Otherwise,
       merge every result into one batched user message, in the calls'
       original order.
    """
    results: dict[str, ToolResultPart] = dict(seed_results) if seed_results else {}
    approval_ids = {call.id for call in calls if _tool_requires_approval(agent_fn, call)}
    normal_calls = [c for c in calls if c.id not in approval_ids and c.id not in results]
    approval_calls = [c for c in calls if c.id in approval_ids]

    pause_exc: _Pause | None = None
    pending_interrupts: list[Interrupt] = []

    if normal_calls:

        def _run_one(
            call: ToolCallPart,
        ) -> tuple[str, ToolResultPart | None, BaseException | None]:
            try:
                with runs.push_scope(f"tool:{call.id}"):
                    return call.id, _execute_one_tool(agent_fn, call), None
            except BaseException as exc:  # noqa: BLE001 -- _Pause (or worse) settled below
                return call.id, None, exc

        if len(normal_calls) == 1:
            outcomes = [_run_one(normal_calls[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(normal_calls)) as pool:
                futures = [pool.submit(tracing.propagate(_run_one), c) for c in normal_calls]
                outcomes = [f.result() for f in futures]

        for call_id, result, exc in outcomes:
            if exc is None:
                assert result is not None
                results[call_id] = result
            elif isinstance(exc, _Pause):
                if pause_exc is None:
                    pause_exc = exc
            else:
                raise exc

    for call in approval_calls:
        if call.id in results:
            continue
        interrupt_id = _approval_interrupt_id(call)
        hit, value = runs.interrupt_lookup(f"__interrupt__:{interrupt_id}")
        if not hit:
            candidate = _pause_for_unanswered_approval(call)
            pending_interrupts.append(candidate.interrupt)
            if pause_exc is None:
                pause_exc = candidate
            continue
        if bool(value):
            try:
                with runs.push_scope(f"tool:{call.id}"):
                    results[call.id] = _execute_one_tool(agent_fn, call)
            except _Pause as exc:
                if pause_exc is None:
                    pause_exc = exc
                continue
        else:
            results[call.id] = _deny_tool_call(call)

    if pause_exc is not None:
        if run_id is not None:
            runs.open_default().persist_pause(
                run_id=run_id,
                interrupts=pending_interrupts,
                scope_key=scope_key,
                messages_json=json.dumps(to_jsonable(conversation)),
                partial_results_json=json.dumps(to_jsonable(results)),
                turn=turn,
            )
        raise pause_exc

    return Message.user([results[c.id] for c in calls])


# --- agent conversation snapshots (Phase 8: pause/resume + crash tolerance) ----


@dataclass
class _RestoredAgentState:
    conversation: list[Message]
    turn: int
    partial_results: dict[str, ToolResultPart]


def _load_agent_state(run_id: str | None, scope_key: str) -> _RestoredAgentState | None:
    if run_id is None:
        return None
    store = runs.open_default()
    row = store.agent_state_get(run_id, scope_key)
    if row is None:
        return None
    conversation = from_jsonable(json.loads(row["messages_json"]))
    partial_results = (
        from_jsonable(json.loads(row["partial_results_json"]))
        if row["partial_results_json"]
        else {}
    )
    return _RestoredAgentState(
        conversation=conversation, turn=row["turn"], partial_results=partial_results
    )


def _snapshot_agent_state(
    run_id: str | None,
    scope_key: str,
    conversation: list[Message],
    turn: int,
    partial_results: dict[str, ToolResultPart],
) -> None:
    """No-op when ``run_id`` is ``None`` (no durable run to snapshot against --
    e.g. an agent nested directly in a bare pipe/aggregate, with no flow).

    Unlike :meth:`~composeai.tracing.Span.set_input`/``set_output``, this
    does *not* check :func:`~composeai.tracing.content_capture_enabled`
    (``COMPOSE_TRACE_CONTENT``): the full conversation -- including every
    tool call's arguments and results -- is written here unconditionally,
    because durable pause/resume needs it to reconstruct the in-progress
    turn exactly. ``COMPOSE_TRACE_CONTENT=0`` only ever governed *span*
    payloads (observability), never this table (see the README's
    ``COMPOSE_DIR``/``COMPOSE_TRACE_CONTENT`` note) -- making resume work
    with it disabled would require a different mechanism entirely (e.g.
    encrypting the snapshot at rest), not attempted here.
    """
    if run_id is None:
        return
    store = runs.open_default()
    store.agent_state_put(
        run_id=run_id,
        scope_key=scope_key,
        messages_json=json.dumps(to_jsonable(conversation)),
        partial_results_json=json.dumps(to_jsonable(partial_results)),
        turn=turn,
        updated_at=time.time(),
    )


# --- standalone agent call args (Phase 8: resumable before any snapshot exists) --


def _encode_agent_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return json.dumps(to_jsonable({"args": list(args), "kwargs": kwargs}))


def _decode_agent_call(args_json: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    payload = from_jsonable(json.loads(args_json))
    return tuple(payload["args"]), payload["kwargs"]


# --- the loop ------------------------------------------------------------


def _run_agent(
    agent_fn: AgentFunction,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    *,
    streaming: bool = False,
    budget: Budget | None = None,
) -> Run:
    # --- Phase 7 (durable flows): minimal, well-marked touch -----------------
    # Inside an active @flow body, the whole agent run auto-journals as one
    # step (see composeai.runs.RunContext) instead of running the loop below
    # directly. Outside any flow, a *trace-root* agent.run() (no enclosing
    # span -- i.e. not nested in a pipe/aggregate/flow) also gets a durable
    # `runs` row (kind "agent") for the CLI to list; nested agent calls don't.
    # Both branches delegate straight back to `_run_agent_uncached` below,
    # which is the original, unmodified loop plus the Phase 8 pause/resume
    # additions (`scope_key` addresses its `agent_state` snapshot: "" for a
    # standalone run, the flow step key for one nested in a flow).
    ctx = runs.current_run_context()
    if ctx is not None:
        return _run_agent_journaled(
            agent_fn, call_args, call_kwargs, ctx, streaming=streaming, budget=budget
        )
    if tracing.current_span() is None:
        args_json = _encode_agent_call(call_args, call_kwargs)
        return runs.run_standalone_agent(
            agent_fn.name,
            args_json,
            lambda: _run_agent_uncached(
                agent_fn, call_args, call_kwargs, streaming=streaming, budget=budget, scope_key=""
            ),
            budget=budget,
        )
    return _run_agent_uncached(
        agent_fn, call_args, call_kwargs, streaming=streaming, budget=budget, scope_key=""
    )


def _run_agent_journaled(
    agent_fn: AgentFunction,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    ctx: runs.JournalScope,
    *,
    streaming: bool,
    budget: Budget | None,
) -> Run:
    """Treat one whole agent run as a single journaled flow step.

    Replay: a journal hit returns a completed ``Run`` built straight from
    the decoded stored output -- no model call, ``usage=Usage()`` (replay
    costs nothing), a single ``replayed=True`` agent span, ``messages=[]``.
    Miss: run the loop for real (via ``_run_agent_uncached``, addressing its
    ``agent_state`` snapshot with this step's own journal key as
    ``scope_key``) and journal its output value. If the agent pauses,
    ``_run_agent_uncached`` raises ``_Pause`` -- deliberately not caught
    here, so it propagates to the enclosing flow's own pause handling (see
    ``composeai.flow._execute_flow``); nothing is journaled for this step
    (a miss again next time), and the agent's own state is already saved
    under ``(run_id, key)`` for when this same call site is reached again.
    """
    key = ctx.next_key(agent_fn.name)
    hit, value = ctx.journal_lookup(key)
    if hit:
        with tracing.span("agent", agent_fn.name, attributes={"step_key": key}) as agent_span:
            agent_span.replayed = True
            agent_span.set_output(value)
        trace = tracing.current_trace()
        assert trace is not None
        return Run(
            id=new_ulid(),
            status="completed",
            output=value,
            usage=Usage(),
            trace=trace,
            messages=[],
            pending=None,
        )
    run = _run_agent_uncached(
        agent_fn,
        call_args,
        call_kwargs,
        streaming=streaming,
        budget=budget,
        scope_key=key,
        step_key=key,
    )
    recorded = ctx.journal_record(key, run.output)
    run.output = recorded
    return run


def resume_standalone_agent(
    run_id: str,
    row: dict[str, Any],
    answers: dict[str, Any] | None = None,
    budget: Budget | None = None,
) -> Run:
    """Resume a paused (or crashed) standalone ``@agent`` run.

    Called by ``composeai.flow.resume()`` when the run row's ``kind`` is
    ``"agent"``. Looks the agent up by name in :data:`_AGENT_REGISTRY`,
    restores its saved conversation from ``agent_state`` (falling back to
    the originally-encoded call args if no snapshot exists yet -- only
    possible if the process died before the very first turn boundary), and
    continues the loop under the *same* ``trace_id`` as the original run --
    with the ``budget`` the original ``.run(budget=...)`` call used, decoded
    from the row (see ``composeai.runs.encode_budget``/``decode_budget``),
    *unless* this call's own ``budget=`` argument overrides it: an explicit
    override here is persisted back onto the run row (``store.update_run``,
    last-write-wins -- a later resume without an override then sees the new
    budget) before the loop resumes, and either way the cap is applied on
    top of a ``baseline`` of spend from earlier attempts (see
    ``RunStore.prior_llm_usage``) so it bounds the run's lifetime spend, not
    just this attempt's.

    ``answers`` are journaled (see
    :func:`~composeai.runs.apply_resume_answers`) only *after* the checks
    below that could otherwise abort the resume without re-executing
    anything (already completed; agent not registered) -- same reasoning as
    ``composeai.flow.resume``'s flow path: an answer given alongside a
    resume attempt that turns out to be impossible must never be
    permanently locked in.

    Unlike ``composeai.flow.resume``'s flow path, there is no fingerprint/
    staleness check here: nothing compares the agent's current definition
    (tools, ``requires_approval`` flags, system prompt, ...) against
    whatever it looked like when the run paused. If a standalone agent's
    definition changes in between -- a tool's ``requires_approval`` is
    removed, a tool is renamed/deleted, the docstring changes -- this
    simply continues with whatever is *currently* registered under
    ``row["name"]``, silently: a previously-pending approval for a tool
    that no longer requires it could execute unmediated, or a resumed call
    to a since-removed tool could produce an "unknown tool" result instead
    of a clear "code changed since this run started" error the way a
    ``@flow`` resume would raise ``ResumeMismatchError``. Not fixed here --
    ``@flow``'s fingerprint is computed from its own source text
    (``composeai.flow._compute_fingerprint``); giving a standalone agent an
    equivalent would need its own design (agents have no single function
    body to hash the way a flow does -- system prompt, tool set, and each
    tool's own definition would all need to factor in).
    """
    if row["status"] == "completed":
        output = from_jsonable(json.loads(row["output_json"])) if row["output_json"] else None
        trace = tracing.Trace(trace_id=row["trace_id"] or "")
        return Run(
            id=run_id,
            status="completed",
            output=output,
            usage=Usage(),
            trace=trace,
            messages=[],
            pending=None,
        )

    agent_fn = _AGENT_REGISTRY.get(row["name"])
    if agent_fn is None:
        raise ConfigError(
            f"agent {row['name']!r} is not registered in this process -- import the "
            "module that defines it (so its @agent decoration runs) before calling resume()"
        )

    store = runs.open_default()
    if budget is not None:
        store.update_run(
            run_id, budget_json=runs.encode_budget(budget), updated_at=time.time()
        )
    runs.apply_resume_answers(store, run_id, answers)
    call_args, call_kwargs = (
        _decode_agent_call(row["args_json"]) if row["args_json"] else ((), {})
    )
    if budget is None:
        budget = runs.decode_budget(row.get("budget_json"))
    budget_baseline = store.prior_llm_usage(run_id) if budget is not None else None
    trace_id = row["trace_id"] or new_ulid()
    with runs.use_run_id(run_id), tracing.use_trace(trace_id):
        return runs.settle_agent_run(
            store,
            run_id,
            lambda: _run_agent_uncached(
                agent_fn,
                call_args,
                call_kwargs,
                budget=budget,
                budget_baseline=budget_baseline,
                scope_key="",
            ),
        )


def _run_agent_uncached(
    agent_fn: AgentFunction,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    *,
    streaming: bool = False,
    budget: Budget | None = None,
    budget_baseline: Usage | None = None,
    scope_key: str = "",
    step_key: str | None = None,
) -> Run:
    """The actual agent loop -- shared by standalone runs, resumed standalone
    runs, and agents nested in a flow (see the three callers above).

    ``run_id``/``scope_key`` (Phase 8) address this call's ``agent_state``
    row: ``runs.current_run_id()`` is the flow's run_id when nested, or this
    call's own durable row id when standalone (set by
    ``runs.use_run_id`` either way) -- ``None`` only for an agent with no
    durable run at all (e.g. nested directly in a bare pipe/aggregate), in
    which case snapshotting/restoring are both no-ops. If a saved snapshot
    exists, its conversation is restored (never re-deriving the prompt via
    ``_build_conversation``) and, if it was paused mid-tool-batch (its last
    message is the assistant's own tool-call message), that batch is
    re-processed first -- seeded with whatever results already completed --
    before the loop continues into its next turn.

    ``step_key`` (Phase 7/10) is set only by ``_run_agent_journaled``'s miss
    path, to tag this call's ``agent`` span with the same ``step_key``
    attribute its replay path already carries -- parity between the two, for
    anything (e.g. ``compose trace``) that reads it off the span.
    """
    run_id = runs.current_run_id()
    restored = _load_agent_state(run_id, scope_key)

    span_attributes = {"step_key": step_key} if step_key is not None else None
    agent_span: tracing.Span | None = None
    try:
        with tracing.span(
            "agent", agent_fn.name, attributes=span_attributes
        ) as agent_span, budget_scope(budget, agent_span, baseline=budget_baseline):
            state = _RunState(_make_slot(agent_fn._model, cache=agent_fn._cache))
            tool_specs = [t.spec for t in agent_fn._tools] or None
            # `start_time` is *this call's* clock, not the run's -- there is
            # no persisted/restored elapsed-time field anywhere (not in
            # `_RestoredAgentState`, not in the `agent_state` table), so a
            # resume after a pause always gets a full fresh `timeout` budget
            # from the moment of resume, regardless of how long the original
            # attempt(s) already ran before pausing. A `timeout=60` agent
            # that pauses for human approval and is resumed an hour later
            # effectively gets another 60s, not "60s total across the whole
            # run" -- documented here since it's the surprising part of an
            # otherwise-accurate docstring on `agent()`'s `timeout` param.
            start_time = time.monotonic()
            output: Any = None

            if restored is not None:
                conversation = restored.conversation
                turn = restored.turn
                last = conversation[-1] if conversation else None
                pending_calls = (
                    [p for p in last.parts if isinstance(p, ToolCallPart)]
                    if last is not None and last.role == "assistant"
                    else []
                )
                if pending_calls:
                    tool_message = _process_tool_use(
                        agent_fn,
                        pending_calls,
                        conversation,
                        turn,
                        run_id,
                        scope_key,
                        seed_results=restored.partial_results,
                    )
                    conversation.append(tool_message)
                    _snapshot_agent_state(run_id, scope_key, conversation, turn, {})
            else:
                conversation = _build_conversation(agent_fn, call_args, call_kwargs)
                turn = 0

            repairs_used = 0

            while True:
                turn += 1
                if turn > agent_fn._max_turns:
                    raise MaxTurnsExceededError(
                        f"@agent {agent_fn.name!r} exceeded max_turns={agent_fn._max_turns} "
                        "(it kept calling tools/turns without finishing; raise max_turns "
                        "if this agent legitimately needs more)"
                    )
                elapsed = time.monotonic() - start_time
                if agent_fn._timeout is not None and elapsed > agent_fn._timeout:
                    raise AgentTimeoutError(
                        f"@agent {agent_fn.name!r} exceeded timeout={agent_fn._timeout}s "
                        "(checked at turn boundaries only; an in-flight model call is "
                        "never interrupted)"
                    )

                response = _perform_turn(
                    agent_fn,
                    agent_span,
                    state,
                    agent_fn._system,
                    conversation,
                    tool_specs,
                    agent_fn._output_schema,
                    streaming,
                )
                conversation.append(response.message)

                if response.stop_reason == StopReason.TOOL_USE:
                    calls = [
                        part for part in response.message.parts if isinstance(part, ToolCallPart)
                    ]
                    if not calls:
                        raise ComposeError(
                            f"@agent {agent_fn.name!r}: the model returned "
                            "stop_reason=TOOL_USE but the response contained no "
                            "tool call parts (a provider/adapter bug) -- refusing "
                            "to append an empty tool-results message"
                        )
                    tool_message = _process_tool_use(
                        agent_fn, calls, conversation, turn, run_id, scope_key
                    )
                    conversation.append(tool_message)
                    _snapshot_agent_state(run_id, scope_key, conversation, turn, {})
                    continue

                if response.stop_reason == StopReason.MAX_TOKENS:
                    reasoning_hint = ""
                    if response.usage.reasoning_tokens:
                        reasoning_hint = (
                            f" (it spent {response.usage.reasoning_tokens} tokens "
                            "on internal reasoning before any visible output -- "
                            "reasoning models can need a much larger max_tokens)"
                        )
                    raise ComposeError(
                        f"@agent {agent_fn.name!r} hit max_tokens={agent_fn._max_tokens} "
                        "before finishing its response; raise max_tokens to give it "
                        "more room" + reasoning_hint
                    )

                if response.stop_reason == StopReason.REFUSAL:
                    raise ModelRefusalError(
                        f"@agent {agent_fn.name!r}: the model refused to respond",
                        raw=response.raw_stop_reason,
                    )

                if response.stop_reason == StopReason.END_TURN:
                    try:
                        output = _extract_output(agent_fn, response)
                    except ComposeError as exc:
                        # Repair turn: give the model its own validation error
                        # and re-ask within the same conversation. Bounded by
                        # max_repairs, and each pass still runs through the
                        # max_turns / timeout checks at the top of the loop.
                        if repairs_used >= agent_fn._max_repairs:
                            raise
                        repairs_used += 1
                        agent_span.attributes.setdefault("repairs", []).append(str(exc))
                        conversation.append(
                            Message.user(
                                "Your previous reply did not match the required "
                                f"output schema. Error: {exc}\n"
                                "Reply again with ONLY a corrected JSON object "
                                "matching the schema -- no markdown fences, no "
                                "commentary."
                            )
                        )
                        continue
                    break

                # Only StopReason.OTHER reaches here in practice (TOOL_USE,
                # MAX_TOKENS, REFUSAL, END_TURN are all handled above; ERROR
                # is converted into a retried/fallback-eligible ProviderError
                # inside `_call_llm`, so it never comes back as a response at
                # all) -- terminal, but the message includes the provider's
                # own raw stop-reason string so it's actionable rather than
                # just naming the normalized enum member.
                raise ComposeError(
                    f"@agent {agent_fn.name!r}: unhandled stop_reason {response.stop_reason!r} "
                    f"(raw={response.raw_stop_reason!r})"
                )

            trace = tracing.current_trace()
            assert trace is not None  # we're inside an active span, so a trace exists
            usage = trace.rollup_usage(agent_span)

            run = Run(
                id=new_ulid(),
                status="completed",
                output=output,
                usage=usage,
                trace=trace,
                messages=conversation,
                pending=None,
            )
    except BaseException as exc:
        # `agent_span` is always set by the time an exception of interest
        # here is raised (it can only fail before assignment on a stdlib
        # failure inside `tracing.span` itself, which isn't a realistic
        # case) -- guarded anyway so this can never itself raise a NameError.
        if agent_span is not None:
            # Phase 8: a pause is not a failure -- see composeai.hitl._Pause and
            # the matching duck-typed check in composeai.runs/composeai.flow.
            if getattr(exc, "_compose_pause", False):
                tracing.emit_run_finished(agent_span, status="paused")
            else:
                tracing.emit_run_finished(
                    agent_span, status="failed", error_type=type(exc).__name__
                )
        raise

    tracing.emit_run_finished(agent_span, status="completed")
    return run


# --- streaming -------------------------------------------------------------


def _stream_run(run_thunk: Callable[[], Run]) -> RunStream:
    """Run ``run_thunk`` on a background thread against a fresh, private bus.

    The generic engine behind every ``.stream(...)`` in composeai --
    ``AgentFunction.stream`` and (from Phase 6 on) ``Pipeline.stream``/
    ``Aggregate.stream`` all build a zero-arg thunk that runs their own
    loop and produces a :class:`~composeai.runs.Run`, then hand it here.
    The thread runs inside ``contextvars.copy_context()`` taken from this
    (the caller's) thread -- via :func:`~composeai.tracing.propagate` -- so
    an enclosing span context propagates, with the new bus installed via
    :func:`~composeai.events.use_bus` inside that copied context.
    """
    bus = events.EventBus()
    subscription = bus.subscribe()

    def _worker() -> None:
        with events.use_bus(bus):
            # Caught, not swallowed: stashed on run_stream so both iteration
            # and `.run` re-raise this exact exception object (never printed
            # or lost here -- the background thread just has nowhere to
            # propagate it to).
            try:
                run = run_thunk()
            except BaseException as exc:
                run_stream._set_outcome(run=None, exception=exc)
            else:
                run_stream._set_outcome(run=run, exception=None)
            finally:
                subscription.close()

    thread = threading.Thread(target=tracing.propagate(_worker), daemon=True)
    run_stream = RunStream(subscription, thread)
    thread.start()
    return run_stream


def _stream_agent(
    agent_fn: AgentFunction,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    *,
    budget: Budget | None = None,
) -> RunStream:
    return _stream_run(
        lambda: _run_agent(agent_fn, call_args, call_kwargs, streaming=True, budget=budget)
    )
