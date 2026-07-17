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

import asyncio
import inspect
import json
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, overload

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ParamSpec, TypeVar

from . import _runtime, events, runs, tracing
from ._dispatch import _run_sync_on_own_thread, gather_settled, run_stage
from ._encoding import from_jsonable, to_jsonable
from ._ids import new_ulid
from ._schema import register_annotation_types, resolve_annotations, seal_schema
from ._storeasync import worker_for
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
from .runs import AsyncRunStream, Budget, Run, RunStream, budget_scope, check_budgets
from .tools import Tool

if TYPE_CHECKING:
    from .combinators import Pipeline, Stage

# --- typing: `AgentFunction[P, R]` generics (v0.5.0 Plan B, Task 4) -----------
#
# `P`/`R` carry PEP 696 defaults (via `typing_extensions`, same convention as
# `runs.R` -- Task 1) so a *bare* `AgentFunction` annotation means
# `AgentFunction[..., Any]` rather than tripping `reportMissingTypeArgument` if
# strict pyright is ever turned on (probed: pyright 1.1.411 accepts
# `ParamSpec("P", default=...)` on pythonVersion 3.10). `P` captures the
# decorated function's whole parameter list -- names included -- so a keyword
# call (`extract(text=...)`) type-checks, and `R` is its return
# (structured-output) type. `X`/`R2`/`NewOut`/`NewIn` are method-scoped: solved
# only from a `>>`/`<<` operand at the call site (the Variant-A single-arg
# `__rshift__` self-type trick, see its docstring), never left bare -- no
# `default=` needed.
P = ParamSpec("P", default=...)
R = TypeVar("R", default=Any)
X = TypeVar("X")
R2 = TypeVar("R2")
NewOut = TypeVar("NewOut")
NewIn = TypeVar("NewIn")

# --- agent registry (Phase 8: resume() routing) ------------------------------
#
# Mirrors `composeai.flow._FLOW_REGISTRY`: `resume()` looks a paused/crashed
# standalone agent run up by name here (see `resume_standalone_agent` below).
# Duplicate names raise `ConfigError` at decoration time, same as `@flow`.
_AGENT_REGISTRY: dict[str, AgentFunction] = {}


class AgentFunction(Generic[P, R]):
    """The callable object produced by ``@agent``.

    ``agent_fn(...)`` is sugar for ``agent_fn.run(...).output``; ``.run(...)``
    runs the full loop and returns a :class:`~composeai.runs.Run`. Both
    accept an optional keyword-only ``budget``: a :class:`~composeai.runs.Budget`
    enforced across every LLM call in the run (see
    :func:`~composeai.runs.check_budgets`). ``.stream(...)`` runs the same
    loop on a background thread and returns a :class:`~composeai.runs.RunStream`
    for live consumption. ``.arun(...)``/``.astream(...)`` (v0.4.0 Plan B)
    are their asyncio-native twins: the agent LOOP itself runs entirely on
    the CALLER's own running event loop (never the composeai runtime loop)
    and is safe to ``await``/gather concurrently from it -- but that's the
    loop, not everything it touches. A sync ``@tool`` call, a nested
    sync-bodied ``@flow`` call, or a sync-only model adapter the loop
    invokes along the way still dispatches onto its own dedicated stage
    worker thread (see ``_dispatch.run_stage``), and the durable row write
    goes through the store's own dedicated writer thread
    (:mod:`composeai._storeasync`) -- neither is the composeai runtime
    thread, but neither is "no background thread at all" either.
    ``.name``, ``.input_type``, and ``.output_type`` support introspection
    (e.g. by ``composeai.combinators.pipe()``).
    """

    def __init__(
        self,
        fn: Callable[P, R],
        *,
        model: str | Model,
        tools: Sequence[Tool],
        temperature: float | None,
        max_tokens: int,
        max_turns: int | None,
        retries: int,
        max_repairs: int,
        timeout: float | None,
        prompt_cache: bool,
        thinking: bool | None,
        effort: str | None,
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
        self._prompt_cache = prompt_cache
        self._thinking = thinking
        self._effort = effort
        self._fallback = fallback
        self._cache = cache
        self._output_schema, self._wrap_result = _build_output_schema(self.output_type)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.run(*args, **kwargs).output

    # Variant A (v0.5.0 Plan B, Task 4): the `self: AgentFunction[[X], R2]`
    # self-type means only a SINGLE-positional-arg agent may compose with
    # `>>` -- a 2-param (or 0-param) agent is a hard static `reportOperatorIssue`
    # (see prototype/probe_variants.py). `X` is the agent's input, `R2` its
    # output, so `a >> b` types the chain end-to-end as `Pipeline[X, NewOut]`.
    # `**kwargs`-only annotations are the only change: the runtime body is
    # byte-identical (still `_rshift_pipe`), and `pipe()` stays the single owner
    # of composition-time type checking.
    def __rshift__(self: AgentFunction[[X], R2], other: Stage[R2, NewOut]) -> Pipeline[X, NewOut]:
        # Local import: composeai.combinators already imports AgentFunction
        # from this module at module scope, so importing it back at module
        # scope here would be circular -- deferring to call time avoids it.
        from .combinators import _rshift_pipe

        return _rshift_pipe(self, other)

    def __rrshift__(self: AgentFunction[[X], R2], other: Stage[NewIn, X]) -> Pipeline[NewIn, R2]:
        from .combinators import _rshift_pipe

        return _rshift_pipe(other, self)

    def run(self, *args: Any, budget: Budget | None = None, **kwargs: Any) -> Run[R]:
        return _run_agent(self, args, kwargs, budget=budget)

    async def arun(self, *args: Any, budget: Budget | None = None, **kwargs: Any) -> Run[R]:
        return await _arun_agent_on_callers_loop(self, args, kwargs, budget=budget)

    def stream(self, *args: Any, budget: Budget | None = None, **kwargs: Any) -> RunStream[R]:
        return _stream_agent(self, args, kwargs, budget=budget)

    def astream(self, *args: Any, budget: Budget | None = None, **kwargs: Any) -> AsyncRunStream[R]:
        return _astream_agent(self, args, kwargs, budget=budget)


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


@overload
def agent(fn: Callable[P, R], /) -> AgentFunction[P, R]: ...
@overload
def agent(
    *,
    model: str | Model,
    tools: Sequence[Tool] = (),
    temperature: float | None = None,
    max_tokens: int = 16000,
    max_turns: int | None = 10,
    retries: int = 0,
    max_repairs: int = 0,
    timeout: float | None = None,
    prompt_cache: bool = True,
    thinking: bool | None = None,
    effort: str | None = None,
    fallback: str | Model | None = None,
    cache: bool = False,
    name: str | None = None,
    replace: bool = False,
) -> Callable[[Callable[P, R]], AgentFunction[P, R]]: ...
def agent(
    fn: Callable[..., Any] | None = None,
    *,
    model: Any = None,
    tools: Sequence[Tool] = (),
    temperature: float | None = None,
    max_tokens: int = 16000,
    max_turns: int | None = 10,
    retries: int = 0,
    max_repairs: int = 0,
    timeout: float | None = None,
    prompt_cache: bool = True,
    thinking: bool | None = None,
    effort: str | None = None,
    fallback: str | Model | None = None,
    cache: bool = False,
    name: str | None = None,
    replace: bool = False,
) -> Any:
    """Decorate a function into an :class:`AgentFunction` -- a typed, runnable agent.

    Usable bare (``@agent``) or with keyword arguments (``@agent(model=...)``)
    -- the two-overload dual form (v0.5.0 Plan B, Task 4) mirrors ``@flow``/
    ``@task``. The bare form infers ``AgentFunction[P, R]`` straight from the
    decorated function's own signature (``P`` its parameters -- names included,
    so keyword calls type-check -- and ``R`` its return/structured-output
    type); a single-positional-arg agent then composes with ``>>`` (see
    :meth:`AgentFunction.__rshift__`). ``model`` is still *required* in the
    parenthesized form (the second overload) and only defaults in the loosely
    typed implementation, which the bare form routes through -- model
    resolution stays lazy (at run time, via ``composeai.models.registry``), so
    a bare-decorated agent that never sets a model only errors when actually
    run.

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

    ``prompt_cache=True`` (the default) marks cacheable prefix spans on
    providers with explicit cache control (Anthropic: a breakpoint on the
    system prompt plus one on the conversation tail once multi-turn) --
    up to ~90% cheaper cached reads in tool loops and fan-outs, at a
    ~1.25x write premium on the first send of a span. A no-op elsewhere
    (OpenAI caches automatically server-side). Set ``prompt_cache=False``
    to send byte-identical requests to 0.5.x. Distinct from ``cache=``
    (the local response cache above): ``prompt_cache`` is provider-side
    billing config and never affects response content or request hashes.

    ``thinking``/``effort`` opt into reasoning config; ``None`` (default)
    sends nothing so each model's own default applies. ``thinking=True``
    requests adaptive thinking (with summarized display so
    ``thinking_delta`` events carry text); ``False`` explicitly disables
    it. ``effort`` is a provider-defined string passed through verbatim
    (Anthropic: ``"low"|"medium"|"high"|"xhigh"|"max"``; OpenAI:
    ``"minimal"|"low"|"medium"|"high"``). Invalid combinations surface as
    ``ProviderError`` from the provider, by design (composeai keeps no
    per-model capability table).

    ``max_turns=None`` removes the turn ceiling entirely -- pair it with
    ``timeout=`` or a run ``Budget`` so something else bounds the loop.

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

    def decorator(f: Callable[..., Any]) -> AgentFunction:
        return AgentFunction(
            f,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
            retries=retries,
            max_repairs=max_repairs,
            timeout=timeout,
            prompt_cache=prompt_cache,
            thinking=thinking,
            effort=effort,
            fallback=fallback,
            cache=cache,
            name=name,
            replace=replace,
        )

    # Bare `@agent` (fn passed positionally) applies the decorator immediately;
    # `@agent(...)` (fn is None) returns it to be applied to the function next.
    if fn is not None:
        return decorator(fn)
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


def _validate_body_result(agent_fn: AgentFunction, body_result: Any) -> list[Message]:
    """Shared str/``list[Message]`` validation for an ``@agent`` body's return value.

    Factored out (v0.4.0 Plan B, Task 6) so :func:`_abuild_conversation` --
    the one conversation builder every engine entry point now uses, sync or
    async (see its own docstring) -- enforces this contract in exactly one
    place, without duplicating the check (and its error message) anywhere
    else.
    """
    if isinstance(body_result, str):
        return [Message.user(body_result)]
    if isinstance(body_result, list) and all(isinstance(item, Message) for item in body_result):
        return list(body_result)
    raise ConfigError(
        f"@agent function {agent_fn.name!r} must return a str (the user prompt) "
        "or a list[Message] (the full conversation) from its body, got "
        f"{type(body_result)!r}"
    )


async def _abuild_conversation(
    agent_fn: AgentFunction, call_args: tuple[Any, ...], call_kwargs: dict[str, Any]
) -> list[Message]:
    """Build the conversation an ``@agent`` body produces (v0.4.0 Plan B, Task 6).

    Calls the agent body, then validates its return value via
    :func:`_validate_body_result`; if the body is itself an ``async def``
    function, calling it returns a coroutine rather than the final
    str/``list[Message]`` -- detected here via ``inspect.iscoroutine`` (the
    already-called return *value*, not ``iscoroutinefunction``, which tests
    the function object itself) -- and that coroutine is awaited before
    validation. A sync body's plain str/``list[Message]`` return passes
    straight through unchanged, so this builder handles both body kinds
    uniformly.

    :func:`_arun_agent_uncached` (the one async engine core, shared by
    every entry point -- standalone, journaled-in-flow, resumed, streamed)
    calls this builder exclusively -- so a SYNC-facade call
    (``agent_fn(...)``/``.run(...)``, which drives this same async engine
    via ``_runtime.run_sync``, see ``_run_agent_uncached``'s docstring)
    reaches an async-bodied agent correctly too: the engine underneath is
    async regardless of which facade the caller used.
    """
    body_result = agent_fn._fn(*call_args, **call_kwargs)
    if inspect.iscoroutine(body_result):
        body_result = await body_result
    return _validate_body_result(agent_fn, body_result)


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


# Sentinel for `_ainvoke_model`'s own-thread `next()` hops (see
# `_dispatch._run_sync_on_own_thread`) over a sync `stream()` iterator --
# `None` can't be used because a legitimate `RawStreamEvent` is never `None`
# but we still need a value that can never collide with one, and
# `next(iterator, default)` needs *a* default to avoid `StopIteration`
# (which can't propagate through a thread boundary cleanly).
_STREAM_SENTINEL = object()


async def _ainvoke_model(
    slot: _ModelSlot, request: ModelRequest, streaming: bool, llm_span: tracing.Span
) -> ModelResponse:
    """Call ``slot.model`` for one request, streaming deltas onto the bus if possible.

    When ``streaming`` is true and the model has an ``astream``/``stream``
    method (an optional extension -- see
    :class:`~composeai.models.base.Model`), drives it and forwards each
    :class:`~composeai.models.base.RawStreamEvent` onto the ambient event
    bus as an :class:`~composeai.events.Event` (a no-op when there's no
    ambient bus), using the model's final ``response_done`` event as the
    result. Otherwise this is ``await slot.model.acomplete(request)`` when
    the model exposes it, or ``slot.model.complete(request)`` run off the
    event loop on its own dedicated thread when it doesn't (see
    :func:`~composeai._dispatch._run_sync_on_own_thread`).

    Model invocation prefers the model's ``acomplete``/``astream`` when it
    exposes them (the duck-discovery rule); otherwise it falls back to
    running the sync ``complete``/``stream`` off the event loop via
    :func:`~composeai._dispatch._run_sync_on_own_thread` -- deliberately NOT
    ``asyncio.to_thread``, whose shared, bounded default executor a custom
    sync ``Model`` that re-enters a composeai facade (e.g. calls another
    ``@agent`` synchronously from inside ``complete()``) could starve at
    fan-out >= pool size, same class of deadlock ``run_stage``'s module
    docstring documents for sync stage bodies. A sync ``stream()`` is an
    ordinary (blocking) iterator, so it's consumed via repeated
    ``_run_sync_on_own_thread(next, (iterator, sentinel), {})`` hops -- one
    dedicated thread per event -- rather than a single call, so control
    returns to the loop between events instead of draining the whole stream
    off-loop in one go. Event handling (the ``events.emit`` call for every
    non-``response_done`` event) is factored into :func:`_forward_raw_event`
    and shared verbatim by both branches.
    """
    if streaming:
        astream_fn = getattr(slot.model, "astream", None)
        stream_fn = getattr(slot.model, "stream", None) if astream_fn is None else None
    else:
        astream_fn = None
        stream_fn = None

    if astream_fn is None and stream_fn is None:
        acomplete_fn = getattr(slot.model, "acomplete", None)
        if acomplete_fn is not None:
            return await acomplete_fn(request)

        def _complete_sync() -> ModelResponse:
            response = slot.model.complete(request)
            # `CachingModel.last_was_hit` (composeai.testing) is backed by a
            # `contextvars.ContextVar`, not a plain thread-local -- but that
            # still doesn't make the flag visible back on the caller's own
            # context here: `_run_sync_on_own_thread` propagates contextvars
            # onto this dedicated thread by running `fn` inside a *copy* of
            # the caller's context (`contextvars.copy_context()`), and a
            # `.set()` made inside that copy while `fn` runs never writes
            # back to the original context the awaiting coroutine resumes
            # in. So if this read instead happened in `_acall_llm`, on the
            # runtime loop thread, right after this returns, it would still
            # observe the ContextVar's untouched default (`False`), not the
            # value `complete()` just set on this thread's copied context.
            # Tag the span here instead, on the very thread (and context
            # copy) the call ran on, while the flag is still visible;
            # `_acall_llm`'s own (now-redundant for this path, still correct
            # for a real `acomplete`) check is left as-is.
            if getattr(slot.model, "last_was_hit", False):
                llm_span.attributes["cached"] = True
            return response

        # NOT asyncio.to_thread: that helper hands the call to the loop's
        # shared, bounded default executor (min(32, cpu+4) threads) -- a
        # custom sync Model.complete() that itself re-enters a composeai
        # facade (e.g. calls another @agent synchronously) would pin one of
        # those slots while the nested call needs a slot of the *same* pool,
        # starving forever at fan-out >= pool size; otherwise it's just a
        # hard cap on model-call throughput. A dedicated thread per call has
        # no shared bound, so nesting can never starve it -- same rationale
        # as `_dispatch.run_stage`'s module docstring for sync stage bodies.
        return await _run_sync_on_own_thread(_complete_sync, (), {})

    def _forward_raw_event(raw_event: Any) -> ModelResponse | None:
        if raw_event.kind == "response_done":
            return raw_event.response
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
        return None

    response: ModelResponse | None = None
    if astream_fn is not None:
        async for raw_event in astream_fn(request):
            result = _forward_raw_event(raw_event)
            if result is not None:
                response = result
    elif stream_fn is not None:
        iterator = stream_fn(request)
        while True:
            # NOT asyncio.to_thread, for the same reason as `_complete_sync`
            # above: its shared, bounded default executor could starve if a
            # custom sync Model.stream() re-enters a composeai facade from
            # inside its iterator. One dedicated thread per hop has no
            # shared bound to starve.
            raw_event = await _run_sync_on_own_thread(next, (iterator, _STREAM_SENTINEL), {})
            if raw_event is _STREAM_SENTINEL:
                break
            result = _forward_raw_event(raw_event)
            if result is not None:
                response = result
    if response is None:
        raise ComposeError(
            f"Model.stream() for {slot.label!r} ended without yielding a "
            "response_done event -- adapters must always yield exactly one, "
            "as their final event"
        )
    return response


async def _acall_llm(
    slot: _ModelSlot,
    system: str | None,
    messages: list[Message],
    tools: list[ToolSpec] | None,
    output_schema: dict[str, Any] | None,
    max_tokens: int,
    temperature: float | None,
    prompt_cache: bool,
    thinking: bool | None,
    effort: str | None,
    retries: int,
    streaming: bool = False,
) -> ModelResponse:
    """Call ``slot.model`` once inside its own ``llm`` span, with retry/fallback bookkeeping.

    Builds the :class:`~composeai.models.base.ModelRequest`, opens the
    ``llm`` span, and calls :func:`_ainvoke_model` -- retrying on
    ``ProviderError`` (and on a response whose own ``stop_reason`` reports
    ``ERROR``, normalized into one here) up to ``retries`` times, recording
    each attempt on the span's ``retries`` attribute. On success, records
    usage, budget-checks, and tags the span ``cached`` when the model's
    ``last_was_hit`` says so (Phase 9's ``@agent(cache=True)``).
    ``tracing.span``/``check_budgets`` stay synchronous -- neither awaits.
    """
    request = ModelRequest(
        model=slot.bare_id,
        messages=list(messages),
        system=system,
        tools=tools,
        output_schema=output_schema,
        max_tokens=max_tokens,
        temperature=temperature,
        provider=slot.provider,
        prompt_cache=prompt_cache,
        thinking=thinking,
        effort=effort,
    )
    attributes: dict[str, Any] = {"model": slot.bare_id}
    if slot.provider is not None:
        attributes["provider"] = slot.provider
    with tracing.span("llm", slot.label, attributes=attributes) as llm_span:
        llm_span.set_input({"system": system, "messages": list(messages)})
        attempt = 0
        while True:
            try:
                response = await _ainvoke_model(slot, request, streaming, llm_span)
            except ProviderError as exc:
                llm_span.attributes.setdefault("retries", []).append(
                    {"type": type(exc).__name__, "message": str(exc)}
                )
                if attempt < retries:
                    attempt += 1
                    continue
                raise
            if response.stop_reason == StopReason.ERROR:
                # A *successful* call (no exception -- `_ainvoke_model` returned
                # normally) whose own stop_reason nonetheless reports failure:
                # no adapter shipped with composeai emits this today, but a
                # custom `Model` might, and treating it as terminal would
                # silently discard the retries/fallback a user configured
                # specifically for resilience against provider failures.
                # Converting it into a `ProviderError` here reuses the exact
                # retry loop above and `_aperform_turn`'s existing fallback
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
            if getattr(slot.model, "last_was_hit", False):
                llm_span.attributes["cached"] = True
            check_budgets()
            llm_span.set_output(response.message)
            return response


async def _aperform_turn(
    agent_fn: AgentFunction,
    agent_span: tracing.Span,
    state: _RunState,
    system: str | None,
    conversation: list[Message],
    tools: list[ToolSpec] | None,
    output_schema: dict[str, Any] | None,
    streaming: bool = False,
) -> ModelResponse:
    """Run one model turn, retrying against ``agent_fn._fallback`` on a ``ProviderError``.

    Calls :func:`_acall_llm` against the currently active model slot; if
    that raises ``ProviderError`` and a fallback is configured (and hasn't
    already been switched to for this run), resolves it via ``_make_slot``,
    marks it active on ``state``, tags the enclosing agent span's
    ``fallback_used`` attribute, and retries once against the fallback.
    ``_make_slot`` (fallback model resolution) stays a plain sync call --
    it's in-memory registry/client-construction work, not I/O the loop
    needs protecting from.
    """
    try:
        return await _acall_llm(
            state.slot,
            system,
            conversation,
            tools,
            output_schema,
            agent_fn._max_tokens,
            agent_fn._temperature,
            agent_fn._prompt_cache,
            agent_fn._thinking,
            agent_fn._effort,
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
        return await _acall_llm(
            state.slot,
            system,
            conversation,
            tools,
            output_schema,
            agent_fn._max_tokens,
            agent_fn._temperature,
            agent_fn._prompt_cache,
            agent_fn._thinking,
            agent_fn._effort,
            agent_fn._retries,
            streaming,
        )


# --- tool execution -----------------------------------------------------------


class UnknownToolError(Exception):
    """Internal control-flow signal: raised inside the ``tool`` span (see
    :func:`_aexecute_one_tool`) when the model calls a name this agent has no
    ``@tool`` for, then caught right there and turned into the same
    ``is_error`` :class:`ToolResultPart` as before (content ``"unknown
    tool"``, loop continues) -- the only effect of routing it through a real
    exception is that the span records status ``"error"`` (see
    ``composeai.tracing.span``), same as a genuine tool-body failure would.
    Deliberately module-private (not part of ``composeai.errors``): it never
    escapes this function, but its name/message do show up in rendered
    traces as the span's error, so both are written for that audience.
    """


async def _aexecute_one_tool(agent_fn: AgentFunction, call: ToolCallPart) -> ToolResultPart:
    """Run one tool call, wrapped in its own ``tool`` span.

    An unknown tool name and a regular ``Exception`` raised by the tool body
    both become an ``is_error`` result (the loop continues) and both mark
    the span ``"error"`` -- only the result content differs (``"unknown
    tool"`` vs. ``f"{type}: {message}"``). A :class:`~composeai.hitl._Pause`
    (e.g. from ``ask_human()`` inside the tool body) is a
    :class:`BaseException`, so it is *not* caught here -- it propagates out
    (through the ``tool`` span, marked ``"paused"`` rather than ``"error"``
    -- see :mod:`composeai.tracing`) to whichever caller is watching for it.

    Execution itself is one call to ``_dispatch.run_stage``, which
    internally makes the ``timeout is None`` distinction explicitly (see its
    module docstring) rather than branching on it here. A separate branch
    picks WHICH bound method to hand ``run_stage`` (v0.4.0 Plan B, Task 6):
    ``tool_obj.aexecute`` for an async-bodied tool (``run_stage`` detects
    it's a coroutine function and awaits it natively, so ``timeout=``
    cancels it cooperatively rather than abandoning a daemon thread) or the
    existing ``tool_obj.execute`` otherwise.
    """
    tool_obj = agent_fn._tools_by_name.get(call.name)
    try:
        with tracing.span("tool", call.name, input=call.arguments) as tool_span:
            if tool_obj is None:
                raise UnknownToolError(
                    f"the model called tool {call.name!r} but this agent has no such tool"
                )
            run_fn = tool_obj.aexecute if tool_obj.is_async else tool_obj.execute
            content = await run_stage(
                run_fn,
                (call.arguments,),
                {},
                timeout=tool_obj.timeout,
                name=call.name,
                kind="@tool",
            )
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

    Does *not* persist anything -- ``_aprocess_tool_use`` collects every
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


async def _aprocess_tool_use(
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
       restored snapshot's already-completed calls) runs concurrently --
       each under its own ``tool:{call_id}`` journal scope (see
       ``composeai.runs``'s scope-stack module docs), so a ``@task``/
       ``@agent`` call inside a tool body gets a deterministic key
       regardless of which coroutine reaches it first. The fan-out is
       ``_dispatch.gather_settled`` over one ``_arun_one`` coroutine per
       call: each coroutine catches ``BaseException`` itself and returns a
       ``(call_id, result, exc)`` triple rather than letting it propagate,
       so ``gather_settled``'s own exception slot is never actually
       observed. ``runs.push_scope`` still wraps each call's execution:
       contextvars are task-local (a coroutine scheduled via ``gather``/
       ``gather_settled`` gets its own copy of the context at the moment
       it's scheduled), giving each concurrently dispatched call its own
       scope isolation.
    2. Approval-gated calls are then checked *in order*: answered True ->
       execute now (same scope); answered False -> a "denied by user"
       ``is_error`` result; unanswered -> build (not yet persist) its pause
       (see :func:`_pause_for_unanswered_approval`) and keep scanning the
       rest (independent calls may still resolve) -- this "keep scanning"
       rule also applies when an *answered* call's own body pauses (e.g.
       nested ``ask_human()``): it ``continue``s rather than aborting the
       batch, matching the unanswered-gate case two lines above.
    3. If anything paused, atomically persist every pending interrupt
       collected in steps 1-2 together with the ``conversation``/``turn``/
       whatever results *did* complete (so a resume doesn't redo them, and
       so a crash right after this point leaves a fully consistent,
       resumable pause -- see :meth:`~composeai.runs.RunStore.persist_pause`)
       -- routed through ``worker_for(store).call(...)`` instead of calling
       the store directly, so the write happens off the runtime loop thread
       -- and raise the *first* pause encountered, in call order. Otherwise,
       merge every result into one batched user message, in the calls'
       original order.

    ``runs.interrupt_lookup`` (approval-answer lookup) and
    ``runs.push_scope`` stay direct sync calls: neither is on this task's
    conversion list (the former is a shared helper also called from the
    flow's own sync body), and the latter is pure in-memory contextvar
    manipulation, not I/O.
    """
    results: dict[str, ToolResultPart] = dict(seed_results) if seed_results else {}
    approval_ids = {call.id for call in calls if _tool_requires_approval(agent_fn, call)}
    normal_calls = [c for c in calls if c.id not in approval_ids and c.id not in results]
    approval_calls = [c for c in calls if c.id in approval_ids]

    pause_exc: _Pause | None = None
    pending_interrupts: list[Interrupt] = []

    if normal_calls:

        async def _arun_one(
            call: ToolCallPart,
        ) -> tuple[str, ToolResultPart | None, BaseException | None]:
            try:
                with runs.push_scope(f"tool:{call.id}"):
                    return call.id, await _aexecute_one_tool(agent_fn, call), None
            except BaseException as exc:  # noqa: BLE001 -- _Pause (or worse) settled below
                return call.id, None, exc

        if len(normal_calls) == 1:
            outcomes = [await _arun_one(normal_calls[0])]
        else:
            settled = await gather_settled([_arun_one(c) for c in normal_calls])
            # `_arun_one` catches BaseException itself and always returns its
            # triple normally, so `gather_settled`'s own exception slot is
            # never populated in practice -- mirrors the sync version, where
            # `f.result()` never raises from `_run_one`'s own body either.
            outcomes = [outcome for outcome, _exc in settled]

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
                    results[call.id] = await _aexecute_one_tool(agent_fn, call)
            except _Pause as exc:
                if pause_exc is None:
                    pause_exc = exc
                continue
        else:
            results[call.id] = _deny_tool_call(call)

    if pause_exc is not None:
        if run_id is not None:
            await worker_for(runs.open_default()).call(
                "persist_pause",
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


async def _aload_agent_state(run_id: str | None, scope_key: str) -> _RestoredAgentState | None:
    """Restore a run's saved ``agent_state`` snapshot, if one exists.

    ``None`` when ``run_id`` is ``None`` (no durable run to restore from --
    e.g. an agent nested directly in a bare pipe/aggregate, with no flow) or
    when no snapshot row exists yet. The store read (``agent_state_get``)
    routes through ``worker_for(store).call(...)`` instead of calling the
    store directly, so it happens off the runtime loop thread (same
    reasoning as :func:`_asnapshot_agent_state` below).
    """
    if run_id is None:
        return None
    store = runs.open_default()
    row = await worker_for(store).call("agent_state_get", run_id, scope_key)
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


async def _asnapshot_agent_state(
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

    The write (``agent_state_put``) routes through
    ``worker_for(store).call(...)`` -- the dedicated store writer thread
    (:mod:`composeai._storeasync`) -- rather than calling the store directly
    from the runtime loop thread, so one turn's snapshot write never blocks
    every other coroutine the runtime loop is concurrently driving. Call
    ordering is preserved (each ``await`` here still happens exactly where
    the sync call did), so durability timing is unchanged.
    """
    if run_id is None:
        return
    store = runs.open_default()
    await worker_for(store).call(
        "agent_state_put",
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
) -> Run[Any]:
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


async def _arun_agent(
    agent_fn: AgentFunction,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    *,
    streaming: bool = False,
    budget: Budget | None = None,
) -> Run[Any]:
    """Async twin of :func:`_run_agent` (v0.4.0 Plan A, Task 8).

    Same three-way routing, each branch calling its own async twin:
    journaled-flow-step (:func:`_arun_agent_journaled`) when nested in a
    ``@flow``, the plain uncached loop (:func:`_arun_agent_uncached`)
    otherwise. ``composeai.combinators._ainvoke_stage`` is this function's
    only caller today, and it only ever calls it from inside an ALREADY-OPEN
    span (``map()``'s own "aggregate" span, or the "pipe"/"aggregate" span
    ``_run_top`` opens before invoking any stage) -- so the
    ``tracing.current_span() is None`` trace-root-standalone-agent branch
    below is, in practice, unreachable from that caller -- mirroring the
    sync ``_run_agent``'s identical branch, equally unreachable now that
    every stage dispatch runs through :func:`~composeai.combinators._ainvoke_stage`
    (the old sync ``combinators._invoke_stage`` had the same property before
    it was removed as dead code). Kept here anyway for structural parity
    with ``_run_agent``: if it were ever somehow reached from a
    coroutine already running on the composeai runtime loop, its call to
    ``runs.run_standalone_agent`` (fully sync) would reach the sync facade
    ``_run_agent_uncached`` -- which would immediately raise
    (``_runtime.run_sync``'s re-entry guard) rather than deadlock, a loud
    failure rather than a silent hang.
    """
    ctx = runs.current_run_context()
    if ctx is not None:
        return await _arun_agent_journaled(
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
    return await _arun_agent_uncached(
        agent_fn, call_args, call_kwargs, streaming=streaming, budget=budget, scope_key=""
    )


async def _arun_agent_on_callers_loop(
    agent_fn: AgentFunction,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    *,
    streaming: bool = False,
    budget: Budget | None = None,
) -> Run[Any]:
    """The async engine behind ``AgentFunction.arun``/``.astream`` (v0.4.0 Plan B, Task 4).

    Same three-way routing as :func:`_run_agent`/:func:`_arun_agent`
    (journaled flow step -> standalone durable row -> plain uncached loop),
    but the middle branch differs on purpose: :func:`_arun_agent` above
    calls the fully-SYNC ``runs.run_standalone_agent`` in that branch,
    which is safe there ONLY because that branch is documented dead code
    for `_arun_agent` -- its one caller (``composeai.combinators``) never
    invokes it from outside an already-open span. This function, in
    contrast, is reached directly from a user's own asyncio loop as a
    genuine trace root (a bare, non-nested ``await agent_fn.arun(...)`` --
    see ``tests/test_async_surface.py``), so it calls
    :func:`~composeai.runs.arun_standalone_agent` (this same task's other
    new entry point) instead. That never touches
    :func:`~composeai._runtime.run_sync` or the composeai runtime loop at
    all -- everything below runs to completion on whichever loop the
    caller is already running, ``await``ed all the way down through
    :func:`_arun_agent_uncached`.
    """
    ctx = runs.current_run_context()
    if ctx is not None:
        return await _arun_agent_journaled(
            agent_fn, call_args, call_kwargs, ctx, streaming=streaming, budget=budget
        )
    if tracing.current_span() is None:
        args_json = _encode_agent_call(call_args, call_kwargs)
        return await runs.arun_standalone_agent(
            agent_fn.name,
            args_json,
            lambda: _arun_agent_uncached(
                agent_fn, call_args, call_kwargs, streaming=streaming, budget=budget, scope_key=""
            ),
            budget=budget,
        )
    return await _arun_agent_uncached(
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
) -> Run[Any]:
    """Treat one whole agent run as a single journaled flow step.

    Replay: a journal hit returns a completed ``Run`` built straight from
    the decoded stored output -- no model call, ``usage=Usage()`` (replay
    costs nothing), a single ``replayed=True`` agent span, ``messages=[]``.
    Miss: run the loop for real (via ``_run_agent_uncached``, addressing its
    ``agent_state`` snapshot with this step's own journal key as
    ``scope_key``) and journal its output value. If the agent pauses,
    ``_run_agent_uncached`` raises ``_Pause`` -- deliberately not caught
    here, so it propagates to the enclosing flow's own pause handling (see
    ``composeai.flow._aexecute_flow``); nothing is journaled for this step
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
    budget_baseline = (
        runs.open_default().prior_llm_usage_for_step(ctx.run_id, key)
        if budget is not None
        else None
    )
    run = _run_agent_uncached(
        agent_fn,
        call_args,
        call_kwargs,
        streaming=streaming,
        budget=budget,
        budget_baseline=budget_baseline,
        scope_key=key,
        step_key=key,
    )
    recorded = ctx.journal_record(key, run.output)
    run.output = recorded
    return run


async def _arun_agent_journaled(
    agent_fn: AgentFunction,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    ctx: runs.JournalScope,
    *,
    streaming: bool,
    budget: Budget | None,
) -> Run[Any]:
    """Async twin of :func:`_run_agent_journaled` (v0.4.0 Plan A, Task 8) --
    see its docstring for the replay/miss contract, byte-identical here.

    Two things differ from the sync version, both because this runs on the
    composeai runtime loop (inside ``composeai.combinators``'s async
    engine), never on an arbitrary caller thread:

    - The miss path's journal write goes through ``ctx.ajournal_record``
      (:meth:`~composeai.runs.RunContext.ajournal_record`) instead of
      ``ctx.journal_record`` -- it awaits
      ``worker_for(store).call("journal_put", ...)`` (the dedicated SQLite
      writer thread, :mod:`composeai._storeasync`) rather than calling the
      store directly, so this never blocks the runtime loop with SQLite
      I/O. First-write-wins is unaffected: ``RunStore.journal_put``'s
      ``INSERT OR IGNORE`` (see its docstring) is still the single atomic
      operation that decides a race's winner -- only which thread executes
      that SQL statement changes, not how the race resolves or what gets
      returned (the winner's own decoded value, same as the sync path).
    - The budget baseline read (``prior_llm_usage_for_step``) likewise
      routes through ``worker_for(...).call(...)`` rather than calling the
      store directly -- mirroring :func:`_aload_agent_state`'s identical
      reasoning for a *read* (not just writes) made inside the async
      engine.

    ``ctx.next_key``/``ctx.journal_lookup`` stay direct, synchronous calls
    (as in the sync version): both only ever touch the in-memory
    ``preloaded`` dict and a ``threading.Lock``-guarded counter, never the
    store/disk, so there is nothing to await and no risk of blocking the
    loop.
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
    budget_baseline = (
        await worker_for(runs.open_default()).call(
            "prior_llm_usage_for_step", ctx.run_id, key
        )
        if budget is not None
        else None
    )
    run = await _arun_agent_uncached(
        agent_fn,
        call_args,
        call_kwargs,
        streaming=streaming,
        budget=budget,
        budget_baseline=budget_baseline,
        scope_key=key,
        step_key=key,
    )
    recorded = await ctx.ajournal_record(key, run.output)
    run.output = recorded
    return run


def resume_standalone_agent(
    run_id: str,
    row: dict[str, Any],
    answers: dict[str, Any] | None = None,
    budget: Budget | None = None,
) -> Run[Any]:
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


async def aresume_standalone_agent(
    run_id: str,
    row: dict[str, Any],
    answers: dict[str, Any] | None = None,
    budget: Budget | None = None,
) -> Run[Any]:
    """Async twin of :func:`resume_standalone_agent` (v0.4.0 Plan B, Task 7)
    -- called by ``composeai.flow.aresume()`` when the run row's ``kind`` is
    ``"agent"``. Same contract and the same check order as the sync
    version -- read its docstring first, unchanged here (completed-row
    short-circuit, then unregistered-agent check, both BEFORE any answer is
    journaled).

    Runs natively on the CALLER's own running event loop (never
    ``_runtime.run_sync``/the composeai runtime loop): every store read/
    write this function makes itself -- the budget-override ``update_run``,
    the budget-baseline ``prior_llm_usage`` -- routes through ``await
    worker_for(store).call(...)`` instead of calling ``store`` directly,
    same reasoning as ``composeai.flow.aresume``/``composeai.runs
    .arun_standalone_agent``. ``answers`` are journaled through
    :func:`~composeai.runs.aapply_resume_answers` (the async twin of
    :func:`~composeai.runs.apply_resume_answers`) instead of the sync
    version, and the loop itself runs through :func:`_arun_agent_uncached`
    (awaited directly, never through the sync facade
    :func:`_run_agent_uncached`), settled by
    :func:`~composeai.runs.asettle_agent_run` instead of
    :func:`~composeai.runs.settle_agent_run`.
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
        await worker_for(store).call(
            "update_run", run_id, budget_json=runs.encode_budget(budget), updated_at=time.time()
        )
    await runs.aapply_resume_answers(store, run_id, answers)
    call_args, call_kwargs = (
        _decode_agent_call(row["args_json"]) if row["args_json"] else ((), {})
    )
    if budget is None:
        budget = runs.decode_budget(row.get("budget_json"))
    budget_baseline = (
        await worker_for(store).call("prior_llm_usage", run_id) if budget is not None else None
    )
    trace_id = row["trace_id"] or new_ulid()
    with runs.use_run_id(run_id), tracing.use_trace(trace_id):
        return await runs.asettle_agent_run(
            store,
            run_id,
            lambda: _arun_agent_uncached(
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
) -> Run[Any]:
    """Sync facade over the async engine (:func:`_arun_agent_uncached`) --
    see its docstring for the full loop contract.

    Every existing caller (the journaled path, the standalone path,
    ``resume_standalone_agent``, and ``.stream(...)``'s background worker
    thread) reaches the loop through this one name, unchanged -- so all of
    them are routed through the async engine with zero call-site changes.
    Legal to call from any thread, including one already running inside the
    async engine itself (e.g. a sync ``@tool`` body calling another agent
    synchronously): :func:`~composeai._runtime.run_sync` hops back onto the
    runtime loop from the dedicated per-call worker thread
    :func:`~composeai._dispatch._run_sync_on_own_thread` started for that
    sync tool body, without deadlocking, since the loop is merely awaiting
    that worker thread's future and is free to run the new submission
    concurrently (documented extra latency, not a deadlock).
    """
    return _runtime.run_sync(
        _arun_agent_uncached(
            agent_fn,
            call_args,
            call_kwargs,
            streaming=streaming,
            budget=budget,
            budget_baseline=budget_baseline,
            scope_key=scope_key,
            step_key=step_key,
        )
    )


async def _arun_agent_uncached(
    agent_fn: AgentFunction,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    *,
    streaming: bool = False,
    budget: Budget | None = None,
    budget_baseline: Usage | None = None,
    scope_key: str = "",
    step_key: str | None = None,
) -> Run[Any]:
    """The actual agent loop -- shared by standalone runs, resumed standalone
    runs, and agents nested in a flow (see the three callers of the sync
    facade :func:`_run_agent_uncached` above). Driven by that facade via
    :func:`~composeai._runtime.run_sync`, which snapshots the calling
    thread's contextvars into this coroutine's task -- so
    ``runs.current_run_id()``/``tracing.current_span()``/the scope and
    budget stacks below all see exactly what the sync caller saw. Nothing
    in this function mutates one of those *ambient* contextvars for the
    caller to observe afterward: it only ever *reads* ``current_run_id()``,
    and the one contextvar push it does perform (``budget_scope`` below) is
    ``with``-scoped entirely within this coroutine's own body, so it is
    task-local and never needs to escape back to the caller (mirrors the
    sync engine's identical scoping, unchanged).

    ``run_id``/``scope_key`` (Phase 8) address this call's ``agent_state``
    row: ``runs.current_run_id()`` is the flow's run_id when nested, or this
    call's own durable row id when standalone (set by
    ``runs.use_run_id`` either way) -- ``None`` only for an agent with no
    durable run at all (e.g. nested directly in a bare pipe/aggregate), in
    which case snapshotting/restoring are both no-ops. If a saved snapshot
    exists, its conversation is restored (never re-deriving the prompt via
    ``_abuild_conversation``) and, if it was paused mid-tool-batch (its last
    message is the assistant's own tool-call message), that batch is
    re-processed first -- seeded with whatever results already completed --
    before the loop continues into its next turn.

    ``step_key`` (Phase 7/10) is set only by ``_run_agent_journaled``'s miss
    path, to tag this call's ``agent`` span with the same ``step_key``
    attribute its replay path already carries -- parity between the two, for
    anything (e.g. ``compose trace``) that reads it off the span.
    """
    run_id = runs.current_run_id()
    restored = await _aload_agent_state(run_id, scope_key)

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
                    tool_message = await _aprocess_tool_use(
                        agent_fn,
                        pending_calls,
                        conversation,
                        turn,
                        run_id,
                        scope_key,
                        seed_results=restored.partial_results,
                    )
                    conversation.append(tool_message)
                    await _asnapshot_agent_state(run_id, scope_key, conversation, turn, {})
            else:
                conversation = await _abuild_conversation(agent_fn, call_args, call_kwargs)
                turn = 0

            repairs_used = 0

            while True:
                turn += 1
                if agent_fn._max_turns is not None and turn > agent_fn._max_turns:
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

                response = await _aperform_turn(
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
                    tool_message = await _aprocess_tool_use(
                        agent_fn, calls, conversation, turn, run_id, scope_key
                    )
                    conversation.append(tool_message)
                    await _asnapshot_agent_state(run_id, scope_key, conversation, turn, {})
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
                # inside `_acall_llm`, so it never comes back as a response at
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


def _stream_run(run_thunk: Callable[[], Run[Any]]) -> RunStream:
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


def _astream_run(run_thunk: Callable[[], Coroutine[Any, Any, Run[Any]]]) -> AsyncRunStream:
    """Async twin of :func:`_stream_run` (v0.4.0 Plan B, Task 4).

    ``run_thunk`` is a zero-arg coroutine *factory* (same contract as
    ``runs.asettle_agent_run``'s ``thunk``) -- calling it produces the
    coroutine the wrapper below awaits, exactly once. Runs the engine as an
    ``asyncio.Task`` on the CALLER's own already-running loop -- the engine
    ITSELF is not dispatched to a background thread the way
    :func:`_stream_run` dispatches it, though a sync ``@tool``/nested
    sync-bodied ``@flow`` call it makes along the way still runs on its own
    dedicated stage worker thread regardless (same caveat as
    :class:`AgentFunction`'s own docstring) -- against a fresh, private
    :class:`~composeai.events.EventBus`.

    Subscribes (``bus.asubscribe()``) *before* creating the task -- the
    same subscribe-before-emit ordering :func:`_stream_run` gets for free
    by subscribing before starting its worker thread. Creating the task
    first here could let the engine start emitting (the first
    ``span_started``/``text_delta``/...) before anything is listening,
    silently dropping those events.

    The wrapper coroutine enters :func:`~composeai.events.use_bus` (a
    plain, synchronous context manager -- the contextvar it pushes stays
    correctly scoped across every ``await`` inside this same coroutine/
    task) and closes the :class:`~composeai.events.AsyncSubscription` in a
    ``finally`` -- mirroring exactly where :func:`_stream_run`'s worker
    closes its own :class:`~composeai.events.Subscription`, so breaking out
    of ``async for`` early (or calling ``.close()``) only unsubscribes; the
    task itself always runs to completion regardless. Unlike
    :class:`~composeai.runs.RunStream`'s manual ``_set_outcome``, no
    outcome-stashing is needed here: ``asyncio.Task`` memoizes its own
    result, so ``await``ing :meth:`~composeai.runs.AsyncRunStream.run`
    (whether before, during, or after iterating) always sees the same,
    stable outcome -- re-raising the task's original exception verbatim if
    it failed.
    """
    bus = events.EventBus()
    subscription = bus.asubscribe()

    async def _wrapper() -> Run[Any]:
        with events.use_bus(bus):
            try:
                return await run_thunk()
            finally:
                subscription.close()

    task = asyncio.get_running_loop().create_task(_wrapper())
    return AsyncRunStream(task, subscription)


def _astream_agent(
    agent_fn: AgentFunction,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    *,
    budget: Budget | None = None,
) -> AsyncRunStream:
    return _astream_run(
        lambda: _arun_agent_on_callers_loop(
            agent_fn, call_args, call_kwargs, streaming=True, budget=budget
        )
    )
