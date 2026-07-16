"""``pipe``/``aggregate``/``map``: compose stages with composition-time type checking.

This is composeai's headline pitch: ``pipe(researcher, copywriter)`` chains
two ``@agent`` functions (or plain callables, or nested ``pipe``/
``aggregate`` results -- a *stage* is any of those) so the first stage's
output feeds the second stage's input. The killer feature is that ``pipe()``
checks every consecutive stage pair for type compatibility *at build time*
-- a wiring bug is a :class:`~composeai.errors.CompositionTypeError` raised
by ``pipe(...)`` itself, before any stage has run and before any API spend.
``aggregate(**branches)`` runs every branch in parallel and gathers a
``{name: output}`` dict; ``map(fn, items)`` is the same fan-out applied to
one stage over many items.

Every stage exposes ``.input_type``/``.output_type`` for this check:
``AgentFunction`` (added in ``agentfn.py``), ``Pipeline``, and ``Aggregate``
all carry them natively; a plain callable's are derived from its signature
via :func:`~composeai._schema.resolve_annotations` (missing annotations
default to ``Any``).
"""

from __future__ import annotations

import asyncio
import inspect
import time
import types
import typing
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any

from . import _runtime, agentfn, runs, tracing
from ._dispatch import gather_settled, run_stage
from ._encoding import register_serializable
from ._ids import new_ulid
from ._schema import resolve_annotations
from ._storeasync import worker_for
from .agentfn import AgentFunction
from .errors import CompositionTypeError, ConfigError
from .runs import AsyncRunStream, Budget, Run, RunStream, budget_scope
from .tools import Tool

Stage = Any
"""Anything usable as a pipe/aggregate/map stage: an :class:`~composeai.agentfn.AgentFunction`,
a :class:`Pipeline`, an :class:`Aggregate`, or a plain callable taking one
positional argument."""


# --- >> composition sugar --------------------------------------------------------


def _rshift_pipe(left: Any, right: Any) -> Pipeline:
    """Shared body for ``__rshift__``/``__rrshift__`` on every stage type.

    ``a >> b`` is exactly ``pipe(a, b)`` -- ``pipe()`` remains the single
    owner of composition-time type checking; this helper never duplicates
    any of it. Flattening keeps chains flat regardless of associativity: a
    ``Pipeline`` operand contributes its own ``_stages`` tuple rather than
    itself, so ``(a >> b) >> c`` and ``a >> (b >> c)`` both build a
    ``Pipeline`` whose ``_stages`` is ``(a, b, c)``, not a ``Pipeline``
    nested inside a ``Pipeline``.
    """

    def _flat(stage: Any) -> tuple[Any, ...]:
        return stage._stages if isinstance(stage, Pipeline) else (stage,)

    return pipe(*_flat(left), *_flat(right))


# --- type introspection --------------------------------------------------------


def _plain_callable_types(fn: Callable[..., Any]) -> tuple[Any, Any]:
    """``(input_type, output_type)`` for a plain callable stage.

    The first positional parameter's annotation and the return annotation,
    resolved via :func:`~composeai._schema.resolve_annotations`; either
    missing (or the callable's signature can't be introspected at all)
    defaults to ``Any``.
    """
    try:
        hints = resolve_annotations(fn, include_extras=True)
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return Any, Any

    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    input_type: Any = hints.get(positional[0].name, Any) if positional else Any
    output_type: Any = hints.get("return", Any)
    return input_type, output_type


def _stage_input_type(stage: Stage) -> Any:
    if isinstance(stage, (AgentFunction, Pipeline, Aggregate, Tool)):
        return stage.input_type
    return _plain_callable_types(stage)[0]


def _stage_output_type(stage: Stage) -> Any:
    if isinstance(stage, (AgentFunction, Pipeline, Aggregate, Tool)):
        return stage.output_type
    return _plain_callable_types(stage)[1]


def _stage_name(stage: Stage) -> str:
    if isinstance(stage, AgentFunction):
        return stage.name
    if isinstance(stage, (Pipeline, Aggregate)):
        return stage._name
    # Duck-typed `.name` covers @task/@tool objects (which have no __name__).
    named = getattr(stage, "name", None)
    if isinstance(named, str) and named:
        return named
    return getattr(stage, "__name__", repr(stage))


def _type_name(t: Any) -> str:
    if t is Any:
        return "Any"
    if isinstance(t, type):
        return t.__name__
    return str(t)


def _union_members(t: Any) -> tuple[Any, ...] | None:
    """``t``'s member types if it's a ``Union``/``X | Y`` (incl. ``Optional``), else ``None``."""
    origin = typing.get_origin(t)
    if origin is typing.Union or origin is types.UnionType:
        return typing.get_args(t)
    return None


def _types_compatible(out_t: Any, in_t: Any) -> bool:
    """Whether a stage returning ``out_t`` may feed a stage expecting ``in_t``.

    - Passes when either side is ``Any`` (including a missing annotation,
      already normalized to ``Any`` by the callers of this function) or
      ``object``.
    - Passes on equality -- the *only* check applied to typing generics
      such as ``list[str]``: they are compared structurally with ``==``,
      never by subclassing.
    - A ``Union``/``X | Y`` (including ``Optional``) on the *output* side
      passes only if *every* member is individually compatible with
      ``in_t`` (an agent returning ``str | int`` can't safely feed a stage
      that only accepts ``str``). A ``Union``/``Optional`` on the *input*
      side passes if *any* member accepts ``out_t`` (a stage returning
      ``str`` safely feeds one accepting ``str | None`` -- the narrower,
      concrete output is always a valid member of the wider input type).
    - Passes when both sides are plain classes and ``issubclass(out_t, in_t)``.
    - Anything else fails.
    """
    if out_t is Any or in_t is Any or out_t is object or in_t is object:
        return True
    if out_t == in_t:
        return True
    out_union = _union_members(out_t)
    if out_union is not None:
        return all(_types_compatible(member, in_t) for member in out_union)
    in_union = _union_members(in_t)
    if in_union is not None:
        return any(_types_compatible(out_t, member) for member in in_union)
    if isinstance(out_t, type) and isinstance(in_t, type):
        try:
            return issubclass(out_t, in_t)
        except TypeError:
            return False
    return False


# --- shared stage-invocation + top-level run assembly --------------------------


async def _ainvoke_stage(
    stage: Stage,
    x: Any,
    *,
    streaming: bool,
    task_name: str | None = None,
    timeout: float | None = None,
    timeout_name: str | None = None,
    timeout_kind: str = "@task",
) -> Any:
    """Run one stage on input ``x``, opening whatever span kind fits it.

    ``AgentFunction`` stages run via ``agentfn._arun_agent`` directly (not
    the public, always-non-streaming ``.run()`` sugar) so a ``streaming``
    pipeline/aggregate can thread ``streaming=True`` down to them and get
    real token deltas on the ambient bus, not just span events. Nested
    ``Pipeline``/``Aggregate`` stages open their own natural span and
    recurse with the same ``streaming`` flag. A plain callable gets a
    ``task`` span with its input/output captured; ``task_name`` overrides
    the default name (used by :func:`map` for per-item span names).

    This also owns a ``timeout``/``timeout_kind`` the old sync engine never
    took: ``map()``/``Aggregate._run_branches`` used to wrap the WHOLE
    dispatch -- sync stage or not -- in the same ``runs._run_with_timeout``
    daemon race, since everything ran synchronously in a worker thread
    regardless of stage kind.

    Threading the timeout down into *this* function -- rather than the
    caller wrapping this function's own result in a second, outer timeout
    -- is what preserves each stage kind's correct cancellation semantics
    via :func:`~composeai._dispatch.run_stage`: for a plain callable,
    ``stage`` itself (not a wrapper) is what's handed to ``run_stage``, so
    its own ``inspect.iscoroutinefunction`` check still sees the real user
    function -- a sync ``def`` still gets the abandoned-thread daemon race
    (:func:`~composeai.runs._run_with_timeout`, unchanged wording -- what
    the existing ``test_map_timeout_per_item_raises``-style tests exercise
    and must keep passing), an ``async def`` (the new, map()-internal-only
    capability) gets cooperative ``asyncio.wait_for`` cancellation. An
    ``AgentFunction``/``Pipeline``/``Aggregate`` stage has no sync body to
    preserve a thread-race for -- it's already async-native -- so it's
    wrapped in a zero-arg coroutine function and always takes
    ``run_stage``'s cooperative-cancel branch instead. (Had this function
    instead wrapped *its own* dispatch in one outer async closure regardless
    of stage kind, every stage -- including a plain sync callable -- would
    look like a coroutine function to ``run_stage``, silently losing the
    abandoned-thread semantics for the very case the existing tests cover.)

    ``task_name`` behaves exactly as in the sync version (span-naming
    override for the plain-callable branch only -- ``map()`` uses this for
    per-item span names like ``f"{fn_name}[{i}]"``; ``Pipeline``/
    ``Aggregate``/``AgentFunction`` branches always use their own inherent
    name). ``timeout_name`` is a SEPARATE identifier used only in the
    ``TaskTimeoutError`` message (defaults to whatever ``task_name``/
    ``_stage_name(stage)`` resolves to when not given): ``map()`` passes
    the same value for both (its old ``dispatch()`` used one ``name`` for
    both purposes); ``Aggregate._arun_branches`` passes only
    ``timeout_name`` (the branch's declared key, e.g. ``"hung"``) and never
    ``task_name`` -- matching the sync version, where a plain-callable
    branch's span name came from ``_stage_name(stage)`` while its timeout
    message named the branch by its dict key, two independently-meaningful
    identifiers that happened to coincide only in ``map()``.
    """
    stage_name = task_name if task_name is not None else _stage_name(stage)
    tname = timeout_name if timeout_name is not None else stage_name

    if isinstance(stage, AgentFunction):

        async def _call() -> Any:
            run = await agentfn._arun_agent(stage, (x,), {}, streaming=streaming)
            return run.output

        return await run_stage(_call, (), {}, timeout=timeout, name=tname, kind=timeout_kind)

    if isinstance(stage, Pipeline):

        async def _call() -> Any:
            with tracing.span("pipe", stage._name):
                return await stage._arun_stages(x, streaming=streaming)

        return await run_stage(_call, (), {}, timeout=timeout, name=tname, kind=timeout_kind)

    if isinstance(stage, Aggregate):

        async def _call() -> Any:
            with tracing.span("aggregate", stage._name):
                return await stage._arun_branches(x, streaming=streaming)

        return await run_stage(_call, (), {}, timeout=timeout, name=tname, kind=timeout_kind)

    with tracing.span("task", stage_name) as task_span:
        task_span.set_input(x)
        result = await run_stage(stage, (x,), {}, timeout=timeout, name=tname, kind=timeout_kind)
        task_span.set_output(result)
        return result


def _run_top(
    kind: tracing.SpanKind, name: str, budget: Budget | None, run_body: Callable[[], Any]
) -> Run:
    """Wrap a top-level ``pipe``/``aggregate`` ``.run()``/``.stream()`` call with a durable run row.

    Regression fix (build-ledger item, Phase 9): a trace-root
    ``pipe``/``aggregate`` call used to open its ``span(kind, name)`` and
    build a ``Run`` directly, with no ``runs`` table row at all -- unlike a
    standalone ``@agent`` run (:func:`~composeai.runs.run_standalone_agent`).
    With no durable row, ``current_run_id()`` stayed ``None`` for the whole
    call, so every span underneath persisted with ``run_id`` SQL NULL too
    (see :func:`~composeai.runs._default_span_sink`) -- the CLI's ``compose
    runs``/``compose trace`` had nothing to find. This mirrors
    ``run_standalone_agent`` exactly: create the row (``kind`` "pipe" or
    "aggregate"), make it ambient via ``use_run_id``/``use_trace``, and
    delegate status transitions (completed/failed/paused) to
    :func:`~composeai.runs.settle_agent_run` -- which is also what makes
    ``run.id`` equal the durable row's id, the same invariant
    ``run_standalone_agent``'s callers rely on.

    The inner ``_thunk`` is otherwise unchanged from before this fix:
    a budget scope (no-op when ``budget`` is ``None``) around
    ``run_body()``, a fresh ``Run`` with ``rollup_usage(root_span)`` and
    ``messages=[]``, and a terminal ``run_finished`` event on the ambient
    bus either way -- ``{"status": "completed"}`` on success, ``{"status":
    "failed", "error": ...}`` on a real exception.

    A pause (a nested ``@agent``'s unanswered ``approve()``/``ask_human()``
    call propagating all the way up to this trace root) does *not* return a
    paused ``Run`` the way ``run_standalone_agent`` does: a bare
    ``pipe()``/``aggregate()`` has no ``resume()`` route at all (v1
    contract -- durable pauses need a ``@flow`` as the root), so returning
    ``status="paused"`` here would be a dead end an operator could easily
    mistake for something resumable. Instead this raises
    :class:`~composeai.errors.ConfigError` naming the fix, and purges the
    row :func:`~composeai.runs.settle_agent_run` just persisted (see
    :meth:`~composeai.runs.RunStore.delete_run`) so nothing paused-but-
    unresumable is left behind for ``compose runs`` to show.
    """
    store = runs.open_default()
    run_id = new_ulid()
    trace_id = new_ulid()
    now = time.time()
    store.create_run(
        run_id=run_id,
        kind=kind,
        name=name,
        status="running",
        created_at=now,
        updated_at=now,
        trace_id=trace_id,
        fingerprint=None,
        args_json=None,
    )

    def _thunk() -> Run:
        root_span: tracing.Span | None = None
        try:
            with tracing.span(kind, name) as root_span:
                with budget_scope(budget, root_span):
                    output = run_body()
                trace = tracing.current_trace()
                assert trace is not None  # we're inside an active span, so a trace exists
                usage = trace.rollup_usage(root_span)
                run = Run(
                    id=new_ulid(),
                    status="completed",
                    output=output,
                    usage=usage,
                    trace=trace,
                    messages=[],
                    pending=None,
                )
        except BaseException as exc:
            if root_span is not None:
                if getattr(exc, "_compose_pause", False):
                    tracing.emit_run_finished(root_span, status="paused")
                else:
                    tracing.emit_run_finished(
                        root_span, status="failed", error_type=type(exc).__name__
                    )
            raise

        tracing.emit_run_finished(root_span, status="completed")
        return run

    with runs.use_run_id(run_id), tracing.use_trace(trace_id):
        result = runs.settle_agent_run(store, run_id, _thunk)

    if result.status == "paused":
        # v1 contract: durable pauses need a @flow as the root. A bare
        # pipe()/aggregate() run has no resume() route -- resume() only
        # knows how to route `kind == "agent"` rows (to
        # composeai.agentfn.resume_standalone_agent) and otherwise looks the
        # row's name up in composeai.flow._FLOW_REGISTRY, which a
        # pipe/aggregate's auto-generated name (e.g. "pipe(a → b)") is never
        # registered under -- so a paused "pipe"/"aggregate" row could never
        # actually be resumed. Rather than leave that unresumable "paused"
        # row behind for `compose runs` to show as if it *were* resumable,
        # refuse clearly right here and purge it.
        store.delete_run(run_id)
        raise ConfigError(
            f"{kind} {name!r} paused on an approval/ask_human interrupt, but a "
            f"bare {kind}()/aggregate() isn't resumable -- durable pauses need a "
            "@flow as the root. Wrap this call in a @flow function (so it has a "
            "name resume() can look up and a fingerprint it can check) and call "
            "that instead, e.g.:\n\n"
            "    @flow\n"
            "    def run_it(x):\n"
            f"        return {name}(x)\n"
        )
    return result


async def _arun_top(
    kind: tracing.SpanKind,
    name: str,
    budget: Budget | None,
    run_body: Callable[[], Coroutine[Any, Any, Any]],
) -> Run:
    """Async twin of :func:`_run_top` (v0.4.0 Plan B, Task 5) -- see its
    docstring for the full contract (durable row, budget scope, terminal
    ``run_finished``, and the paused-bare-pipe/aggregate ``ConfigError``
    refusal), byte-identical here except for what has to change to run on
    the CALLER's own event loop instead of a background thread:

    - Row creation (``create_run``) and the paused-row purge
      (``delete_run``) both go through ``await
      composeai._storeasync.worker_for(store).call(...)`` instead of calling
      ``store`` directly -- a direct, blocking SQLite write/delete here
      would freeze the caller's loop instead of merely parking an idle
      thread the way the sync version's direct calls do (mirrors
      :func:`~composeai.runs.arun_standalone_agent`'s identical reasoning
      for row creation).
    - Settling goes through :func:`~composeai.runs.asettle_agent_run`
      instead of :func:`~composeai.runs.settle_agent_run` -- same status
      transitions and pause handling, just awaited rather than called, and
      its own store writes are likewise non-blocking.
    - The inner thunk is ``async def`` and ``await run_body()`` instead of
      calling it directly, since ``run_body`` here is a zero-arg coroutine
      *factory* (``Pipeline.arun``/``Aggregate.arun`` pass
      ``self._arun_stages``/``self._arun_branches``), not a plain callable.

    ``runs.use_run_id``/``tracing.use_trace`` stay plain, synchronous
    context managers -- the contextvars they push stay correctly scoped
    across every ``await`` inside this same coroutine, so there is nothing
    async-specific needed there (same reasoning as
    :func:`~composeai.runs.arun_standalone_agent`).
    """
    store = runs.open_default()
    run_id = new_ulid()
    trace_id = new_ulid()
    now = time.time()
    await worker_for(store).call(
        "create_run",
        run_id=run_id,
        kind=kind,
        name=name,
        status="running",
        created_at=now,
        updated_at=now,
        trace_id=trace_id,
        fingerprint=None,
        args_json=None,
    )

    async def _thunk() -> Run:
        root_span: tracing.Span | None = None
        try:
            with tracing.span(kind, name) as root_span:
                with budget_scope(budget, root_span):
                    output = await run_body()
                trace = tracing.current_trace()
                assert trace is not None  # we're inside an active span, so a trace exists
                usage = trace.rollup_usage(root_span)
                run = Run(
                    id=new_ulid(),
                    status="completed",
                    output=output,
                    usage=usage,
                    trace=trace,
                    messages=[],
                    pending=None,
                )
        except BaseException as exc:
            if root_span is not None:
                if getattr(exc, "_compose_pause", False):
                    tracing.emit_run_finished(root_span, status="paused")
                else:
                    tracing.emit_run_finished(
                        root_span, status="failed", error_type=type(exc).__name__
                    )
            raise

        tracing.emit_run_finished(root_span, status="completed")
        return run

    with runs.use_run_id(run_id), tracing.use_trace(trace_id):
        result = await runs.asettle_agent_run(store, run_id, _thunk)

    if result.status == "paused":
        # Same v1 refusal as `_run_top` -- see its docstring for the full
        # rationale. `delete_run` routes through the store worker here
        # (see this function's own docstring) rather than calling it
        # directly.
        await worker_for(store).call("delete_run", run_id)
        raise ConfigError(
            f"{kind} {name!r} paused on an approval/ask_human interrupt, but a "
            f"bare {kind}()/aggregate() isn't resumable -- durable pauses need a "
            "@flow as the root. Wrap this call in a @flow function (so it has a "
            "name resume() can look up and a fingerprint it can check) and call "
            "that instead, e.g.:\n\n"
            "    @flow\n"
            "    def run_it(x):\n"
            f"        return {name}(x)\n"
        )
    return result


# --- Pipeline -------------------------------------------------------------------


class Pipeline:
    """The callable object produced by :func:`pipe`.

    ``pipeline(x)`` is sugar for ``pipeline.run(x).output``. ``.input_type``/
    ``.output_type`` are the first stage's input and the last stage's
    output -- so a ``Pipeline`` can itself be a stage of an outer ``pipe()``
    or a branch of an ``aggregate()``, and still be type-checked correctly.
    """

    def __init__(self, stages: tuple[Stage, ...]) -> None:
        self._stages = stages
        self.input_type: Any = _stage_input_type(stages[0])
        self.output_type: Any = _stage_output_type(stages[-1])
        self._name = "pipe(" + " → ".join(_stage_name(s) for s in stages) + ")"

    def __call__(self, x: Any) -> Any:
        return self.run(x).output

    def __rshift__(self, other: Any) -> Pipeline:
        return _rshift_pipe(self, other)

    def __rrshift__(self, other: Any) -> Pipeline:
        return _rshift_pipe(other, self)

    def run(self, x: Any, budget: Budget | None = None) -> Run:
        return _run_top("pipe", self._name, budget, lambda: self._run_stages(x, streaming=False))

    async def arun(self, x: Any, budget: Budget | None = None) -> Run:
        """Async twin of :meth:`run` (v0.4.0 Plan B, Task 5) -- runs entirely
        on the CALLER's own running event loop via :func:`_arun_top`, never
        the composeai runtime loop."""
        return await _arun_top(
            "pipe", self._name, budget, lambda: self._arun_stages(x, streaming=False)
        )

    def stream(self, x: Any, budget: Budget | None = None) -> RunStream:
        return agentfn._stream_run(
            lambda: _run_top(
                "pipe", self._name, budget, lambda: self._run_stages(x, streaming=True)
            )
        )

    def astream(self, x: Any, budget: Budget | None = None) -> AsyncRunStream:
        """Async twin of :meth:`stream` (v0.4.0 Plan B, Task 5) -- mirrors
        ``AgentFunction.astream``'s shape (:func:`~composeai.agentfn._astream_run`):
        a fresh, private bus, subscribed before the task is created, run as
        an ``asyncio.Task`` on the caller's own loop."""
        return agentfn._astream_run(
            lambda: _arun_top(
                "pipe", self._name, budget, lambda: self._arun_stages(x, streaming=True)
            )
        )

    def _run_stages(self, x: Any, *, streaming: bool) -> Any:
        """Sync facade over the async core :meth:`_arun_stages` (v0.4.0 Plan
        A, Task 8) -- assumes an enclosing 'pipe' span is already open (same
        contract as before). One ``_runtime.run_sync`` hop per call: every
        caller (``.run()``, ``.stream()``'s background thread, and
        ``combinators._ainvoke_stage``'s Pipeline-as-nested-stage branch,
        which calls :meth:`_arun_stages` directly instead of this facade --
        see its own docstring for why) reaches the engine through here or
        directly through the async core, never both.
        """
        return _runtime.run_sync(self._arun_stages(x, streaming=streaming))

    async def _arun_stages(self, x: Any, *, streaming: bool) -> Any:
        """Async core: run every stage in sequence, assuming an enclosing
        'pipe' span is already open."""
        current = x
        for stage in self._stages:
            current = await _ainvoke_stage(stage, current, streaming=streaming)
        return current


def pipe(*stages: Stage) -> Pipeline:
    """Build a :class:`Pipeline` chaining ``stages`` in sequence.

    Requires at least 2 stages (else :class:`~composeai.errors.CompositionTypeError`).
    Every consecutive pair is type-checked *before* the pipeline is
    returned: ``stages[i]``'s output type must be
    :func:`compatible <_types_compatible>` with ``stages[i + 1]``'s input
    type, or ``pipe()`` itself raises ``CompositionTypeError`` naming both
    stages and both types -- no stage ever runs, so a wiring bug never
    costs an API call.
    """
    if len(stages) < 2:
        raise CompositionTypeError(f"pipe() requires at least 2 stages, got {len(stages)}")

    for i in range(len(stages) - 1):
        left, right = stages[i], stages[i + 1]
        out_t = _stage_output_type(left)
        in_t = _stage_input_type(right)
        if not _types_compatible(out_t, in_t):
            raise CompositionTypeError(
                f"pipe(): stage {i + 1} ({_stage_name(left)}) returns {_type_name(out_t)} "
                f"but stage {i + 2} ({_stage_name(right)}) expects {_type_name(in_t)}"
            )

    return Pipeline(tuple(stages))


# --- Aggregate ------------------------------------------------------------------


class Aggregate:
    """The callable object produced by :func:`aggregate`.

    ``agg(x)`` is sugar for ``agg.run(x).output`` -- a ``{branch_name:
    output}`` dict in declaration order. ``.input_type`` is the branches'
    common input type, or ``Any`` if they disagree; ``.output_type`` is
    always ``dict``.
    """

    def __init__(
        self, branches: dict[str, Stage], *, timeout_per_branch: float | None = None
    ) -> None:
        self._branches = dict(branches)
        self._timeout_per_branch = timeout_per_branch
        input_types = [_stage_input_type(s) for s in self._branches.values()]
        first = input_types[0]
        self.input_type: Any = first if all(t == first for t in input_types) else Any
        self.output_type: Any = dict
        self._name = "aggregate(" + ", ".join(self._branches.keys()) + ")"

    def __call__(self, x: Any) -> dict[str, Any]:
        return self.run(x).output

    def __rshift__(self, other: Any) -> Pipeline:
        return _rshift_pipe(self, other)

    def __rrshift__(self, other: Any) -> Pipeline:
        return _rshift_pipe(other, self)

    def run(self, x: Any, budget: Budget | None = None) -> Run:
        return _run_top(
            "aggregate", self._name, budget, lambda: self._run_branches(x, streaming=False)
        )

    async def arun(self, x: Any, budget: Budget | None = None) -> Run:
        """Async twin of :meth:`run` (v0.4.0 Plan B, Task 5) -- runs entirely
        on the CALLER's own running event loop via :func:`_arun_top`, never
        the composeai runtime loop."""
        return await _arun_top(
            "aggregate", self._name, budget, lambda: self._arun_branches(x, streaming=False)
        )

    def stream(self, x: Any, budget: Budget | None = None) -> RunStream:
        return agentfn._stream_run(
            lambda: _run_top(
                "aggregate", self._name, budget, lambda: self._run_branches(x, streaming=True)
            )
        )

    def astream(self, x: Any, budget: Budget | None = None) -> AsyncRunStream:
        """Async twin of :meth:`stream` (v0.4.0 Plan B, Task 5) -- mirrors
        ``AgentFunction.astream``'s shape (:func:`~composeai.agentfn._astream_run`):
        a fresh, private bus, subscribed before the task is created, run as
        an ``asyncio.Task`` on the caller's own loop."""
        return agentfn._astream_run(
            lambda: _arun_top(
                "aggregate", self._name, budget, lambda: self._arun_branches(x, streaming=True)
            )
        )

    def _run_branches(self, x: Any, *, streaming: bool) -> dict[str, Any]:
        """Sync facade over the async core :meth:`_arun_branches` (v0.4.0
        Plan A, Task 8) -- see its docstring for the full contract, byte-
        identical here. Assumes an enclosing 'aggregate' span is already
        open, same as before. One ``_runtime.run_sync`` hop per call
        (``.run()``, ``.stream()``'s background thread) --
        ``combinators._ainvoke_stage``'s Aggregate-as-nested-stage branch
        calls :meth:`_arun_branches` directly instead, never both.
        """
        return _runtime.run_sync(self._arun_branches(x, streaming=streaming))

    async def _arun_branches(self, x: Any, *, streaming: bool) -> dict[str, Any]:
        """Async core: run every branch concurrently, assuming an enclosing
        'aggregate' span is already open.

        Every branch settles (success or exception) before this returns;
        on any failure the first branch's exception *in declaration order*
        is raised, regardless of which branch actually finished first --
        :func:`~composeai._dispatch.gather_settled` returns settled pairs in
        the SAME order as the coroutines it was given (``asyncio.gather``'s
        own ordering guarantee), and this iterates ``self._branches`` (a
        plain ``dict``, insertion-ordered) to preserve that.

        This applies equally when a branch pauses (``composeai.hitl._Pause``
        is a ``BaseException``, caught by ``_arun_one`` like anything else):
        if two or more branches call ``approve()``/``ask_human()`` (or hit an
        unanswered ``@tool(requires_approval=True)`` call) with no journaled
        answer, only the first (in declaration order) actually raises and
        pauses the enclosing flow/run this attempt -- the others' pauses are
        silently discarded *this* attempt and simply happen again, one at a
        time, on each subsequent ``resume()`` that still lacks their answer
        (or all at once if every answer is supplied together).

        Inside an active ``@flow`` body, each branch runs under its own
        deterministic journal scope (``"aggregate#n/{branch_name}"``,
        reserved serially -- in declaration order -- on *this* (engine)
        coroutine, before any branch is dispatched) -- see
        ``composeai.runs``'s scope-stack module docs. Without this, any
        ``@task``/``@agent``/nested-``@flow`` call inside a branch would
        journal under a key assigned by whichever concurrently-scheduled
        task happened to reach it first, not by declaration order --
        nondeterministic across a crash/resume, potentially attaching a
        completed branch's cached step to the wrong branch. This applies to
        *every* stage kind (bare callables, ``@agent`` functions, nested
        ``Pipeline``/``Aggregate``), not just ``@task``. Each branch's
        ``runs.push_scope`` call moves INSIDE its own coroutine (wrapping
        that branch's stage invocation), rather than around the fan-out as
        a whole: contextvars are task-local, so a push made inside a
        coroutine only affects that coroutine's own task-local context copy
        (taken by ``asyncio.gather`` at the moment each task is scheduled) --
        the same isolation the sync version's per-thread context copy gave.
        """
        ctx = runs.current_run_context()
        if ctx is not None:
            agg_segment = ctx.reserve_scope_segment("aggregate")
            branch_scopes: dict[str, str | None] = {
                name: f"{agg_segment}/{name}" for name in self._branches
            }
        else:
            branch_scopes = {name: None for name in self._branches}

        async def _adispatch(name: str, stage: Stage) -> Any:
            async def _call() -> Any:
                return await _ainvoke_stage(
                    stage,
                    x,
                    streaming=streaming,
                    timeout=self._timeout_per_branch,
                    timeout_name=name,
                    timeout_kind="aggregate branch",
                )

            scope = branch_scopes[name]
            if scope is None:
                return await _call()
            with runs.push_scope(scope):
                return await _call()

        async def _arun_one(name: str, stage: Stage) -> tuple[Any, BaseException | None]:
            try:
                return await _adispatch(name, stage), None
            except BaseException as exc:  # settled below, re-raised as-is once every branch is done
                return None, exc

        settled = await gather_settled(
            [_arun_one(name, stage) for name, stage in self._branches.items()]
        )
        outcomes = dict(
            zip(self._branches.keys(), (outcome for outcome, _exc in settled), strict=True)
        )

        for _output, exc in outcomes.values():
            if exc is not None:
                raise exc
        return {name: output for name, (output, _exc) in outcomes.items()}


def aggregate(*, timeout_per_branch: float | None = None, **branches: Stage) -> Aggregate:
    """Build an :class:`Aggregate` running every branch in parallel.

    ``timeout_per_branch`` (seconds) bounds each branch with the same
    daemon-thread race ``@task(timeout=)`` and ``map(timeout_per_item=)``
    use; a timed-out branch raises :class:`~composeai.errors.TaskTimeoutError`
    under the existing first-branch-in-declaration-order rule. One
    consequence of taking this as a keyword parameter: a branch cannot
    itself be named ``timeout_per_branch``. Requires at least 1 branch
    (else :class:`~composeai.errors.CompositionTypeError`).
    """
    if timeout_per_branch is not None and not isinstance(timeout_per_branch, (int, float)):
        raise ConfigError(
            f"aggregate(): timeout_per_branch must be a number of seconds, "
            f"got {type(timeout_per_branch)!r}"
        )
    if not branches:
        raise CompositionTypeError(f"aggregate() requires at least 1 branch, got {len(branches)}")
    return Aggregate(branches, timeout_per_branch=timeout_per_branch)


# --- map ----------------------------------------------------------------------


@register_serializable
@dataclass
class MapResult:
    """One item's settled outcome from ``compose.map(..., on_error="collect")``.

    ``error``/``error_type`` are plain strings (exceptions don't round-trip
    the journal); registered at import time so a resumed flow in a fresh
    process can decode journaled values containing these.
    """

    ok: bool
    value: Any = None
    error: str | None = None
    error_type: str | None = None


def map(
    fn: Stage,
    items: Sequence[Any],
    *,
    max_workers: int | None = None,
    timeout_per_item: float | None = None,
    on_error: str = "raise",
) -> list[Any]:
    """Apply ``fn`` to every item in ``items`` in parallel, preserving order.

    Exported so users write ``compose.map(fetch, urls)`` (shadowing the
    builtin inside composeai's own namespace is intended). By default
    (``on_error="raise"``), every item settles (success or exception)
    before this returns; on any failure the first failing item *by index*
    is raised. Wrapped in ``span("aggregate", f"map({fn_name})")``; a
    plain-callable ``fn`` gets a ``task`` span per item named
    ``f"{fn_name}[{i}]"``.

    ``timeout_per_item`` bounds each item's run on its own daemon thread
    (the same race :func:`~composeai.runs._run_with_timeout` uses for
    ``@task(timeout=)``) -- a single hung branch raises
    :class:`~composeai.errors.TaskTimeoutError` for that item instead of
    blocking every other item, and the whole ``map()`` call, forever.
    ``None`` (the default) means no per-item timeout.

    ``on_error`` controls how failures (including per-item timeouts) are
    reported:

    - ``"raise"`` (default): the first failure *by index* is re-raised
      from ``map()`` once every item has settled, exactly as above.
    - ``"collect"``: every item's outcome is instead returned as a
      :class:`MapResult` (``ok``, ``value``, ``error``, ``error_type``) --
      the *whole* ``list[MapResult]`` comes back with no exception raised,
      one entry per item in input order, so a caller can salvage the
      items that succeeded even when others failed. Errors are carried as
      strings (``error``/``error_type``), never as exception objects, so
      the result is safe to journal and replay. Interrupts (a nested
      ``approve()``/``ask_human()`` pause) and interpreter-level exits
      (e.g. ``KeyboardInterrupt``, ``SystemExit``) are never collected --
      those are ``BaseException``s by design and always propagate even in
      collect mode.

    Same "first one wins" rule applies to pauses in ``on_error="raise"``
    mode (see :meth:`Aggregate._arun_branches`'s docstring for the
    human-in-the-loop implication): if several items independently call
    ``approve()``/``ask_human()`` unanswered, only the first (by index)
    raises and pauses this attempt -- the rest simply re-pause on a later
    ``resume()``.

    Inside an active ``@flow`` body, each item runs under its own
    deterministic journal scope (``"map#n[i]"``, ``n`` this ``map()``
    call's own step number and ``i`` the item's *input* index -- both
    reserved serially, before any item is dispatched, on this (engine)
    coroutine -- see ``composeai.runs``'s scope-stack module docs). Every
    completed item is journaled individually as it finishes, so a failed
    ``map()`` (whether it raises or, with ``on_error="collect"``, records a
    failed ``MapResult``) never discards its siblings' completed work across
    a ``resume()`` -- only the unfinished tail actually re-runs. Without
    per-item scoping, any ``@task``/``@agent``/nested-``@flow``/nested-
    ``pipe``/nested-``aggregate`` call inside ``fn`` would journal under a
    key assigned by whichever concurrently-scheduled item happened to reach
    it first -- parallel *completion* order, not input order -- so a crash
    mid-map followed by a differently-scheduled resume could attach a
    completed item's cached output to a different item, silently. This
    applies to *every* stage kind (bare callables, ``@agent`` functions,
    ``Pipeline``/``Aggregate``), not just ``@task`` -- there is no
    special-casing by stage type anymore.
    """
    if on_error not in ("raise", "collect"):
        raise ConfigError(f"map(): on_error must be 'raise' or 'collect', got {on_error!r}")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be greater than 0")

    return _runtime.run_sync(
        _amap(
            fn,
            items,
            max_workers=max_workers,
            timeout_per_item=timeout_per_item,
            on_error=on_error,
        )
    )


async def amap(
    fn: Stage,
    items: Sequence[Any],
    *,
    max_workers: int | None = None,
    timeout_per_item: float | None = None,
    on_error: str = "raise",
) -> list[Any]:
    """Async twin of :func:`map` (v0.4.0 Plan B, Task 5) -- same contract,
    see its docstring in full. Runs entirely on the CALLER's own running
    event loop (never the composeai runtime loop): ``await``s the async core
    :func:`_amap` directly instead of routing through
    :func:`~composeai._runtime.run_sync`, which is the ONLY difference from
    :func:`map`'s own body -- both validate ``on_error``/``max_workers``
    identically, and up front, so a bad argument raises immediately without
    touching the composeai runtime (or, here, without even needing a running
    loop to have started any work yet).
    """
    if on_error not in ("raise", "collect"):
        raise ConfigError(f"amap(): on_error must be 'raise' or 'collect', got {on_error!r}")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be greater than 0")

    return await _amap(
        fn,
        items,
        max_workers=max_workers,
        timeout_per_item=timeout_per_item,
        on_error=on_error,
    )


async def _amap(
    fn: Stage,
    items: Sequence[Any],
    *,
    max_workers: int | None,
    timeout_per_item: float | None,
    on_error: str,
) -> list[Any]:
    """Async core of :func:`map` (v0.4.0 Plan A, Task 8) -- see its
    docstring for the full contract, byte-identical here. ``on_error``
    validation stays in :func:`map` itself so it raises immediately,
    without even touching the composeai runtime.

    ``max_workers`` (``None`` means "as many as there are items", same
    default as before) bounds concurrency via an ``asyncio.Semaphore``
    rather than an OS thread pool's size -- there is no thread to bound any
    more for a stage that runs natively as a coroutine (an ``AgentFunction``/
    ``Pipeline``/``Aggregate``/``async def`` callable); a plain sync
    callable still runs off-loop, on its own dedicated daemon thread (see
    :func:`~composeai._dispatch.run_stage`'s ``_run_sync_on_own_thread``
    bridge -- deliberately NOT ``asyncio.to_thread``, whose shared, bounded
    default executor would starve under nested fan-out), so the semaphore
    is what actually limits how many are in flight at once, same as the old
    pool size did.

    Every item's coroutine (``_arun_one``) catches ``BaseException`` itself
    and always returns its own ``(output, exc)`` pair rather than letting
    anything propagate through ``gather_settled`` -- mirroring
    ``composeai.agentfn._aprocess_tool_use``'s identical pattern for its own
    parallel batch: ``gather_settled``'s own exception slot is never
    populated in practice, exactly like the sync version's
    ``future.result()`` never raising from ``_run_one``'s own body.
    """
    items = list(items)
    fn_name = _stage_name(fn)

    ctx = runs.current_run_context()
    if ctx is not None:
        map_segment = ctx.reserve_scope_segment("map")
        item_scopes: list[str | None] = [f"{map_segment}[{i}]" for i in range(len(items))]
    else:
        item_scopes = [None] * len(items)

    workers = max_workers if max_workers is not None else (len(items) or 1)
    semaphore = asyncio.Semaphore(workers)

    async def _adispatch(i: int, item: Any) -> Any:
        scope = item_scopes[i]
        name = f"{fn_name}[{i}]"

        async def _call() -> Any:
            return await _ainvoke_stage(
                fn,
                item,
                streaming=False,
                task_name=name,
                timeout=timeout_per_item,
                timeout_kind="@task",
            )

        async with semaphore:
            if scope is None:
                return await _call()
            with runs.push_scope(scope):
                return await _call()

    with tracing.span("aggregate", f"map({fn_name})"):

        async def _arun_one(i: int, item: Any) -> tuple[Any, BaseException | None]:
            try:
                return await _adispatch(i, item), None
            except BaseException as exc:  # settled below, re-raised as-is once every item is done
                return None, exc

        settled = await gather_settled([_arun_one(i, item) for i, item in enumerate(items)])
        outcomes = [outcome for outcome, _exc in settled]

        if on_error == "collect":
            results: list[Any] = []
            for output, exc in outcomes:
                if exc is None:
                    results.append(MapResult(ok=True, value=output))
                elif isinstance(exc, Exception) and not getattr(exc, "_compose_pause", False):
                    results.append(
                        MapResult(ok=False, error=str(exc), error_type=type(exc).__name__)
                    )
                else:
                    # Pauses (BaseException by design) and interpreter-level
                    # exits always propagate -- collect never swallows them.
                    raise exc
            return results

        for _output, exc in outcomes:
            if exc is not None:
                raise exc
        return [output for output, _exc in outcomes]
