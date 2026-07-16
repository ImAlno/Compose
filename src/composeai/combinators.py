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
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, overload

from typing_extensions import TypeVar

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

# `In`/`Out` carry PEP 696 `default=Any` (via `typing_extensions`, same as
# `runs.R` -- v0.5.0 Plan B, Task 1) so a bare `Stage`/`Pipeline`/`Aggregate`
# annotation still means `Stage[Any, Any]`/`Pipeline[Any, Any]`/
# `Aggregate[Any]` rather than tripping `reportMissingTypeArgument` if strict
# pyright is ever turned on. They are intentionally the SAME TypeVar objects
# reused across `Stage`'s `Protocol[In, Out]` base and `Pipeline`'s
# `Generic[In, Out]` base below (and `Aggregate`'s `Generic[In]`) -- not a
# naming coincidence: `In` only ever appears in an input (parameter) position
# and `Out` only ever in an output (return) position across all three types,
# so the same contravariant/covariant variance is correct for all of them,
# exactly mirroring `Callable[[In], Out]`'s own variance.
In = TypeVar("In", contravariant=True, default=Any)
Out = TypeVar("Out", covariant=True, default=Any)
# Method-scoped: only ever solved from a `__rshift__`/`__rrshift__` call's own
# argument, never left bare in an annotation -- no `default=` needed here.
NewOut = TypeVar("NewOut")
NewIn = TypeVar("NewIn")

# `pipe()`'s overload-ladder typevars (v0.5.0 Plan B, Task 3): plain and
# invariant (no variance, no `default=`) -- each is solved from a concrete
# stage callable at the call site, never left bare. One per rung boundary:
# stage k is `Stage[<k>, <k+1>]`, so the arity-9 ladder threads A -> B -> ...
# -> K across nine stages and returns `Pipeline[A, K]`. `I`/`O`/`l` are
# skipped (ruff E741 ambiguous-name). `A`/`B` are reused by `map`/`amap`'s own
# `fn: Stage[A, B]` / `items: Sequence[A]` overloads further down.
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
G = TypeVar("G")
H = TypeVar("H")
J = TypeVar("J")
K = TypeVar("K")
# `aggregate()`'s common-branch-input typevar (Task 3): solved from every
# typed `**branches` value through the `Stage[AggIn, Any]` arm of its
# parameter union -- see `aggregate`'s own docstring for the full contract,
# including why the `| Callable[[Any], Any]` escape-hatch arm (finding I1) is
# what keeps a bare, unannotated lambda branch's body clean (its parameter
# gets `Any`, not the still-unsolved `AggIn`). `default=Any` (same convention
# as `In`/`Out`/`runs.R`/`T_mr`) makes the all-bare-lambda case -- nothing to
# solve `AggIn` from -- fall back to `Aggregate[Any]` rather than
# `Aggregate[Unknown]`.
AggIn = TypeVar("AggIn", default=Any)


class Stage(Protocol[In, Out]):
    """Anything usable as a pipe/aggregate/map stage: an :class:`~composeai.agentfn.AgentFunction`,
    a :class:`Pipeline`, an :class:`Aggregate`, or a plain callable taking one
    positional argument.

    A structural :class:`~typing.Protocol` -- replaces the old ``Stage = Any``
    runtime alias in every annotation below (v0.5.0 Plan B, Task 2). It is
    never referenced at runtime anywhere in this module -- no
    ``isinstance(x, Stage)``, no ``Stage(...)`` construction; the isinstance
    cluster a few lines down dispatches by concrete type
    (``AgentFunction``/``Pipeline``/``Aggregate``/``Tool``) instead, and
    ``never isinstance() against a subscripted generic`` applies there, not
    here -- so there was no runtime alias to preserve. Every site that used
    to read ``stage: Stage`` keeps reading ``stage: Stage``, now resolving to
    this ``Protocol`` (bare, i.e. ``Stage[Any, Any]``) instead of ``Any``.

    The ``/`` in ``__call__`` is load-bearing: without it, the protocol would
    require the matching callable's parameter to be literally named ``x``
    (pyright: "Parameter name mismatch"), and almost nothing would
    structurally match -- see
    ``plans/superpowers/research/2026-07-16-typing/prototype/proto.py``.
    """

    def __call__(self, x: In, /) -> Out: ...


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

    Stays fully ``Any``-erased (the bare ``Pipeline`` return annotation means
    ``Pipeline[Any, Any]`` under the ``default=Any`` TypeVars above,
    v0.5.0 Plan B, Task 2) -- it is the single runtime body behind FOUR
    differently-parameterized call sites (``Pipeline``/``Aggregate`` ×
    ``__rshift__``/``__rrshift__``), each of which already carries its own
    precise ``Stage[...] -> Pipeline[...]`` generic signature. A
    ``Pipeline[Any, Any]`` return is freely assignable to any of those
    (``Any`` type arguments bypass pyright's generic-argument invariance),
    so nothing is lost by leaving this shared helper itself untyped --
    ``pipe()`` itself stays untyped this task too (its overload ladder is
    v0.5.0 Plan B, Task 3).
    """

    def _flat(stage: Any) -> tuple[Any, ...]:
        return stage._stages if isinstance(stage, Pipeline) else (stage,)

    return pipe(*_flat(left), *_flat(right))


# --- type introspection --------------------------------------------------------

# NOTE: never isinstance() against a subscripted generic (Pipeline[X]) --
# TypeError at runtime. Every isinstance() below (and in _ainvoke_stage
# further down) checks against the bare runtime classes (AgentFunction,
# Pipeline, Aggregate, Tool) -- Pipeline/Aggregate becoming Generic[...]
# (v0.5.0 Plan B, Task 2) doesn't change that: `isinstance(x, Pipeline)`
# stays legal (Generic supplies plain __class_getitem__, the runtime class
# itself is still unparameterized), but `isinstance(x, Pipeline[str, int])`
# would raise immediately -- never write the latter.


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


async def _arun_pipeline_nested(stage: Pipeline, x: Any, *, streaming: bool = False) -> Any:
    """Run ``stage`` (a :class:`Pipeline`) as a nested step of whatever
    trace/run/journal is already ambient.

    Opens the pipe's own ``pipe`` span nested under whatever span is
    currently active and recurses straight into
    :meth:`Pipeline._arun_stages` -- never through :func:`_run_top`/
    :func:`_arun_top`, so no new ``run_id``/``trace_id`` is minted and no
    ``runs`` row is created. This is the exact shape ``_ainvoke_stage``'s
    ``Pipeline``-as-a-stage branch has always used (a pipe used as a stage
    of an outer ``pipe()``/``aggregate()`` never had the cross-trace bug --
    see the trace-linkage investigation); :meth:`Pipeline.__call__`, when
    there's already an ambient span or run context, now shares this same
    helper instead of always routing through ``.run()``/``_run_top``.
    """
    with tracing.span("pipe", stage._name):
        return await stage._arun_stages(x, streaming=streaming)


async def _arun_pipeline_nested_run(
    stage: Pipeline, x: Any, *, streaming: bool = False
) -> Run[Any]:
    """:class:`Run`-returning twin of :func:`_arun_pipeline_nested`, used by
    :meth:`Pipeline.arun`/:meth:`Pipeline.astream`'s own nested-adoption
    branch (v0.5.0 Plan A follow-up, closing finding I1).

    Identical execution shape -- same ``pipe`` span, same recursion straight
    into :meth:`Pipeline._arun_stages`, never through :func:`_arun_top` -- so
    no new ``run_id``/``trace_id``/``runs`` row here either. The only reason
    this isn't just ``_arun_pipeline_nested`` itself: ``.arun()``/``.astream()``
    have to return a ``Run`` (their public contract, matched even on this
    nested path -- see ``agentfn._arun_agent_journaled``'s identical
    contract for a nested ``@agent`` call), so the span has to stay captured
    (``as span``) to compute a step-scoped usage rollup, whereas
    ``_arun_pipeline_nested``'s two callers (``_ainvoke_stage``'s
    Pipeline-as-a-stage branch and :meth:`Pipeline.__call__`) only ever want
    the raw output. ``usage=trace.rollup_usage(span)`` is scoped to just this
    pipe's own subtree -- not the whole enclosing run -- mirroring
    :func:`~composeai.flow._arun_flow_journaled`'s ``Run.usage`` contract for
    a nested ``await inner.arun(...)`` flow call.
    """
    with tracing.span("pipe", stage._name) as span:
        output = await stage._arun_stages(x, streaming=streaming)
    trace = tracing.current_trace()
    assert trace is not None  # we're inside an active span, so a trace exists
    return Run(
        id=new_ulid(),
        status="completed",
        output=output,
        usage=trace.rollup_usage(span),
        trace=trace,
        messages=[],
        pending=None,
    )


async def _arun_aggregate_nested(stage: Aggregate, x: Any, *, streaming: bool = False) -> Any:
    """Run ``stage`` (an :class:`Aggregate`) as a nested step of whatever
    trace/run/journal is already ambient.

    Opens the aggregate's own ``aggregate`` span nested under whatever span
    is currently active and recurses straight into
    :meth:`Aggregate._arun_branches` -- never through :func:`_run_top`/
    :func:`_arun_top`, so no new ``run_id``/``trace_id`` is minted and no
    ``runs`` row is created. This is the exact shape ``_ainvoke_stage``'s
    ``Aggregate``-as-a-stage branch has always used (an aggregate used as a
    stage of an outer ``pipe()``/``aggregate()`` never had the cross-trace
    bug -- see the trace-linkage investigation, and its "worse than pipe"
    pin: called directly inside a ``@flow`` body, an ``aggregate()`` result
    used to render ZERO children under the flow root at all);
    :meth:`Aggregate.__call__`, when there's already an ambient span or run
    context, now shares this same helper instead of always routing through
    ``.run()``/``_run_top`` -- see :func:`_arun_pipeline_nested`'s docstring
    for the parallel ``Pipeline`` fix (v0.5.0 Plan A, Task 1) this mirrors.
    """
    with tracing.span("aggregate", stage._name):
        return await stage._arun_branches(x, streaming=streaming)


async def _arun_aggregate_nested_run(
    stage: Aggregate, x: Any, *, streaming: bool = False
) -> Run[Any]:
    """:class:`Run`-returning twin of :func:`_arun_aggregate_nested`, used by
    :meth:`Aggregate.arun`/:meth:`Aggregate.astream`'s own nested-adoption
    branch -- the ``Aggregate`` twin of :func:`_arun_pipeline_nested_run`
    (v0.5.0 Plan A follow-up, closing finding I1); see that function's
    docstring for the full rationale this mirrors exactly (span captured for
    a step-scoped usage rollup, ``_arun_top`` never involved, no new
    ``run_id``/``trace_id``/``runs`` row).
    """
    with tracing.span("aggregate", stage._name) as span:
        output = await stage._arun_branches(x, streaming=streaming)
    trace = tracing.current_trace()
    assert trace is not None  # we're inside an active span, so a trace exists
    return Run(
        id=new_ulid(),
        status="completed",
        output=output,
        usage=trace.rollup_usage(span),
        trace=trace,
        messages=[],
        pending=None,
    )


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
            return await _arun_pipeline_nested(stage, x, streaming=streaming)

        return await run_stage(_call, (), {}, timeout=timeout, name=tname, kind=timeout_kind)

    if isinstance(stage, Aggregate):

        async def _call() -> Any:
            return await _arun_aggregate_nested(stage, x, streaming=streaming)

        return await run_stage(_call, (), {}, timeout=timeout, name=tname, kind=timeout_kind)

    with tracing.span("task", stage_name) as task_span:
        task_span.set_input(x)
        result = await run_stage(stage, (x,), {}, timeout=timeout, name=tname, kind=timeout_kind)
        task_span.set_output(result)
        return result


def _run_top(
    kind: tracing.SpanKind, name: str, budget: Budget | None, run_body: Callable[[], Any]
) -> Run[Any]:
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

    def _thunk() -> Run[Any]:
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
) -> Run[Any]:
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

    async def _thunk() -> Run[Any]:
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


async def _arun_pipeline_on_callers_loop(
    stage: Pipeline, x: Any, budget: Budget | None, *, streaming: bool = False
) -> Run[Any]:
    """Shared async core behind both :meth:`Pipeline.arun` and
    :meth:`Pipeline.astream` -- mirrors
    ``agentfn._arun_agent_on_callers_loop``'s identical relationship to
    ``AgentFunction.arun``/``AgentFunction.astream`` (both funnel through
    that one ctx-checking core; see its docstring) and, in turn,
    :meth:`Pipeline.__call__`'s own ctx-check (v0.5.0 Plan A, Task 1).

    Closes finding I1 (v0.5.0 Plan A final review): before this, ``arun``/
    ``astream`` never checked for an ambient span/run context at all -- they
    always routed through :func:`_arun_top` and minted an independent
    ``runs`` row, even from inside an active ``@flow`` body, so
    ``await the_pipe.arun(x)`` nested in an async flow leaked an unbudgeted
    extra call past ``Budget``, undercounted the enclosing run's
    ``.usage``, and left an orphan ``kind="pipe"`` row whose own trace
    rendered header-only (empirically probed -- see the final-review
    report). Now: an ambient span or run context routes to
    :func:`_arun_pipeline_nested_run` (join as a nested step, no new
    ``run_id``/``trace_id``); otherwise :func:`_arun_top` mints the genuine
    top-level durable run, unchanged.

    ``budget=...`` on the nested path raises :class:`~composeai.errors.ConfigError`
    instead of silently doing nothing -- mirrors :meth:`~composeai.flow.Flow.arun`'s
    identical nested budget guard (see its docstring): a nested step can't
    take its own cap, the enclosing run's budget already governs every span
    inside it.
    """
    if tracing.current_span() is not None or runs.current_run_context() is not None:
        if budget is not None:
            raise ConfigError(
                f"pipe {stage._name!r}: budget= is not supported on a nested "
                "arun()/astream() call (the enclosing run's budget governs); "
                "set it on the outer run"
            )
        return await _arun_pipeline_nested_run(stage, x, streaming=streaming)
    return await _arun_top(
        "pipe", stage._name, budget, lambda: stage._arun_stages(x, streaming=streaming)
    )


async def _arun_aggregate_on_callers_loop(
    stage: Aggregate, x: Any, budget: Budget | None, *, streaming: bool = False
) -> Run[Any]:
    """``Aggregate`` twin of :func:`_arun_pipeline_on_callers_loop` -- shared
    async core behind both :meth:`Aggregate.arun` and
    :meth:`Aggregate.astream`; see that function's docstring for the full
    rationale (closing finding I1) this mirrors exactly.
    """
    if tracing.current_span() is not None or runs.current_run_context() is not None:
        if budget is not None:
            raise ConfigError(
                f"aggregate {stage._name!r}: budget= is not supported on a nested "
                "arun()/astream() call (the enclosing run's budget governs); "
                "set it on the outer run"
            )
        return await _arun_aggregate_nested_run(stage, x, streaming=streaming)
    return await _arun_top(
        "aggregate", stage._name, budget, lambda: stage._arun_branches(x, streaming=streaming)
    )


# --- Pipeline -------------------------------------------------------------------


class Pipeline(Generic[In, Out]):
    """The callable object produced by :func:`pipe`.

    ``pipeline(x)`` is sugar for ``pipeline.run(x).output``. ``.input_type``/
    ``.output_type`` are the first stage's input and the last stage's
    output -- so a ``Pipeline`` can itself be a stage of an outer ``pipe()``
    or a branch of an ``aggregate()``, and still be type-checked correctly.

    ``Generic[In, Out]`` (v0.5.0 Plan B, Task 2) is erasure-only, same as
    :class:`~composeai.runs.Run`'s ``Generic[R]`` (Task 1): ``__init__``
    itself stays ``Any``-typed (it builds ``.input_type``/``.output_type`` by
    runtime introspection, same as always), so constructing a ``Pipeline``
    directly -- or via the still-untyped :func:`pipe` -- infers bare
    ``Pipeline[Any, Any]`` (the ``default=Any`` TypeVars' fallback); only an
    explicit ``Pipeline[X, Y]`` annotation, or chaining onward with ``>>``
    (whose ``NewOut``/``NewIn`` genuinely get solved from the other operand),
    narrows it further. :func:`pipe`'s own overload ladder -- which infers
    ``In``/``Out`` from the actual stage callables -- is v0.5.0 Plan B,
    Task 3.
    """

    def __init__(self, stages: tuple[Stage, ...]) -> None:
        self._stages = stages
        self.input_type: Any = _stage_input_type(stages[0])
        self.output_type: Any = _stage_output_type(stages[-1])
        self._name = "pipe(" + " → ".join(_stage_name(s) for s in stages) + ")"

    def __call__(self, x: In) -> Out:
        """Sugar for :meth:`run`'s output -- UNLESS a span or run context is
        already ambient (a ``@flow``/``@task`` body, another ``@agent``
        call, ...), in which case this joins that trace/run as a nested step
        instead of minting an independent one (v0.5.0 Plan A, Task 1 -- see
        :func:`_arun_pipeline_nested`'s docstring for the full contract and
        the trace-linkage investigation this fixes). Bridges via
        ``_runtime.run_sync`` the same way ``.run()``'s own sync facade
        (:meth:`_run_stages`) does -- legal here because this is only ever
        reached from a non-runtime thread (the caller's own thread, or a
        sync ``@flow``/``@task`` body's dedicated worker thread -- never the
        composeai runtime loop itself), exactly like every other nested
        sync-facade call in this codebase (e.g. ``_run_flow_journaled``'s
        identical ``_runtime.run_sync`` re-entry for a nested ``@flow``
        call). The genuine trace-root case (no ambient span/context at all)
        is untouched: still ``self.run(x).output``, byte-identical to
        before.
        """
        if tracing.current_span() is not None or runs.current_run_context() is not None:
            return _runtime.run_sync(_arun_pipeline_nested(self, x))
        return self.run(x).output

    def __rshift__(self, other: Stage[Out, NewOut]) -> Pipeline[In, NewOut]:
        return _rshift_pipe(self, other)

    def __rrshift__(self, other: Stage[NewIn, In]) -> Pipeline[NewIn, Out]:
        return _rshift_pipe(other, self)

    def run(self, x: In, budget: Budget | None = None) -> Run[Out]:
        return _run_top("pipe", self._name, budget, lambda: self._run_stages(x, streaming=False))

    async def arun(self, x: In, budget: Budget | None = None) -> Run[Out]:
        """Async twin of :meth:`run` (v0.4.0 Plan B, Task 5) -- runs entirely
        on the CALLER's own running event loop, never the composeai runtime
        loop. Ctx-checks for an ambient span/run context exactly like
        :meth:`__call__` does (v0.5.0 Plan A follow-up, finding I1): a
        nested ``await inner.arun(x)`` from inside an active ``@flow``/
        ``@task``/``@agent`` body joins that enclosing trace/run as a nested
        step (:func:`_arun_pipeline_nested_run`) instead of always minting
        an independent one via :func:`_arun_top` the way this used to,
        unconditionally -- see :func:`_arun_pipeline_on_callers_loop`'s
        docstring for the full contract, and
        ``agentfn._arun_agent_on_callers_loop`` for the identical shape
        ``AgentFunction.arun`` already had.
        """
        return await _arun_pipeline_on_callers_loop(self, x, budget, streaming=False)

    def stream(self, x: In, budget: Budget | None = None) -> RunStream[Out]:
        return agentfn._stream_run(
            lambda: _run_top(
                "pipe", self._name, budget, lambda: self._run_stages(x, streaming=True)
            )
        )

    def astream(self, x: In, budget: Budget | None = None) -> AsyncRunStream[Out]:
        """Async twin of :meth:`stream` (v0.4.0 Plan B, Task 5) -- mirrors
        ``AgentFunction.astream``'s shape (:func:`~composeai.agentfn._astream_run`):
        a fresh, private bus, subscribed before the task is created, run as
        an ``asyncio.Task`` on the caller's own loop. Wraps
        :func:`_arun_pipeline_on_callers_loop` -- the SAME ctx-checking core
        :meth:`arun` uses (v0.5.0 Plan A follow-up, finding I1) -- so a
        nested ``the_pipe.astream(x)`` inside an active flow/task/agent body
        adopts the enclosing trace/run exactly like ``arun`` does, instead
        of always minting an independent run the way this used to.
        Confirmed empirically to mirror ``AgentFunction.astream``'s own
        nested behavior (it already ctx-checks via the same
        ``_arun_agent_on_callers_loop`` core its ``arun`` uses) -- i.e. this
        was the one combinator entry point NOT already matching its
        ``@agent`` counterpart.
        """
        return agentfn._astream_run(
            lambda: _arun_pipeline_on_callers_loop(self, x, budget, streaming=True)
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


# --- pipe() overload ladder (v0.5.0 Plan B, Task 3) ---------------------------
#
# Eight overloads, arities 2..9, each returning a fully-typed
# ``Pipeline[A, <last>]`` inferred from the actual stage callables. There is
# deliberately NO ``Any`` fallback overload: a 10-or-more-stage ``pipe()`` call
# is a STATIC no-matching-overload error (pyright ``reportCallIssue``) by
# design -- the runtime ``*stages`` implementation below still accepts any
# count, and docs (Plan C) point longer chains at ``>>`` (which types
# end-to-end at any length). EVERY overload's params are positional-only
# (trailing ``/``) to stay consistent with the ``*stages`` implementation --
# without it pyright reports ``reportInconsistentOverload`` against the impl
# (see prototype/proto.py). Wrong wiring inside the ladder (stage k's output
# not assignable to stage k+1's input) is a ``reportArgumentType`` at the
# offending stage, mirroring the ``CompositionTypeError`` the body raises at
# build time.
@overload
def pipe(s1: Stage[A, B], s2: Stage[B, C], /) -> Pipeline[A, C]: ...
@overload
def pipe(s1: Stage[A, B], s2: Stage[B, C], s3: Stage[C, D], /) -> Pipeline[A, D]: ...
@overload
def pipe(
    s1: Stage[A, B], s2: Stage[B, C], s3: Stage[C, D], s4: Stage[D, E], /
) -> Pipeline[A, E]: ...
@overload
def pipe(
    s1: Stage[A, B], s2: Stage[B, C], s3: Stage[C, D], s4: Stage[D, E], s5: Stage[E, F], /
) -> Pipeline[A, F]: ...
@overload
def pipe(
    s1: Stage[A, B],
    s2: Stage[B, C],
    s3: Stage[C, D],
    s4: Stage[D, E],
    s5: Stage[E, F],
    s6: Stage[F, G],
    /,
) -> Pipeline[A, G]: ...
@overload
def pipe(
    s1: Stage[A, B],
    s2: Stage[B, C],
    s3: Stage[C, D],
    s4: Stage[D, E],
    s5: Stage[E, F],
    s6: Stage[F, G],
    s7: Stage[G, H],
    /,
) -> Pipeline[A, H]: ...
@overload
def pipe(
    s1: Stage[A, B],
    s2: Stage[B, C],
    s3: Stage[C, D],
    s4: Stage[D, E],
    s5: Stage[E, F],
    s6: Stage[F, G],
    s7: Stage[G, H],
    s8: Stage[H, J],
    /,
) -> Pipeline[A, J]: ...
@overload
def pipe(
    s1: Stage[A, B],
    s2: Stage[B, C],
    s3: Stage[C, D],
    s4: Stage[D, E],
    s5: Stage[E, F],
    s6: Stage[F, G],
    s7: Stage[G, H],
    s8: Stage[H, J],
    s9: Stage[J, K],
    /,
) -> Pipeline[A, K]: ...
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


class Aggregate(Generic[In]):
    """The callable object produced by :func:`aggregate`.

    ``agg(x)`` is sugar for ``agg.run(x).output`` -- a ``{branch_name:
    output}`` dict in declaration order. ``.input_type`` is the branches'
    common input type, or ``Any`` if they disagree; ``.output_type`` is
    always ``dict``.

    ``Generic[In]`` (v0.5.0 Plan B, Task 2) -- output stays plain
    ``dict[str, Any]`` (never generic: branches can return unrelated types,
    there is no single ``Out`` to speak of), same erasure-only relationship
    to ``__init__`` as :class:`Pipeline`'s (see its docstring): direct
    construction infers bare ``Aggregate[Any]``; an explicit ``Aggregate[X]``
    annotation, or ``>>``-chaining onward, is what narrows it.
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

    def __call__(self, x: In) -> dict[str, Any]:
        """Sugar for :meth:`run`'s output -- UNLESS a span or run context is
        already ambient (a ``@flow``/``@task`` body, another ``@agent``
        call, ...), in which case this joins that trace/run as a nested step
        instead of minting an independent one (v0.5.0 Plan A, Task 2 -- the
        ``Aggregate`` twin of Task 1's ``Pipeline.__call__`` fix; see
        :func:`_arun_aggregate_nested`'s docstring for the full contract and
        the trace-linkage investigation this fixes). Bridges via
        ``_runtime.run_sync`` the same way ``.run()``'s own sync facade
        (:meth:`_run_branches`) does -- legal here for the same reason as
        ``Pipeline.__call__``: only ever reached from a non-runtime thread.
        The genuine trace-root case (no ambient span/context at all) is
        untouched: still ``self.run(x).output``, byte-identical to before.
        """
        if tracing.current_span() is not None or runs.current_run_context() is not None:
            return _runtime.run_sync(_arun_aggregate_nested(self, x))
        return self.run(x).output

    def __rshift__(self, other: Stage[dict[str, Any], NewOut]) -> Pipeline[In, NewOut]:
        return _rshift_pipe(self, other)

    def __rrshift__(self, other: Stage[NewIn, In]) -> Pipeline[NewIn, dict[str, Any]]:
        return _rshift_pipe(other, self)

    def run(self, x: In, budget: Budget | None = None) -> Run[dict[str, Any]]:
        return _run_top(
            "aggregate", self._name, budget, lambda: self._run_branches(x, streaming=False)
        )

    async def arun(self, x: In, budget: Budget | None = None) -> Run[dict[str, Any]]:
        """Async twin of :meth:`run` (v0.4.0 Plan B, Task 5) -- runs entirely
        on the CALLER's own running event loop, never the composeai runtime
        loop. Ctx-checks for an ambient span/run context exactly like
        :meth:`__call__` does (v0.5.0 Plan A follow-up, finding I1): a
        nested ``await inner.arun(x)`` from inside an active ``@flow``/
        ``@task``/``@agent`` body joins that enclosing trace/run as a nested
        step (:func:`_arun_aggregate_nested_run`) instead of always minting
        an independent one via :func:`_arun_top` the way this used to,
        unconditionally -- see :func:`_arun_aggregate_on_callers_loop`'s
        docstring (and :func:`_arun_pipeline_on_callers_loop`'s, which it
        mirrors) for the full contract.
        """
        return await _arun_aggregate_on_callers_loop(self, x, budget, streaming=False)

    def stream(self, x: In, budget: Budget | None = None) -> RunStream[dict[str, Any]]:
        return agentfn._stream_run(
            lambda: _run_top(
                "aggregate", self._name, budget, lambda: self._run_branches(x, streaming=True)
            )
        )

    def astream(self, x: In, budget: Budget | None = None) -> AsyncRunStream[dict[str, Any]]:
        """Async twin of :meth:`stream` (v0.4.0 Plan B, Task 5) -- mirrors
        ``AgentFunction.astream``'s shape (:func:`~composeai.agentfn._astream_run`):
        a fresh, private bus, subscribed before the task is created, run as
        an ``asyncio.Task`` on the caller's own loop. Wraps
        :func:`_arun_aggregate_on_callers_loop` -- the SAME ctx-checking
        core :meth:`arun` uses (v0.5.0 Plan A follow-up, finding I1) -- so a
        nested ``the_agg.astream(x)`` inside an active flow/task/agent body
        adopts the enclosing trace/run exactly like ``arun`` does. See
        :meth:`Pipeline.astream`'s docstring for the empirical confirmation
        (against ``AgentFunction.astream``) that motivates mirroring this
        shape rather than leaving it independent.
        """
        return agentfn._astream_run(
            lambda: _arun_aggregate_on_callers_loop(self, x, budget, streaming=True)
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


def aggregate(
    *,
    timeout_per_branch: float | None = None,
    **branches: Stage[AggIn, Any] | Callable[[Any], Any],
) -> Aggregate[AggIn]:
    """Build an :class:`Aggregate` running every branch in parallel.

    ``timeout_per_branch`` (seconds) bounds each branch with the same
    daemon-thread race ``@task(timeout=)`` and ``map(timeout_per_item=)``
    use; a timed-out branch raises :class:`~composeai.errors.TaskTimeoutError`
    under the existing first-branch-in-declaration-order rule. One
    consequence of taking this as a keyword parameter: a branch cannot
    itself be named ``timeout_per_branch``. Requires at least 1 branch
    (else :class:`~composeai.errors.CompositionTypeError`).

    Statically (v0.5.0 Plan B, Task 3) this infers ``Aggregate[AggIn]`` where
    ``AggIn`` is the branches' common input type, solved from every branch via
    ``**branches: Stage[AggIn, Any] | Callable[[Any], Any]`` (a DIRECT typed
    signature, not an ``@overload`` -- a lone overload trips
    ``reportInconsistentOverload``). The ``| Callable[[Any], Any]`` arm is the
    fix for finding I1 (v0.5.0 Plan B, Task 3 review): a bare, unannotated
    lambda branch (``aggregate(a=lambda x: x + 1)``) is idiomatic and
    runtime-legit, but a lone ``Stage[AggIn, Any]`` fed the still-unsolved
    ``AggIn`` into the lambda's own parameter, so ``x + 1`` tripped
    ``reportOperatorIssue`` inside the body -- a static regression against the
    old untyped signature. The escape-hatch arm gives such a lambda's
    parameter ``Any`` (clean body, ``Aggregate[Any]`` result) while a typed
    branch still solves ``AggIn`` through the ``Stage`` arm.

    One consequence of that escape hatch, accepted deliberately: a
    *mismatched* typed branch (``aggregate(a=str_stage, c=int_stage)``) is no
    longer statically rejected -- pyright can't tell a bare lambda from a
    mismatched typed callable at the ``Callable[[Any], Any]`` arm, so keeping
    bare lambdas clean (probe (c)) and rejecting mismatched typed branches
    (probe (b)) are mutually exclusive. The plan's core promise (no static
    regression on runtime-legit code) outranks mismatch rejection, and there
    is no runtime cost either way: unlike ``pipe()``, ``aggregate()`` does NOT
    raise on disagreeing branches -- ``Aggregate.__init__`` just falls back to
    ``input_type = Any`` (probed before adoption; see this task's report).
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

T_mr = TypeVar("T_mr", default=Any)
"""``default=Any`` (PEP 696, via ``typing_extensions``) so a bare ``MapResult``
annotation means ``MapResult[Any]`` -- same rationale as ``runs.R``."""


@register_serializable
@dataclass
class MapResult(Generic[T_mr]):
    """One item's settled outcome from ``compose.map(..., on_error="collect")``.

    ``error``/``error_type`` are plain strings (exceptions don't round-trip
    the journal); registered at import time so a resumed flow in a fresh
    process can decode journaled values containing these.
    """

    ok: bool
    value: T_mr | None = None
    error: str | None = None
    error_type: str | None = None


# The async-stage overload pair comes FIRST (finding I2, v0.5.0 Plan B, Task 3
# review): map()/amap() DO accept an ``async def`` plain callable as a stage
# (it runs natively through ``_ainvoke_stage``/``run_stage``; see
# ``test_engine_async``), and map() collects each item's AWAITED value, not the
# coroutine. A lone ``fn: Stage[A, B] -> list[B]`` overload would bind ``B`` to
# an async stage's returned ``Coroutine[..., R]``, mistyping the result as
# ``list[Coroutine[...]]``. Matching ``fn: Callable[[A], Awaitable[B]]`` first
# unwraps the awaitable so ``B`` is the value map() actually returns; a sync
# stage never matches this arm (its return isn't an ``Awaitable``) and falls
# through to the ``Stage[A, B]`` pair below.
@overload
def map(
    fn: Callable[[A], Awaitable[B]],
    items: Sequence[A],
    *,
    max_workers: int | None = ...,
    timeout_per_item: float | None = ...,
    on_error: Literal["raise"] = ...,
) -> list[B]: ...
@overload
def map(
    fn: Callable[[A], Awaitable[B]],
    items: Sequence[A],
    *,
    max_workers: int | None = ...,
    timeout_per_item: float | None = ...,
    on_error: Literal["collect"],
) -> list[MapResult[B]]: ...
@overload
def map(
    fn: Stage[A, B],
    items: Sequence[A],
    *,
    max_workers: int | None = ...,
    timeout_per_item: float | None = ...,
    on_error: Literal["raise"] = ...,
) -> list[B]: ...
@overload
def map(
    fn: Stage[A, B],
    items: Sequence[A],
    *,
    max_workers: int | None = ...,
    timeout_per_item: float | None = ...,
    on_error: Literal["collect"],
) -> list[MapResult[B]]: ...
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
    reported -- and, statically (v0.5.0 Plan B, Task 3), the return type:
    overloads select ``list[B]`` for the default ``on_error="raise"`` (that
    overload carries the default) versus ``list[MapResult[B]]`` for the
    explicit ``on_error="collect"``, with ``B`` inferred from ``fn``'s own
    output and ``items`` checked against ``fn``'s input (``items:
    Sequence[A]``). Each ``on_error`` value has an async-stage overload
    (``fn: Callable[[A], Awaitable[B]]``, matched first so an ``async def``
    stage's ``B`` is its AWAITED value, not the ``Coroutine`` it returns --
    finding I2) and a sync-stage one (``fn: Stage[A, B]``). The implementation
    signature below stays loose (``on_error: str``); ``amap`` carries the
    identical overload set:

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


# Async-stage overload pair first, exactly like :func:`map` above (finding I2).
@overload
async def amap(
    fn: Callable[[A], Awaitable[B]],
    items: Sequence[A],
    *,
    max_workers: int | None = ...,
    timeout_per_item: float | None = ...,
    on_error: Literal["raise"] = ...,
) -> list[B]: ...
@overload
async def amap(
    fn: Callable[[A], Awaitable[B]],
    items: Sequence[A],
    *,
    max_workers: int | None = ...,
    timeout_per_item: float | None = ...,
    on_error: Literal["collect"],
) -> list[MapResult[B]]: ...
@overload
async def amap(
    fn: Stage[A, B],
    items: Sequence[A],
    *,
    max_workers: int | None = ...,
    timeout_per_item: float | None = ...,
    on_error: Literal["raise"] = ...,
) -> list[B]: ...
@overload
async def amap(
    fn: Stage[A, B],
    items: Sequence[A],
    *,
    max_workers: int | None = ...,
    timeout_per_item: float | None = ...,
    on_error: Literal["collect"],
) -> list[MapResult[B]]: ...
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
