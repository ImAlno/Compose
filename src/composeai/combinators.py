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

import inspect
import time
import types
import typing
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from . import agentfn, runs, tracing
from ._encoding import register_serializable
from ._ids import new_ulid
from ._schema import resolve_annotations
from .agentfn import AgentFunction
from .errors import CompositionTypeError, ConfigError
from .runs import Budget, Run, RunStream, _run_with_timeout, budget_scope
from .tools import Tool

Stage = Any
"""Anything usable as a pipe/aggregate/map stage: an :class:`~composeai.agentfn.AgentFunction`,
a :class:`Pipeline`, an :class:`Aggregate`, or a plain callable taking one
positional argument."""


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


def _invoke_stage(stage: Stage, x: Any, *, streaming: bool, task_name: str | None = None) -> Any:
    """Run one stage on input ``x``, opening whatever span kind fits it.

    ``AgentFunction`` stages run via ``agentfn._run_agent`` directly (not
    the public, always-non-streaming ``.run()`` sugar) so a ``streaming``
    pipeline/aggregate can thread ``streaming=True`` down to them and get
    real token deltas on the ambient bus, not just span events. Nested
    ``Pipeline``/``Aggregate`` stages open their own natural span and
    recurse with the same ``streaming`` flag. A plain callable gets a
    ``task`` span with its input/output captured; ``task_name`` overrides
    the default name (used by :func:`map` for per-item span names).
    """
    if isinstance(stage, AgentFunction):
        return agentfn._run_agent(stage, (x,), {}, streaming=streaming).output
    if isinstance(stage, Pipeline):
        with tracing.span("pipe", stage._name):
            return stage._run_stages(x, streaming=streaming)
    if isinstance(stage, Aggregate):
        with tracing.span("aggregate", stage._name):
            return stage._run_branches(x, streaming=streaming)

    name = task_name if task_name is not None else _stage_name(stage)
    with tracing.span("task", name) as task_span:
        task_span.set_input(x)
        result = stage(x)
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

    def run(self, x: Any, budget: Budget | None = None) -> Run:
        return _run_top("pipe", self._name, budget, lambda: self._run_stages(x, streaming=False))

    def stream(self, x: Any, budget: Budget | None = None) -> RunStream:
        return agentfn._stream_run(
            lambda: _run_top(
                "pipe", self._name, budget, lambda: self._run_stages(x, streaming=True)
            )
        )

    def _run_stages(self, x: Any, *, streaming: bool) -> Any:
        """Run every stage in sequence, assuming an enclosing 'pipe' span is already open."""
        current = x
        for stage in self._stages:
            current = _invoke_stage(stage, current, streaming=streaming)
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

    def run(self, x: Any, budget: Budget | None = None) -> Run:
        return _run_top(
            "aggregate", self._name, budget, lambda: self._run_branches(x, streaming=False)
        )

    def stream(self, x: Any, budget: Budget | None = None) -> RunStream:
        return agentfn._stream_run(
            lambda: _run_top(
                "aggregate", self._name, budget, lambda: self._run_branches(x, streaming=True)
            )
        )

    def _run_branches(self, x: Any, *, streaming: bool) -> dict[str, Any]:
        """Run every branch in parallel, assuming an enclosing 'aggregate' span is already open.

        Every branch settles (success or exception) before this returns;
        on any failure the first branch's exception *in declaration order*
        is raised, regardless of which branch actually finished first.

        This applies equally when a branch pauses (``composeai.hitl._Pause``
        is a ``BaseException``, caught by ``_run_one`` like anything else):
        if two or more branches call ``approve()``/``ask_human()`` (or hit an
        unanswered ``@tool(requires_approval=True)`` call) with no journaled
        answer, only the first (in declaration order) actually raises and
        pauses the enclosing flow/run this attempt -- the others' pauses are
        silently discarded *this* attempt and simply happen again, one at a
        time, on each subsequent ``resume()`` that still lacks their answer
        (or all at once if every answer is supplied together).

        Inside an active ``@flow`` body, each branch runs under its own
        deterministic journal scope (``"aggregate#n/{branch_name}"``,
        reserved serially -- in declaration order -- on *this* thread before
        any branch is dispatched) -- see ``composeai.runs``'s scope-stack
        module docs. Without this, any ``@task``/``@agent``/nested-``@flow``
        call inside a branch would journal under a key assigned by whichever
        thread happened to reach it first, not by declaration order --
        nondeterministic across a crash/resume, potentially attaching a
        completed branch's cached step to the wrong branch. This applies to
        *every* stage kind (bare callables, ``@agent`` functions, nested
        ``Pipeline``/``Aggregate``), not just ``@task``.
        """
        ctx = runs.current_run_context()
        if ctx is not None:
            agg_segment = ctx.reserve_scope_segment("aggregate")
            branch_scopes: dict[str, str | None] = {
                name: f"{agg_segment}/{name}" for name in self._branches
            }
        else:
            branch_scopes = {name: None for name in self._branches}

        def _run_one(name: str, stage: Stage) -> tuple[Any, BaseException | None]:
            def call() -> Any:
                return _invoke_stage(stage, x, streaming=streaming)

            def run() -> Any:
                if self._timeout_per_branch is None:
                    return call()
                return _run_with_timeout(
                    call, (), {}, self._timeout_per_branch, name, kind="aggregate branch"
                )

            try:
                scope = branch_scopes[name]
                if scope is None:
                    return run(), None
                with runs.push_scope(scope):
                    return run(), None
            except BaseException as exc:  # settled below, re-raised as-is once every branch is done
                return None, exc

        with ThreadPoolExecutor(max_workers=len(self._branches)) as pool:
            futures = {
                name: pool.submit(tracing.propagate(_run_one), name, stage)
                for name, stage in self._branches.items()
            }
            outcomes = {name: future.result() for name, future in futures.items()}

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
    mode (see :meth:`Aggregate._run_branches`'s docstring for the
    human-in-the-loop implication): if several items independently call
    ``approve()``/``ask_human()`` unanswered, only the first (by index)
    raises and pauses this attempt -- the rest simply re-pause on a later
    ``resume()``.

    Inside an active ``@flow`` body, each item runs under its own
    deterministic journal scope (``"map#n[i]"``, ``n`` this ``map()``
    call's own step number and ``i`` the item's *input* index -- both
    reserved serially, before any item is dispatched, on this thread --
    see ``composeai.runs``'s scope-stack module docs). Every completed
    item is journaled individually as it finishes, so a failed ``map()``
    (whether it raises or, with ``on_error="collect"``, records a failed
    ``MapResult``) never discards its siblings' completed work across a
    ``resume()`` -- only the unfinished tail actually re-runs. Without
    per-item scoping, any ``@task``/``@agent``/nested-``@flow``/nested-
    ``pipe``/nested-``aggregate`` call inside ``fn`` would journal under a
    key assigned by whichever worker thread happened to reach it first --
    parallel *completion* order, not input order -- so a crash mid-map
    followed by a differently-scheduled resume could attach a completed
    item's cached output to a different item, silently. This applies to
    *every* stage kind (bare callables, ``@agent`` functions,
    ``Pipeline``/``Aggregate``), not just ``@task`` -- there is no
    special-casing by stage type anymore.
    """
    if on_error not in ("raise", "collect"):
        raise ConfigError(f"map(): on_error must be 'raise' or 'collect', got {on_error!r}")

    items = list(items)
    fn_name = _stage_name(fn)

    ctx = runs.current_run_context()
    if ctx is not None:
        map_segment = ctx.reserve_scope_segment("map")
        item_scopes: list[str | None] = [f"{map_segment}[{i}]" for i in range(len(items))]
    else:
        item_scopes = [None] * len(items)

    def dispatch(i: int, item: Any) -> Any:
        scope = item_scopes[i]
        name = f"{fn_name}[{i}]"

        def call() -> Any:
            return _invoke_stage(fn, item, streaming=False, task_name=name)

        def run() -> Any:
            if timeout_per_item is None:
                return call()
            # Same daemon-thread race @task(timeout=) uses -- see
            # runs._run_with_timeout for why not a ThreadPoolExecutor, and
            # for the journal abandon-guard the worker gets.
            return _run_with_timeout(call, (), {}, timeout_per_item, name)

        if scope is None:
            return run()
        with runs.push_scope(scope):
            return run()

    with tracing.span("aggregate", f"map({fn_name})"):

        def _run_one(i: int, item: Any) -> tuple[Any, BaseException | None]:
            try:
                return dispatch(i, item), None
            except BaseException as exc:  # settled below, re-raised as-is once every item is done
                return None, exc

        workers = max_workers if max_workers is not None else (len(items) or 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(tracing.propagate(_run_one), i, item) for i, item in enumerate(items)
            ]
            outcomes = [future.result() for future in futures]

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
