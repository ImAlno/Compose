"""``@task``/``@flow``: the durable flow runtime, and ``resume`` for crash recovery.

A ``@flow``-decorated function is a plain Python function whose body calls
``@task``-decorated functions (and/or ``@agent`` functions, and/or
``compose.map`` over a ``@task``). Every call to a ``@task`` (or an
``@agent`` -- see the marked touch in :mod:`composeai.agentfn`) inside an
active flow body is journaled to a :class:`~composeai.runs.RunStore`
keyed by call order (``f"{name}#{n}"``, ``n`` assigned in flow-body order,
never completion order -- see :class:`~composeai.runs.RunContext`): a
journal hit replays the stored value without re-executing; a miss executes
and journals the result. If the flow crashes partway through (or simply
raises), :func:`resume` re-runs the *same* function with the *same*
``run_id``/``trace_id`` -- completed steps replay instantly and only the
unfinished tail actually runs, in the same process or a brand new one.

Flow-body determinism is a documented contract, not an enforced one: the
body must be a deterministic function of its journaled step results (side
effects belong inside ``@task``/``@agent`` calls, not the flow body itself)
-- this module does nothing to detect a violation, it just won't replay
correctly if the contract is broken. Wall-clock reads and randomness are
the two exceptions with dedicated journal-safe helpers: :func:`now` and
:func:`random` may be called directly in a flow body (each call journals
one value in flow-body order and replays it verbatim on resume); raw
``datetime.now()``/``random.random()`` still belong inside ``@task``.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import random as _stdlib_random
import textwrap
import time
from collections.abc import Callable
from typing import Any, overload

from . import _runtime, agentfn, runs, tracing
from ._dispatch import run_stage
from ._encoding import from_jsonable, to_jsonable
from ._ids import new_ulid
from ._schema import register_annotation_types
from ._storeasync import worker_for
from .errors import ConfigError, ResumeMismatchError, TaskTimeoutError
from .messages import Usage
from .runs import (
    Budget,
    JournalScope,
    Run,
    RunContext,
    RunStream,
    _run_with_timeout,
    apersist_pending_interrupt,
    apply_resume_answers,
    budget_scope,
    current_run_context,
    use_run_context,
)
from .runs import open_default as _open_default_store

# --- @task ------------------------------------------------------------------


def _call_input_dict(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {"args": list(args), "kwargs": kwargs}


_TASK_REGISTRY: dict[str, Task] = {}


class Task:
    """The callable object produced by ``@task``.

    Directly callable (``task_obj(...)``) whether or not a flow is
    active: outside a flow it just executes inside a plain ``task`` span;
    inside one (an ambient :class:`~composeai.runs.JournalScope`) each call
    auto-journals as one step -- keyed under whatever scope is active (see
    :func:`~composeai.runs.push_scope`), so a call dispatched by
    ``compose.map``/``aggregate``/the agent loop's parallel tool execution
    gets a deterministic key regardless of thread-scheduling order.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        retries: int,
        timeout: float | None,
        name: str | None,
        replace: bool,
    ) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        self._retries = retries
        self._timeout = timeout
        register_annotation_types(fn)
        # Names are the journal's per-step counter key (see
        # RunContext.reserve_scope_segment) -- two @task objects sharing one
        # name would silently share one counter/namespace inside any flow
        # that calls both, the same reason @flow/@agent names are unique.
        # `replace=True` re-binds an existing name instead of raising --
        # only affects steps not yet journaled (a replaced @task's changed
        # body has no fingerprint/staleness check of its own; the enclosing
        # @flow's fingerprint check is what protects already-journaled runs).
        if not replace and self.name in _TASK_REGISTRY:
            raise ConfigError(
                f"@task name {self.name!r} is already registered -- task names "
                "must be unique per process (they key the durable journal's "
                "per-step counter); pass an explicit name=... to disambiguate"
            )
        _TASK_REGISTRY[self.name] = self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ctx = current_run_context()
        if ctx is None:
            return self._run_uncached(args, kwargs)
        key = ctx.next_key(self.name)
        return self._run_journaled(ctx, key, args, kwargs)

    def _run_uncached(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        with tracing.span("task", self.name) as task_span:
            task_span.set_input(_call_input_dict(args, kwargs))
            result = self._execute(args, kwargs, task_span)
            task_span.set_output(result)
            return result

    def _run_journaled(
        self, ctx: JournalScope, key: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        hit, value = ctx.journal_lookup(key)
        with tracing.span("task", self.name, attributes={"step_key": key}) as task_span:
            if hit:
                task_span.replayed = True
                task_span.set_output(value)
                return value
            task_span.set_input(_call_input_dict(args, kwargs))
            result = self._execute(args, kwargs, task_span)
            recorded = ctx.journal_record(key, result)
            task_span.set_output(recorded)
            return recorded

    def _execute(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], task_span: tracing.Span
    ) -> Any:
        attempt = 0
        while True:
            try:
                if self._timeout is None:
                    return self._fn(*args, **kwargs)
                return _run_with_timeout(self._fn, args, kwargs, self._timeout, self.name)
            except TaskTimeoutError:
                raise  # never retried: retrying would pile up more abandoned threads
            except Exception as exc:
                task_span.attributes.setdefault("retries", []).append(
                    {"type": type(exc).__name__, "message": str(exc)}
                )
                if attempt < self._retries:
                    attempt += 1
                    continue
                raise


@overload
def task(fn: Callable[..., Any]) -> Task: ...


@overload
def task(
    *,
    retries: int = 0,
    timeout: float | None = None,
    name: str | None = None,
    replace: bool = False,
) -> Callable[[Callable[..., Any]], Task]: ...


def task(
    fn: Callable[..., Any] | None = None,
    *,
    retries: int = 0,
    timeout: float | None = None,
    name: str | None = None,
    replace: bool = False,
) -> Task | Callable[[Callable[..., Any]], Task]:
    """Decorate a plain function into a :class:`Task` -- journaled when called inside a ``@flow``.

    Usable bare (``@task``) or with arguments (``@task(retries=2,
    timeout=30, name=...)``).

    ``replace=True`` re-binds an existing task name instead of raising
    ``ConfigError``. A replaced task only affects steps not yet journaled --
    an already-journaled step's stored value still replays unchanged; the
    enclosing ``@flow``'s own fingerprint check (not this) is what guards
    against replaying stale journal entries against changed flow code.
    """

    def decorator(f: Callable[..., Any]) -> Task:
        return Task(f, retries=retries, timeout=timeout, name=name, replace=replace)

    if fn is not None:
        return decorator(fn)
    return decorator


def now() -> datetime.datetime:
    """Journal-safe clock read for ``@flow`` bodies.

    Inside an active flow, each call journals one timezone-aware UTC
    timestamp in flow-body order (keys ``now#1``, ``now#2``, ...): the value
    is drawn once and replays verbatim on every resume, so flow bodies may
    branch on time without violating the determinism contract. Outside a
    flow it is exactly ``datetime.now(timezone.utc)``.
    """
    ctx = current_run_context()
    if ctx is None:
        return datetime.datetime.now(datetime.timezone.utc)
    key = ctx.next_key("now")
    hit, value = ctx.journal_lookup(key)
    if hit:
        return value
    return ctx.journal_record(key, datetime.datetime.now(datetime.timezone.utc))


def random() -> float:
    """Journal-safe uniform draw in ``[0, 1)`` for ``@flow`` bodies.

    Same journaling contract as :func:`now`: one draw per call in flow-body
    order, replayed verbatim on resume. Outside a flow it is exactly
    ``random.random()``. Shadows the stdlib name inside composeai's own
    namespace only -- ``compose.random()`` -- same deliberate choice as
    ``compose.map``.
    """
    ctx = current_run_context()
    if ctx is None:
        return _stdlib_random.random()
    key = ctx.next_key("random")
    hit, value = ctx.journal_lookup(key)
    if hit:
        return value
    return ctx.journal_record(key, _stdlib_random.random())


# --- @flow --------------------------------------------------------------

_FLOW_REGISTRY: dict[str, Flow] = {}


def _compute_fingerprint(fn: Callable[..., Any]) -> str:
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except OSError:
        # No retrievable source (REPL, stdin, some notebook/exec contexts).
        # Degrade instead of failing: flows still run and resume, but
        # changed-code detection is unavailable for them (the fingerprint
        # only distinguishes identity, not source revisions).
        return f"nosource:{fn.__module__}:{fn.__qualname__}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _encode_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """JSON-encode a flow call's arguments. Raises SerializationError up front on a bad arg."""
    payload = {"args": list(args), "kwargs": kwargs}
    return json.dumps(to_jsonable(payload))


def _decode_call(args_json: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    payload = from_jsonable(json.loads(args_json))
    return tuple(payload["args"]), payload["kwargs"]


class Flow:
    """The callable object produced by ``@flow``.

    ``flow_obj(x)`` is sugar for ``flow_obj.run(x).output``. ``.run()``
    always starts a *new* durable run; use :func:`resume` to continue an
    existing one (by ``run_id``), in this process or a fresh one.
    """

    def __init__(self, fn: Callable[..., Any], *, name: str | None, replace: bool) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        self.fingerprint = _compute_fingerprint(fn)
        register_annotation_types(fn)
        # `replace=True` re-binds an existing name instead of raising -- safe
        # for already-journaled/paused runs: `resume()` compares the run's
        # stored fingerprint against *this* (the currently-registered)
        # flow's fingerprint and raises ResumeMismatchError on a mismatch
        # (see `resume()`'s docstring), so a replaced flow's changed source
        # is still caught there, not silently resumed like a replaced
        # standalone @agent would be.
        if not replace and self.name in _FLOW_REGISTRY:
            raise ConfigError(
                f"@flow name {self.name!r} is already registered -- flow names must be "
                "unique per process; pass an explicit name=... to disambiguate"
            )
        _FLOW_REGISTRY[self.name] = self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # A @flow called from inside another active @flow body is a *nested*
        # flow call -- journal it as one step of the ENCLOSING flow (like a
        # nested @agent call -- see agentfn._run_agent), instead of always
        # minting a brand-new durable run row the way a top-level `.run()`
        # does. Without this, resuming the outer flow re-executed the whole
        # inner flow (and every @task/@agent inside it) from scratch on
        # every attempt, even after it had already completed.
        ctx = current_run_context()
        if ctx is not None:
            return _run_flow_journaled(self, ctx, args, kwargs).output
        return self.run(*args, **kwargs).output

    def run(self, *args: Any, budget: Budget | None = None, **kwargs: Any) -> Run:
        args_json = _encode_call(args, kwargs)  # rejects unserializable args up front
        store = _open_default_store()
        run_id = new_ulid()
        trace_id = new_ulid()
        now = time.time()
        store.create_run(
            run_id=run_id,
            kind="flow",
            name=self.name,
            status="running",
            created_at=now,
            updated_at=now,
            trace_id=trace_id,
            fingerprint=self.fingerprint,
            args_json=args_json,
            budget_json=runs.encode_budget(budget),
        )
        return _runtime.run_sync(
            _aexecute_flow(
                self, run_id=run_id, trace_id=trace_id, args=args, kwargs=kwargs, budget=budget
            )
        )

    def stream(self, *args: Any, budget: Budget | None = None, **kwargs: Any) -> RunStream:
        return agentfn._stream_run(lambda: self.run(*args, budget=budget, **kwargs))


@overload
def flow(fn: Callable[..., Any]) -> Flow: ...


@overload
def flow(
    *, name: str | None = None, replace: bool = False
) -> Callable[[Callable[..., Any]], Flow]: ...


def flow(
    fn: Callable[..., Any] | None = None, *, name: str | None = None, replace: bool = False
) -> Flow | Callable[[Callable[..., Any]], Flow]:
    """Decorate a plain function into a :class:`Flow` -- a durable, journaled run.

    Usable bare (``@flow``) or with arguments (``@flow(name=...)``).
    Registers ``name -> Flow`` in a module-level registry (duplicate names
    raise :class:`~composeai.errors.ConfigError`) so :func:`resume` can
    look a flow up by name from a run row.

    ``replace=True`` re-binds an existing flow name instead of raising
    ``ConfigError``. Safe for already-paused/journaled runs: a replaced
    flow's changed source still gets caught on resume by the fingerprint
    check (:class:`~composeai.errors.ResumeMismatchError`, unless
    ``allow_code_change=True`` -- see :func:`resume`).
    """

    def decorator(f: Callable[..., Any]) -> Flow:
        return Flow(f, name=name, replace=replace)

    if fn is not None:
        return decorator(fn)
    return decorator


async def _aexecute_flow(
    flow_obj: Flow,
    *,
    run_id: str,
    trace_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: Budget | None,
) -> Run:
    """Async core of the flow engine (v0.4.0 Plan A, Task 9) -- mirrors the
    former ``_execute_flow`` exactly; ``Flow.run()``/``resume()`` are its
    only two callers, both sync facades dispatching via
    ``_runtime.run_sync(_aexecute_flow(...))`` (``create_run`` itself stays
    caller-side/sync in both, exactly as before -- this function never
    creates the row, only ever updates one that already exists).

    Every direct store call this function makes itself (as opposed to a
    span's own persistence on exit, via ``tracing``'s module-global sink --
    that stays synchronous on this coroutine's own thread, unchanged from
    every other async engine core in this codebase; see
    ``composeai.agentfn``'s identical precedent) routes through
    ``worker_for(store).call(...)`` -- the store's dedicated writer thread
    (:mod:`composeai._storeasync`) -- so a durable flow run driven through
    the async engine (nested inside a ``pipe``/``aggregate``/``map``, or a
    top-level ``.run()``/``resume()`` dispatched via ``_runtime.run_sync``)
    never blocks the composeai runtime loop with SQLite I/O: the initial
    journal preload (``journal_all``) and budget baseline
    (``prior_llm_usage``), and every ``update_run``/pause-persistence write
    below.

    The flow BODY itself stays a sync user function in this plan (v0.4.0
    Plan A) -- it runs via ``_dispatch.run_stage(flow_obj._fn, ...)``, which
    dispatches a plain sync callable onto its OWN dedicated daemon thread
    (see ``_dispatch._run_sync_on_own_thread``): every ``@task``/``@agent``/
    ``now()``/``random()``/``approve()`` call made from inside the body
    reaches the sync facades on that thread -- legal re-entry into the
    runtime loop from a non-runtime thread (see ``_runtime.run_sync``'s own
    re-entry guard, which only refuses re-entry from the runtime loop's OWN
    thread, never from an arbitrary worker thread like this one); documented
    extra latency, never a deadlock, same as every other sync-facade-from-a-
    worker-thread call already exercised elsewhere in this codebase (e.g.
    ``test_nested_sync_agent_call_from_tool_body_no_deadlock``).
    ``composeai.hitl._Pause`` raised from the body is a ``BaseException``;
    ``_run_sync_on_own_thread``'s worker resolves the awaited future with
    ANY ``BaseException`` the callable raised (its own ``except
    BaseException`` clause, not ``except Exception``), so it propagates
    through the ``await run_stage(...)`` below completely unchanged and
    reaches this function's own ``except BaseException`` immediately after
    -- exactly as it did when the sync ``_execute_flow`` called
    ``flow_obj._fn`` directly on its own (the caller's) thread with no
    bridge in between at all.
    """
    store = _open_default_store()
    preloaded = await worker_for(store).call("journal_all", run_id)
    ctx = RunContext(run_id=run_id, store=store, preloaded=preloaded)

    # Budget enforcement is cumulative across attempts: seed with spend
    # already persisted by earlier attempts of this run (zero for a fresh
    # run -- no spans exist yet; skipped entirely when there's no budget).
    baseline = (
        await worker_for(store).call("prior_llm_usage", run_id) if budget is not None else None
    )

    with use_run_context(ctx), tracing.use_trace(trace_id):
        root_span: tracing.Span | None = None
        try:
            with tracing.span("flow", flow_obj.name) as root_span:
                with budget_scope(budget, root_span, baseline=baseline):
                    output = await run_stage(
                        flow_obj._fn, args, kwargs, timeout=None, name=flow_obj.name, kind="@flow"
                    )
                trace = tracing.current_trace()
                assert trace is not None
                usage = trace.rollup_usage(root_span)
                run = Run(
                    id=run_id,
                    status="completed",
                    output=output,
                    usage=usage,
                    trace=trace,
                    messages=[],
                    pending=None,
                )
        except BaseException as exc:
            # Phase 8 (human-in-the-loop): a pause is not a failure -- `approve()`/
            # `ask_human()` (directly in the flow body, in a nested @task, or
            # propagating up from a nested @agent's tool call) raise
            # `composeai.hitl._Pause`, duck-typed-detected here (this module
            # cannot import `composeai.hitl`, which imports `composeai.runs` for
            # the ambient-run-context helpers `_aexecute_flow` itself uses --
            # importing it back here would cycle). The flow *returns* a paused
            # `Run` instead of raising; the process may exit right after.
            if getattr(exc, "_compose_pause", False):
                # `exc.interrupt` (duck-typed -- this module doesn't import
                # `composeai.hitl.Interrupt`, so pyright can't check the
                # attribute access statically either).
                interrupt = getattr(exc, "interrupt")  # noqa: B009
                await apersist_pending_interrupt(store, run_id, interrupt)
                if root_span is not None:
                    tracing.emit_run_finished(root_span, status="paused")
                trace = tracing.current_trace()
                assert trace is not None
                return Run(
                    id=run_id,
                    status="paused",
                    output=None,
                    usage=Usage(),
                    trace=trace,
                    messages=[],
                    pending=interrupt,
                )
            now = time.time()
            await worker_for(store).call(
                "update_run",
                run_id,
                status="failed",
                updated_at=now,
                error_json=json.dumps({"type": type(exc).__name__, "message": str(exc)}),
            )
            if root_span is not None:
                tracing.emit_run_finished(root_span, status="failed", error_type=type(exc).__name__)
            raise

        now = time.time()
        await worker_for(store).call(
            "update_run",
            run_id,
            status="completed",
            updated_at=now,
            output_json=json.dumps(to_jsonable(output)),
        )
        tracing.emit_run_finished(root_span, status="completed")
        return run


def _run_flow_journaled(
    flow_obj: Flow, ctx: JournalScope, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Run:
    """Treat one whole nested ``@flow`` call (invoked from inside another
    active ``@flow`` body) as a single journaled step of the ENCLOSING flow.

    Mirrors ``composeai.agentfn._run_agent_journaled`` for the identical
    reason: without this, resuming the outer flow would re-execute the
    inner flow's entire body -- and every ``@task``/``@agent`` call inside
    it, including paid LLM calls -- from scratch on every attempt, even
    after it had already completed successfully once.

    Replay (journal hit): build a completed ``Run`` straight from the
    decoded stored value -- the inner flow's body never runs again. Miss:
    reserve this step's own scope segment (so every ``@task``/``@agent``/
    nested-``@flow`` call made from inside the inner flow's body is keyed
    under it -- see ``composeai.runs``'s scope-stack module docs -- and can
    never collide with the outer flow's own step names), run the inner
    flow's body for real inside a nested ``"flow"`` span (same trace, no
    new ``run_id``/``trace_id``/``runs`` row -- this is one step of the
    OUTER run, not a durable run of its own), and journal its output.

    A pause (``composeai.hitl._Pause``) raised from inside the inner flow's
    body is deliberately not caught here -- same as a nested ``@task``
    call, it simply propagates to the outermost flow's own pause handling
    (see ``_aexecute_flow``'s ``except`` clause); nothing is journaled for
    this step (a miss again next time), and whatever the inner flow itself
    journaled up to the pause point is unaffected (it shares the same
    journal, just under its own scoped keys).
    """
    segment = ctx.reserve_scope_segment(flow_obj.name)
    key = ctx.qualify(segment)
    hit, value = ctx.journal_lookup(key)
    if hit:
        with tracing.span("flow", flow_obj.name, attributes={"step_key": key}) as flow_span:
            flow_span.replayed = True
            flow_span.set_output(value)
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

    with tracing.span("flow", flow_obj.name, attributes={"step_key": key}) as flow_span:
        with runs.push_scope(segment):
            output = flow_obj._fn(*args, **kwargs)
        flow_span.set_output(output)
    recorded = ctx.journal_record(key, output)
    trace = tracing.current_trace()
    assert trace is not None
    return Run(
        id=new_ulid(),
        status="completed",
        output=recorded,
        usage=trace.rollup_usage(flow_span),
        trace=trace,
        messages=[],
        pending=None,
    )


# --- resume ---------------------------------------------------------------


def resume(
    run_id: str,
    answers: dict[str, Any] | None = None,
    *,
    budget: Budget | None = None,
    allow_code_change: bool = False,
) -> Run:
    """Resume a durable run (``@flow`` or standalone ``@agent``) by ``run_id``.

    The one entry point for every kind of durable run, in this process or a
    fresh one. ``answers`` (if any) are journaled under ``f"__interrupt__:{id}"``
    keys -- a bare tool name resolves to a pending ``tool:{name}:{call_id}``
    interrupt when unambiguous (see
    :func:`~composeai.runs.resolve_answer_key`) -- so that whichever
    ``approve()``/``ask_human()``/approval-gated tool call raised the pause
    finds its answer already there when the run/agent body re-executes.
    Journaling an answer is a durable, first-write-wins commit, so it only
    happens *after* every check below that could otherwise abort the
    resume without re-executing anything -- an answer given alongside a
    resume that turns out to be impossible (unregistered flow, fingerprint
    mismatch, ...) must never be permanently locked in.

    - Missing run: :class:`~composeai.errors.ConfigError`.
    - Run row ``kind == "agent"``: routed to
      ``composeai.agentfn.resume_standalone_agent`` (looks the agent up by
      name in its own registry, restores its saved conversation state, and
      continues the loop -- or re-pauses on the next unanswered interrupt).
    - Otherwise (``kind == "flow"``):
      - Flow not registered in this process (its defining module was never
        imported): :class:`~composeai.errors.ConfigError` naming the fix.
      - Already ``"completed"``: returns the stored output as a completed
        ``Run`` without re-executing anything.
      - Source changed since the run started (fingerprint mismatch):
        :class:`~composeai.errors.ResumeMismatchError`, unless
        ``allow_code_change=True``.
      - Otherwise (``"running"``, ``"paused"``, or ``"failed"``):
        re-executes the flow body with the *same* ``run_id`` and *same*
        ``trace_id`` (so resumed spans join the original trace); journaled
        steps replay, the rest actually runs -- an unanswered (or newly
        reached) interrupt pauses again, with the same ``Run`` shape. The
        run's original ``budget`` (if any -- see ``Flow.run``) is decoded
        from the row and reapplied for this attempt too. ``budget`` (if
        given) replaces the run's stored budget for this attempt and every
        later one -- persisted as a plain last-write-wins column update
        (deliberately not journaled: the journal is first-write-wins and
        would freeze the first override). Prior attempts' real spend still
        counts against the new cap. ``None`` keeps the stored budget;
        clearing one via resume is not supported.
    """
    store = _open_default_store()
    row = store.get_run(run_id)
    if row is None:
        raise ConfigError(f"no run found with run_id {run_id!r}")

    if row["kind"] == "agent":
        return agentfn.resume_standalone_agent(run_id, row, answers, budget=budget)

    flow_obj = _FLOW_REGISTRY.get(row["name"])
    if flow_obj is None:
        raise ConfigError(
            f"flow {row['name']!r} is not registered in this process -- import the "
            "module that defines it (so its @flow decoration runs) before calling resume()"
        )

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

    if row["fingerprint"] != flow_obj.fingerprint and not allow_code_change:
        raise ResumeMismatchError(
            f"@flow {flow_obj.name!r}'s source changed since run {run_id!r} was created "
            "(fingerprint mismatch) -- pass allow_code_change=True to resume anyway "
            "(the journal may no longer match the new code's call sequence)"
        )

    if budget is not None:
        # Persist AFTER every abort-check above, same reasoning as answers:
        # an override alongside an impossible resume must not be locked in.
        store.update_run(
            run_id, budget_json=runs.encode_budget(budget), updated_at=time.time()
        )
    apply_resume_answers(store, run_id, answers)
    args, kwargs = _decode_call(row["args_json"])
    effective_budget = budget if budget is not None else runs.decode_budget(row.get("budget_json"))
    return _runtime.run_sync(
        _aexecute_flow(
            flow_obj,
            run_id=run_id,
            trace_id=row["trace_id"],
            args=args,
            kwargs=kwargs,
            budget=effective_budget,
        )
    )


__all__ = ["Flow", "Task", "flow", "now", "random", "resume", "task"]
