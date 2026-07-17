"""``Run``: the result of executing an ``@agent`` function.

A durable ``RunStore`` (SQLite-backed) arrives in Phase 7; this module only
defines the in-memory shape returned by :meth:`~composeai.agentfn.AgentFunction.run`.

This module also defines :class:`Budget` and the machinery that enforces
it: :func:`budget_scope` pushes a ``(budget, root_span, baseline)`` triple
onto a contextvar stack for the duration of a run -- ``baseline`` is spend
from earlier attempts of the same durable run (``None`` for a fresh run),
added on top of the current attempt's in-memory rollup so a resumed run's
budget caps lifetime spend rather than just the latest attempt's -- and
:func:`check_budgets` -- called by :mod:`composeai.agentfn` right after
every LLM call -- walks that stack and raises
:class:`~composeai.errors.BudgetExceededError` the moment any active entry
is exceeded. Living here (rather than in ``agentfn.py`` or ``tracing.py``)
keeps it usable by both ``@agent`` runs and the ``pipe``/``aggregate``
combinators without either module depending on the other.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import sqlite3
import threading
import time
import warnings
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, runtime_checkable

from typing_extensions import TypeVar

from . import tracing
from ._encoding import from_jsonable, to_jsonable
from ._ids import new_ulid
from .errors import BudgetExceededError, ConfigError, SerializationError, TaskTimeoutError
from .events import AsyncSubscription, Event, Subscription
from .messages import Message, Usage
from .tracing import Span, Trace, current_trace

if TYPE_CHECKING:
    # Only for type annotations (`Run.pending`) -- `composeai.hitl` imports
    # this module (for the ambient-run-context helpers below), so importing
    # it back here at runtime would cycle. `from __future__ import
    # annotations` (above) means the annotation is never evaluated at
    # runtime anyway; this import only serves static type checkers.
    from .hitl import Interrupt

# `default=Any` (PEP 696, via `typing_extensions` since this repo's floor is
# py>=3.10 -- native `typing.TypeVar(default=...)` only exists from 3.13) is
# what makes a *bare* `Run`/`RunStream`/`AsyncRunStream` annotation mean
# `Run[Any]` rather than tripping `reportMissingTypeArgument` under strict
# pyright -- verified in
# plans/superpowers/research/2026-07-16-typing/prototype/probe_run_default.py.
# composeai's own ~40 internal producer/consumer sites (agentfn.py,
# combinators.py, flow.py, this module) are re-annotated `Run[Any]`
# explicitly below regardless, since they're pyright-only concerned with
# suppressing the reportMissingTypeArgument the moment strict is ever
# turned on -- the default only has to carry *external* bare usage.
R = TypeVar("R", default=Any)


@dataclass
class Run(Generic[R]):
    """The outcome of one ``@agent`` invocation.

    ``output`` is typed ``R`` -- the *completed*-run output type; a paused or
    failed run has none, so discriminate those via ``status``/``pending``
    rather than guarding on ``output`` (see docs/typing.md).

    ``pending`` is set (to the :class:`~composeai.hitl.Interrupt` that
    caused it) exactly when ``status == "paused"`` -- ``None`` otherwise.
    Pausing is produced by ``@flow.run()``/``resume()`` and by ``@agent``
    runs (standalone or nested in a flow) when ``approve()``/``ask_human()``
    (or an unanswered ``@tool(requires_approval=True)`` call) has no
    journaled answer yet; it is not a failure -- the process may exit right
    after receiving a paused ``Run``.
    """

    id: str
    status: Literal["completed", "paused", "failed"]
    output: R
    usage: Usage
    trace: Trace
    messages: list[Message]
    pending: Interrupt | None = None


class RunStream(Generic[R]):
    """Iterable of :class:`~composeai.events.Event` for one streaming ``@agent`` call.

    Constructed by :meth:`~composeai.agentfn.AgentFunction.stream` -- never
    directly by users. The agent loop runs in a background worker thread
    against a fresh, private :class:`~composeai.events.EventBus`; this object
    is a read-only view onto that bus (iterable) plus a handle onto the
    thread (:attr:`run`).

    Iterating blocks until an event arrives or the run finishes, and stops
    after the terminal ``run_finished`` event. If the loop raised, iteration
    re-raises that *same* exception object once the subscription is drained.
    Breaking out of iteration early (or calling :meth:`close`) only
    unsubscribes -- the worker thread is never disturbed by it, so it always
    runs to completion and :attr:`run` always joins successfully (raising
    the run's exception if it failed).
    """

    def __init__(self, subscription: Subscription, thread: threading.Thread) -> None:
        self._subscription = subscription
        self._thread = thread
        self._run: Run[R] | None = None
        self._exception: BaseException | None = None

    def __iter__(self) -> Iterator[Event]:
        yield from self._subscription
        self._thread.join()
        if self._exception is not None:
            raise self._exception

    def close(self) -> None:
        """Unsubscribe from the underlying bus, ending iteration.

        Safe to call more than once, and does not affect the worker thread
        (still joinable via :attr:`run` regardless).
        """
        self._subscription.close()

    def _set_outcome(self, *, run: Run[R] | None, exception: BaseException | None) -> None:
        """Record the worker thread's result. Called once, by that thread itself."""
        self._run = run
        self._exception = exception

    @property
    def run(self) -> Run[R]:
        """Join the worker thread and return its :class:`Run`.

        Re-raises the loop's original exception (same object) if it failed.
        """
        self._thread.join()
        if self._exception is not None:
            raise self._exception
        assert self._run is not None  # the thread always sets one or the other
        return self._run


class AsyncRunStream(Generic[R]):
    """Asyncio-native twin of :class:`RunStream` (v0.4.0 Plan B).

    Constructed by the async engine's streaming entry point -- never
    directly by users. The agent loop runs as an ``asyncio.Task`` against a
    fresh, private :class:`~composeai.events.EventBus`; this object is a
    read-only view onto that bus (``async for``) plus a handle onto the
    task (:meth:`run`).

    ``async for`` blocks until an event arrives or the subscription closes.
    The producing task closes its :class:`~composeai.events.AsyncSubscription`
    in a ``finally``, exactly where the sync worker closes its own
    :class:`~composeai.events.Subscription` -- so breaking out of iteration
    early (or calling :meth:`close`) only unsubscribes; the task itself is
    never disturbed by it, and always runs to completion. Because
    ``asyncio.Task`` memoizes its own result, :meth:`run` after iterating
    (or iterating twice) always sees the same, stable outcome -- awaiting
    it re-raises the task's original exception verbatim if it failed.

    GC-warning asymmetry vs. :class:`RunStream`: if a failed run's task is
    abandoned -- never awaited via :meth:`run` and never fully iterated --
    asyncio prints "Task exception was never retrieved" when that task is
    garbage-collected (its exception was never retrieved, exactly as the
    warning says). :class:`RunStream`'s background *thread* has no such
    warning: an unjoined thread just silently keeps running. Await
    :meth:`run` or iterate to completion to avoid it.
    """

    def __init__(self, task: asyncio.Task[Run[R]], subscription: AsyncSubscription) -> None:
        self._task = task
        self._subscription = subscription

    async def __aiter__(self) -> AsyncIterator[Event]:
        async for event in self._subscription:
            yield event

    def close(self) -> None:
        """Unsubscribe from the underlying bus, ending iteration.

        Safe to call more than once, and does not affect the task (still
        awaitable via :meth:`run` regardless).
        """
        self._subscription.close()

    async def run(self) -> Run[R]:
        """Await the task and return its :class:`Run`.

        Re-raises the task's original exception verbatim if it failed.
        """
        return await self._task


# --- Budget --------------------------------------------------------------


@dataclass
class Budget:
    """A spend cap enforced across every LLM call within a run's subtree.

    ``tokens`` is input+output tokens combined. At least one of ``usd``/
    ``tokens`` must be set (else :class:`~composeai.errors.ConfigError`) --
    an empty ``Budget()`` couldn't enforce anything.

    USD enforcement only counts calls with a known cost: an adapter that
    can't price a call reports ``Usage(cost_usd=None)``, and that call's
    cost is treated as ``0`` for budgeting purposes (see
    :func:`check_budgets`) -- a ``usd`` budget simply can't see spend it has
    no price for. Pass ``tokens`` too if you need a hard cap regardless of
    pricing.

    Enforcement is cumulative across ``resume()`` attempts: spend persisted
    by earlier attempts of the same run counts against the cap (replayed
    steps themselves cost nothing).
    """

    usd: float | None = None
    tokens: int | None = None

    def __post_init__(self) -> None:
        if self.usd is None and self.tokens is None:
            raise ConfigError(
                "Budget requires at least one of `usd` or `tokens` to be set"
            )


def encode_budget(budget: Budget | None) -> str | None:
    """JSON-encode ``budget`` for the ``runs.budget_json`` column, or ``None``."""
    if budget is None:
        return None
    return json.dumps({"usd": budget.usd, "tokens": budget.tokens})


def decode_budget(budget_json: str | None) -> Budget | None:
    """Inverse of :func:`encode_budget`."""
    if not budget_json:
        return None
    payload = json.loads(budget_json)
    return Budget(usd=payload.get("usd"), tokens=payload.get("tokens"))


_budget_stack: contextvars.ContextVar[tuple[tuple[Budget, Span, Usage | None], ...]] = (
    contextvars.ContextVar("_budget_stack", default=())
)


@contextmanager
def budget_scope(
    budget: Budget | None, root_span: Span, baseline: Usage | None = None
) -> Iterator[None]:
    """Push ``(budget, root_span, baseline)`` onto the active-budgets stack.

    ``baseline`` is spend from *earlier attempts* of the same durable run
    (persisted llm spans -- see :meth:`RunStore.prior_llm_usage`), added on
    top of the current attempt's in-memory rollup so a ``Budget`` caps a
    run's lifetime spend, not each attempt's. A no-op when ``budget`` is
    ``None``.
    """
    if budget is None:
        yield
        return
    stack = _budget_stack.get()
    token = _budget_stack.set((*stack, (budget, root_span, baseline)))
    try:
        yield
    finally:
        _budget_stack.reset(token)


def check_budgets() -> None:
    """Raise :class:`~composeai.errors.BudgetExceededError` if any active budget was exceeded.

    Checks every entry on the stack (innermost and outermost alike -- a
    nested, tighter budget and an enclosing, looser one are both live at
    once), computing each entry's usage as
    ``current_trace().rollup_usage(root_span)`` plus that entry's
    ``baseline`` (spend from earlier resume attempts, if any). A no-op when
    there is no active trace or no active budget.
    """
    trace = current_trace()
    if trace is None:
        return
    for budget, root_span, baseline in _budget_stack.get():
        used = trace.rollup_usage(root_span)
        if baseline is not None:
            used = used + baseline
        if budget.tokens is not None:
            total_tokens = used.input_tokens + used.output_tokens
            if total_tokens > budget.tokens:
                raise BudgetExceededError(
                    f"budget exceeded on {root_span.kind} {root_span.name!r}: "
                    f"tokens budget={budget.tokens}, used={total_tokens}"
                )
        if budget.usd is not None:
            used_usd = used.cost_usd or 0.0
            if used_usd > budget.usd:
                raise BudgetExceededError(
                    f"budget exceeded on {root_span.kind} {root_span.name!r}: "
                    f"usd budget={budget.usd}, used=${used_usd:.4f}"
                )


def _run_with_timeout(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    timeout: float,
    name: str,
    *,
    kind: str = "@task",
) -> Any:
    """Run ``fn(*args, **kwargs)`` on a dedicated daemon thread, bounded by ``timeout``.

    Deliberately a raw daemon :class:`threading.Thread`, not a
    ``concurrent.futures.ThreadPoolExecutor``: CPython registers an
    ``atexit`` hook that joins every executor worker thread at interpreter
    shutdown, which would hang the process forever waiting on a task
    that's genuinely stuck (e.g. an infinite loop) -- exactly the case a
    timeout exists to route around. A daemon thread carries none of that
    baggage: on timeout this raises immediately and the abandoned thread
    keeps running in the background (harvested only when it finishes on
    its own, or the process exits).

    That abandoned thread must not keep mutating the run's journal after
    the caller has moved on (the run might already be marked "completed" by
    then, if the caller caught ``TaskTimeoutError`` and continued -- or a
    fresh ``resume()`` might later build an independent ``RunContext`` whose
    zeroed-out counters can legitimately collide with the zombie's own).
    So: if there's an ambient :class:`~composeai.runs.RunContext`, the
    worker thread's *own copied context* (never the caller's) gets a guard
    proxy installed in its place (see
    :func:`~composeai.runs.abandon_guard`/:func:`~composeai.runs.use_journal_scope`)
    with an ``abandoned`` event this function sets the instant it gives up
    waiting -- any journal touch after that point raises internally inside
    the worker and unwinds its stack at that point. Pure side effects
    already in flight (a network call, a file write, ...) still run to
    completion regardless -- there is no safe way to interrupt an arbitrary
    Python thread; that part is inherent, not something this guard changes.
    """
    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}
    done = threading.Event()
    abandoned = threading.Event()

    def _worker() -> None:
        ctx = current_run_context()
        guarded = abandon_guard(ctx, abandoned)
        with use_journal_scope(guarded):
            try:
                result["value"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 -- forwarded to the caller thread below
                error["exc"] = exc
            finally:
                done.set()

    thread = threading.Thread(target=tracing.propagate(_worker), daemon=True)
    thread.start()
    if not done.wait(timeout=timeout):
        abandoned.set()
        raise TaskTimeoutError(
            f"{kind} {name!r} exceeded timeout={timeout}s -- the abandoned thread keeps "
            "running in the background as a daemon; its eventual result is discarded, "
            "and it loses journal write access immediately (its pure side effects, if "
            "any, still run to completion -- there is no safe way to stop that)"
        )
    if "exc" in error:
        raise error["exc"]
    return result["value"]


# --- RunStore: the SQLite-backed durable store (Phase 7) ------------------
#
# One connection per thread (a `threading.local`), each with its own
# schema-init pass (idempotent `CREATE TABLE IF NOT EXISTS`) and its own
# `PRAGMA journal_mode=WAL`. WAL is a database-level setting that persists
# in the file itself once set, so re-issuing the pragma on every new
# connection is just a cheap no-op after the first -- it's done everywhere
# rather than only in `__init__` so a `RunStore` used from multiple threads
# never has a thread accidentally holding a connection from *before* WAL
# was durably set.
#
# Threading model: confining each connection to the thread that created it
# (rather than passing `check_same_thread=False` and sharing one
# connection) means the sqlite3 module's own thread-affinity check stays
# on as a safety net against accidental cross-thread reuse -- correctness
# instead relies on SQLite's own file-level locking (WAL readers don't
# block the writer; two writers serialize via the busy timeout below) for
# the actual concurrent-access guarantees this module needs (e.g. two
# `RunStore` handles on the same file racing a `journal_put`).


class RunStore:
    """SQLite-backed durable store for runs, their journal, and their spans."""

    #: The schema version this code writes/expects (see :meth:`_init_schema`
    #: and the ``PRAGMA user_version`` guard). Bump this -- and add a real
    #: migration path -- the day the ``spans``/``runs``/... table shapes
    #: below actually change; until then, "migration" just means refusing
    #: to open a store some other version wrote.
    _SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._fts_available = True
        self._fts_insert_warned = False
        self._connect()  # eagerly init schema in the constructing thread

    def close(self) -> None:
        """Close *this thread's* connection, if one was opened.

        A `RunStore` used from multiple threads leaves other threads'
        connections open -- there is no cross-thread handle to close from
        here. Fine for this codebase's usage (short-lived test processes,
        or a long-lived process where connections live for its duration).
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema(conn)
        self._local.conn = conn
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        # Schema-migration guard: a `runs.db` some *other* schema version
        # wrote (today, that only means "predates PRAGMA user_version
        # tracking entirely" -- there's only ever been one released shape)
        # is refused outright, loudly, rather than opened and silently
        # breaking the first time code tries to read/write a column that
        # version's tables don't have. Reproduced concretely before this
        # guard existed: an older-shape `spans` table missing `cost_usd`/
        # `replayed` raised a raw `sqlite3.OperationalError` from deep
        # inside `persist_span`, which the store worker's one-time-warning
        # error handling (`_storeasync._warn_once_on_cast_failure`) then
        # converted into "tracing silently goes dark forever after one
        # RuntimeWarning" -- no migration, no clear
        # error, just quietly missing data from then on. Checked *before*
        # the `CREATE TABLE IF NOT EXISTS` below runs, so a mismatched
        # store is never mutated on the way to raising. A brand-new
        # (no `runs` table yet) database has nothing to check -- it always
        # proceeds to create fresh, current-version tables. Real migrations
        # (rewriting an old-version store forward) are deliberately out of
        # scope for now -- this project has never shipped a release with a
        # different schema, so there is nothing to migrate *from* yet.
        preexisting = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
        ).fetchone()
        if preexisting is not None:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version != self._SCHEMA_VERSION:
                raise ConfigError(
                    f"{self._path} was created with schema user_version={version}, "
                    f"but this version of composeai expects {self._SCHEMA_VERSION} -- "
                    "in-place schema migration isn't supported yet. Point COMPOSE_DIR "
                    "at a fresh, empty directory (or delete this runs.db) to continue."
                )
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                kind TEXT,
                name TEXT,
                status TEXT,
                created_at REAL,
                updated_at REAL,
                trace_id TEXT,
                fingerprint TEXT,
                args_json TEXT,
                output_json TEXT,
                error_json TEXT,
                budget_json TEXT
            );
            CREATE TABLE IF NOT EXISTS journal (
                run_id TEXT,
                step_key TEXT,
                value_json TEXT,
                created_at REAL,
                PRIMARY KEY (run_id, step_key)
            );
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT,
                run_id TEXT,
                parent_span_id TEXT,
                kind TEXT,
                name TEXT,
                started_at REAL,
                ended_at REAL,
                status TEXT,
                replayed INTEGER,
                attributes_json TEXT,
                usage_json TEXT,
                cost_usd REAL,
                error_json TEXT
            );
            CREATE TABLE IF NOT EXISTS span_payloads (
                span_id TEXT PRIMARY KEY,
                input_json TEXT,
                output_json TEXT
            );
            CREATE TABLE IF NOT EXISTS pending_interrupts (
                run_id TEXT,
                interrupt_id TEXT,
                kind TEXT,
                question TEXT,
                payload_json TEXT,
                created_at REAL,
                PRIMARY KEY (run_id, interrupt_id)
            );
            CREATE TABLE IF NOT EXISTS agent_state (
                run_id TEXT,
                scope_key TEXT,
                messages_json TEXT,
                partial_results_json TEXT,
                turn INTEGER,
                updated_at REAL,
                PRIMARY KEY (run_id, scope_key)
            );
            """
        )
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS span_fts "
                "USING fts5(span_id UNINDEXED, name, content)"
            )
        except sqlite3.OperationalError:
            # FTS5 not compiled into this sqlite3 build: feature-detect and
            # degrade -- search just isn't available, nothing else breaks.
            # This is the *only* failure that should ever disable FTS for
            # the store's whole lifetime (see `persist_span`'s per-insert
            # handling below, which deliberately does NOT do this for a
            # single failed insert).
            self._fts_available = False
        conn.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
        conn.commit()

    # --- runs table ---------------------------------------------------

    # Every column `update_run` is allowed to SET, besides `run_id` itself
    # (the WHERE key, never a settable field). `update_run`'s SET clause is
    # built by interpolating `fields`' *keys* (not just its values) into
    # the SQL text -- the values are parameterized, but a column name can't
    # be a bind parameter, so this allow-list is what keeps an
    # attacker/externally-influenced key (e.g. a future API/CLI wrapper
    # that forwards caller-chosen field names) from being interpolated as
    # arbitrary SQL instead of a column name.
    _UPDATABLE_RUN_COLUMNS = frozenset(
        {
            "kind",
            "name",
            "status",
            "created_at",
            "updated_at",
            "trace_id",
            "fingerprint",
            "args_json",
            "output_json",
            "error_json",
            "budget_json",
        }
    )

    def create_run(
        self,
        *,
        run_id: str,
        kind: str,
        name: str,
        status: str,
        created_at: float,
        updated_at: float,
        trace_id: str | None,
        fingerprint: str | None,
        args_json: str | None,
        output_json: str | None = None,
        error_json: str | None = None,
        budget_json: str | None = None,
    ) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO runs (run_id, kind, name, status, created_at, updated_at, "
            "trace_id, fingerprint, args_json, output_json, error_json, budget_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                kind,
                name,
                status,
                created_at,
                updated_at,
                trace_id,
                fingerprint,
                args_json,
                output_json,
                error_json,
                budget_json,
            ),
        )
        conn.commit()

    def update_run(self, run_id: str, **fields: Any) -> None:
        """Update arbitrary columns of the ``runs`` row for ``run_id``.

        ``updated_at`` defaults to ``time.time()`` if not supplied. A no-op
        (no UPDATE issued) when ``fields`` is empty.

        Raises :class:`~composeai.errors.ConfigError` if ``fields`` names
        anything outside :attr:`_UPDATABLE_RUN_COLUMNS` -- every current
        in-repo call site only ever passes hardcoded literal keyword names,
        but this method builds its SQL ``SET`` clause from ``fields``'
        *keys*, which can't be parameterized the way values can; an
        allow-list is what keeps a future caller that forwards an
        externally-influenced field name (e.g. an API/CLI wrapper) from
        being able to inject SQL through the column name instead of the
        value.
        """
        if not fields:
            return
        invalid = fields.keys() - self._UPDATABLE_RUN_COLUMNS
        if invalid:
            raise ConfigError(
                f"update_run: not updatable column(s) {sorted(invalid)!r} -- "
                f"allowed: {sorted(self._UPDATABLE_RUN_COLUMNS)!r}"
            )
        fields.setdefault("updated_at", time.time())
        conn = self._connect()
        set_clause = ", ".join(f"{key} = ?" for key in fields)
        conn.execute(
            f"UPDATE runs SET {set_clause} WHERE run_id = ?",  # noqa: S608 -- keys checked against _UPDATABLE_RUN_COLUMNS above
            (*fields.values(), run_id),
        )
        conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def list_runs(
        self, *, kind: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        conn = self._connect()
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        query = "SELECT * FROM runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # --- journal --------------------------------------------------------

    def journal_get(self, run_id: str, step_key: str) -> str | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT value_json FROM journal WHERE run_id = ? AND step_key = ?",
            (run_id, step_key),
        ).fetchone()
        return row["value_json"] if row is not None else None

    def journal_all(self, run_id: str) -> dict[str, str]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT step_key, value_json FROM journal WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {row["step_key"]: row["value_json"] for row in rows}

    def prior_llm_usage(self, run_id: str) -> Usage:
        """Sum of every persisted llm span's usage for ``run_id``.

        This is spend from *earlier attempts*: replayed steps never re-create
        llm spans, and the current attempt's spans are persisted only as they
        finish (after this is read at attempt start), so it never
        double-counts. Used to seed :func:`budget_scope` on resume.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT usage_json FROM spans WHERE run_id = ? AND kind = 'llm' "
            "AND usage_json IS NOT NULL",
            (run_id,),
        ).fetchall()
        total = Usage()
        for row in rows:
            payload = json.loads(row["usage_json"])
            # usage_json is to_jsonable(Usage): {"$kind": "pydantic",
            # "$type": "composeai.messages:Usage", "value": {...}} --
            # validated directly rather than via from_jsonable so this
            # never depends on registry state in a fresh process.
            value = payload.get("value") if isinstance(payload, dict) else None
            if isinstance(value, dict):
                total = total + Usage.model_validate(value)
        return total

    def prior_llm_usage_for_step(self, run_id: str, step_key: str) -> Usage:
        """Sum of persisted llm-span usage under agent spans tagged ``step_key``.

        The per-step analog of :meth:`prior_llm_usage`, for a nested
        ``@agent(budget=...)`` call inside a resumed flow: only THAT step's
        earlier attempts may count against the agent's own budget --
        charging the whole run's spend here would over-count. Agent spans
        carry their journal key in ``attributes_json["step_key"]``; their
        descendant llm spans hold the usage. Replayed steps re-create no
        llm spans, so replays contribute nothing.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT span_id, parent_span_id, kind, usage_json, attributes_json "
            "FROM spans WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        children: dict[str | None, list[Any]] = {}
        matching_roots: list[str] = []
        for row in rows:
            children.setdefault(row["parent_span_id"], []).append(row)
            if row["kind"] == "agent" and row["attributes_json"]:
                try:
                    attributes = json.loads(row["attributes_json"])
                except json.JSONDecodeError:
                    continue
                if attributes.get("step_key") == step_key:
                    matching_roots.append(row["span_id"])

        total = Usage()
        stack = list(matching_roots)
        while stack:
            span_id = stack.pop()
            for child in children.get(span_id, []):
                stack.append(child["span_id"])
                if child["kind"] == "llm" and child["usage_json"]:
                    payload = json.loads(child["usage_json"])
                    value = payload.get("value") if isinstance(payload, dict) else None
                    if isinstance(value, dict):
                        total = total + Usage.model_validate(value)
        return total

    def journal_put(self, run_id: str, step_key: str, value_json: str) -> str:
        """INSERT OR IGNORE (first-write-wins); return the value that ended up stored.

        On a race (two writers, same ``(run_id, step_key)``), the insert is
        ignored for whichever writer loses, and that writer's returned
        value is the *winner's* -- read back from the table -- never its
        own. Callers must decode this return value rather than assuming
        their own ``value_json`` was the one persisted.
        """
        conn = self._connect()
        cursor = conn.execute(
            "INSERT OR IGNORE INTO journal (run_id, step_key, value_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (run_id, step_key, value_json, time.time()),
        )
        conn.commit()
        if cursor.rowcount == 0:
            winner = self.journal_get(run_id, step_key)
            assert winner is not None  # the row must exist: that's why rowcount was 0
            return winner
        return value_json

    # --- spans: hot/cold split + FTS -------------------------------------

    def persist_span(self, span: Span, run_id: str | None) -> None:
        """Persist ``span`` (hot row + cold payload row + FTS row if available).

        ``run_id`` is ``None`` for spans outside any run (plain
        ``@agent``/``pipe``/``aggregate`` calls with no durable-run
        wrapper) -- stored as SQL NULL, never a placeholder string.
        """
        conn = self._connect()
        attributes_json = json.dumps(_best_effort_jsonable(span.attributes))
        usage_json = json.dumps(to_jsonable(span.usage)) if span.usage is not None else None
        cost_usd = span.usage.cost_usd if span.usage is not None else None
        error_json = (
            json.dumps(
                {
                    "type": span.error.type,
                    "message": span.error.message,
                    "stacktrace": span.error.stacktrace,
                }
            )
            if span.error is not None
            else None
        )
        conn.execute(
            "INSERT OR REPLACE INTO spans (span_id, trace_id, run_id, parent_span_id, kind, "
            "name, started_at, ended_at, status, replayed, attributes_json, usage_json, "
            "cost_usd, error_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                span.span_id,
                span.trace_id,
                run_id,
                span.parent_span_id,
                span.kind,
                span.name,
                span.started_at,
                span.ended_at,
                span.status,
                int(span.replayed),
                attributes_json,
                usage_json,
                cost_usd,
                error_json,
            ),
        )

        input_json = (
            json.dumps(_best_effort_jsonable(span.input)) if span.input is not None else None
        )
        output_json = (
            json.dumps(_best_effort_jsonable(span.output)) if span.output is not None else None
        )
        conn.execute(
            "INSERT OR REPLACE INTO span_payloads (span_id, input_json, output_json) "
            "VALUES (?, ?, ?)",
            (span.span_id, input_json, output_json),
        )

        if self._fts_available:
            content = " ".join(text for text in (input_json, output_json) if text)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO span_fts (span_id, name, content) VALUES (?, ?, ?)",
                    (span.span_id, span.name, content),
                )
            except sqlite3.OperationalError as exc:
                # Deliberately does NOT flip `self._fts_available = False`
                # here: that flag means "FTS5 isn't compiled into this
                # sqlite3 build at all" (set once, at schema-init time, in
                # `_init_schema` -- see its own comment), a permanent,
                # store-wide condition. This except clause instead catches
                # a *transient* failure on one insert (lock contention
                # under concurrent writers, or any other momentary hiccup
                # unrelated to FTS5 availability) -- disabling search for
                # the rest of the process over a single blip would silently
                # make every later `compose runs -q` miss every span
                # persisted from here on, with no indication results are
                # now incomplete. So: warn once (loudly, but not once per
                # span for the rest of the run) and keep trying on every
                # subsequent span -- this one span just isn't searchable.
                if not self._fts_insert_warned:
                    warnings.warn(
                        f"span full-text-search insert failed ({exc}) -- this "
                        "span (and any other that hits the same transient "
                        "failure) will be missing from `compose runs -q` "
                        "results; span/run persistence itself is unaffected, "
                        "and indexing keeps being attempted for future spans "
                        "(this warning only fires once per RunStore)",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._fts_insert_warned = True

        conn.commit()

    # --- pending_interrupts (Phase 8: human-in-the-loop) -----------------

    def pending_interrupt_put(
        self,
        *,
        run_id: str,
        interrupt_id: str,
        kind: str,
        question: str | None,
        payload_json: str,
        created_at: float,
    ) -> None:
        """Upsert (``INSERT OR REPLACE``) the pending interrupt for ``(run_id, interrupt_id)``.

        Replacing rather than plain-inserting matters when a run re-pauses
        on the very same unanswered interrupt (e.g. ``resume()`` with no new
        answers) -- the row already exists and must not raise a PK conflict.
        """
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO pending_interrupts "
            "(run_id, interrupt_id, kind, question, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, interrupt_id, kind, question, payload_json, created_at),
        )
        conn.commit()

    def pending_interrupt_delete(self, run_id: str, interrupt_id: str) -> None:
        conn = self._connect()
        conn.execute(
            "DELETE FROM pending_interrupts WHERE run_id = ? AND interrupt_id = ?",
            (run_id, interrupt_id),
        )
        conn.commit()

    def pending_interrupts_all(self, run_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM pending_interrupts WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    # --- agent_state (Phase 8: agent conversation snapshots) --------------

    def agent_state_put(
        self,
        *,
        run_id: str,
        scope_key: str,
        messages_json: str,
        partial_results_json: str,
        turn: int,
        updated_at: float,
    ) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO agent_state "
            "(run_id, scope_key, messages_json, partial_results_json, turn, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, scope_key, messages_json, partial_results_json, turn, updated_at),
        )
        conn.commit()

    def agent_state_get(self, run_id: str, scope_key: str) -> dict[str, Any] | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM agent_state WHERE run_id = ? AND scope_key = ?",
            (run_id, scope_key),
        ).fetchone()
        return dict(row) if row is not None else None

    # --- atomic pause: snapshot + every pending interrupt + status, one commit --

    def persist_pause(
        self,
        *,
        run_id: str,
        interrupts: list[Any],
        scope_key: str,
        messages_json: str,
        partial_results_json: str,
        turn: int,
    ) -> None:
        """Atomically persist a tool-batch pause: every pending interrupt in
        ``interrupts`` (duck-typed: ``.id``/``.kind``/``.question``/``.payload``,
        same convention as :func:`persist_pending_interrupt`), the
        ``agent_state`` snapshot for ``scope_key``, and ``runs.status =
        'paused'`` -- all in one SQLite transaction (one ``commit()``).

        Closes a durability gap: previously, each unanswered approval-gated
        tool call's pending row was committed the moment it was found (so
        the run was already durably "paused" mid-batch), and the turn's
        ``agent_state`` snapshot was written separately, afterward. A crash
        in between left a pending interrupt with no snapshot to resume it
        against -- ``resume()`` would find no valid ``agent_state`` for the
        turn and re-run the whole thing from scratch, including tool calls
        that had already executed (and, since real side effects aren't
        idempotent by default, re-executing them). Doing every write here
        instead means a crash before this call leaves nothing durably
        paused at all (the whole turn simply re-runs, same as any other
        mid-turn crash), and a crash after it leaves a fully consistent,
        resumable pause -- there is no window in between.
        """
        conn = self._connect()
        now = time.time()
        for interrupt in interrupts:
            conn.execute(
                "INSERT OR REPLACE INTO pending_interrupts "
                "(run_id, interrupt_id, kind, question, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    interrupt.id,
                    interrupt.kind,
                    interrupt.question,
                    json.dumps(to_jsonable(interrupt.payload)),
                    now,
                ),
            )
        conn.execute(
            "INSERT OR REPLACE INTO agent_state "
            "(run_id, scope_key, messages_json, partial_results_json, turn, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, scope_key, messages_json, partial_results_json, turn, now),
        )
        conn.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
            ("paused", now, run_id),
        )
        conn.commit()

    # --- full purge: used when a run turns out to be un-resumable ----------

    def delete_run(self, run_id: str) -> None:
        """Purge every row keyed by ``run_id`` (``runs``, ``journal``, ``spans``,
        ``pending_interrupts``, ``agent_state``) in one transaction, along
        with each purged span's ``span_payloads`` row.

        Used when a run must leave nothing behind that an operator (or
        ``compose runs``) could mistake for something resumable -- see
        ``composeai.combinators._run_top``'s refusal of a paused bare
        ``pipe()``/``aggregate()`` run (durable pauses need a ``@flow`` root;
        see that function's docstring). ``span_payloads`` rows are keyed by
        ``span_id``, not ``run_id``, so they're purged via a subselect
        against ``spans`` *before* the ``spans`` rows themselves are
        deleted -- otherwise they'd be unreachable orphans left behind
        forever.
        """
        conn = self._connect()
        conn.execute(
            "DELETE FROM span_payloads WHERE span_id IN "
            "(SELECT span_id FROM spans WHERE run_id = ?)",
            (run_id,),
        )
        for table in ("runs", "journal", "spans", "pending_interrupts", "agent_state"):
            conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))  # noqa: S608
        conn.commit()


def _best_effort_jsonable(value: Any) -> Any:
    """Like ``to_jsonable``, but falls back to ``repr`` instead of raising.

    Span persistence is observability, not the durable journal: an
    unencodable payload must not stop a span (or the run it belongs to)
    from being persisted at all.
    """
    try:
        return to_jsonable(value)
    except SerializationError:
        return repr(value)


# --- default store lifecycle + span-sink installation ----------------------

_default_store: RunStore | None = None
_default_store_lock = threading.Lock()
_sink_installed = False


def _default_db_path() -> Path:
    base = os.environ.get("COMPOSE_DIR") or "./.compose"
    return Path(base) / "runs.db"


def open_default() -> RunStore:
    """Return the process-wide default :class:`RunStore`, opening it on first use.

    Rooted at ``{COMPOSE_DIR or ./.compose}/runs.db`` -- read from the
    environment lazily, at first-open time, so tests can redirect it via
    ``monkeypatch.setenv`` + :func:`reset_default`. Also ensures the
    default span-persistence sink is installed (see
    :func:`~composeai.tracing.set_span_sink`) -- the sink is installed at
    most once per process (or once per :func:`reset_default` call).
    """
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = RunStore(_default_db_path())
    _ensure_default_sink_installed()
    return _default_store


def reset_default() -> None:
    """Test hook: drop the cached default store and uninstall the span sink.

    The next :func:`open_default` call re-reads ``COMPOSE_DIR``/``./.compose``
    and reinstalls the sink from scratch. Used by the test suite's
    session-autouse fixture so every test gets an isolated store rooted at
    a temp dir instead of ever touching the repo's ``./.compose``.

    Also closes the old store's :class:`~composeai._storeasync.StoreWorker`
    (if the async engine ever routed work through it), not just the store
    itself -- otherwise its dedicated "composeai-store" writer thread is
    orphaned (never joined) until interpreter exit, one per reset, since a
    closed ``RunStore`` with a still-registered worker is never looked up
    again under its old ``id()``. Guarded for the common case where no
    worker was ever created for this store. Deferred import: same
    module-scope import-cycle reason :meth:`RunContext.ajournal_record`
    documents (``composeai._storeasync`` imports :class:`RunStore` from
    this module at module scope).

    Also clears :func:`~composeai._storeasync.reset_cast_warned_state`'s
    "already warned about a failed cast()" latch, same reasoning as
    resetting :func:`~composeai.tracing.reset_span_sink_state` below --
    both are process-wide "first failure only" latches guarding the two
    layers a span persistence failure can be caught at (a synchronous sink
    exception in ``tracing._notify_span_sink``, or the store worker
    thread's asynchronous ``cast()`` failure in ``StoreWorker._resolve``),
    and each test in the suite should get its own "first failure" chance
    rather than inheriting whichever test happened to fail first.
    """
    global _default_store, _sink_installed
    from composeai._storeasync import reset_cast_warned_state

    if _default_store is not None:
        old_store = _default_store
        old_store.close()
        from composeai._storeasync import close_worker_if_registered

        close_worker_if_registered(old_store)
    reset_cast_warned_state()
    _default_store = None
    _sink_installed = False
    tracing.set_span_sink(None)
    tracing.reset_span_sink_state()


def _ensure_default_sink_installed() -> None:
    global _sink_installed
    if _sink_installed:
        return
    tracing.set_span_sink(_default_span_sink)
    _sink_installed = True


def _default_span_sink(span: Span) -> None:
    """Persist ``span`` off the emitting thread, via the store's writer thread.

    Routes ``persist_span`` through
    ``composeai._storeasync.worker_for(store).cast(...)`` rather than calling
    it directly -- the same FIFO queue the store's journal writes go
    through, so span rows still land in the same order relative to those
    (a `cast()` enqueued here before a later journal `call()`/
    `call_blocking()` is guaranteed to run first), but the write itself no
    longer blocks whichever thread emitted the span: the runtime loop, a
    user's own asyncio loop (Plan B), or a plain synchronous call. Trade-off
    documented on :meth:`~composeai._storeasync.StoreWorker.cast`: a hard
    process crash can lose the last few spans that were cast but not yet
    drained -- observability-only, since journal durability is unaffected
    (those writes use `call`/`call_blocking`, which block until landed).

    Deferred import: same import-cycle reason documented on
    :meth:`RunContext.ajournal_record` (``composeai._storeasync`` imports
    :class:`RunStore` from this module at module scope, so this module can
    never import it back at module scope).
    """
    from composeai._storeasync import worker_for

    run_id = current_run_id()
    store = open_default()
    worker_for(store).cast("persist_span", span, run_id)


# --- ambient run_id (span-sink tagging) + RunContext (flow journaling) ----

_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_run_id", default=None
)


def current_run_id() -> str | None:
    """The ``run_id`` new spans should be tagged with when persisted, or ``None``.

    Set for the duration of a durable ``.run()`` -- a `@flow` body (via
    :class:`RunContext`) or a standalone, trace-root ``agent.run()`` (via
    :func:`run_standalone_agent`) -- so every span persisted during that
    call, not just ones a flow journals as steps, is tagged with the right
    run. ``None`` outside any durable run: such spans persist with
    ``run_id`` SQL NULL (see :meth:`RunStore.persist_span`).
    """
    return _current_run_id.get()


@contextmanager
def use_run_id(run_id: str | None) -> Iterator[None]:
    token = _current_run_id.set(run_id)
    try:
        yield
    finally:
        _current_run_id.reset(token)


# --- scope stack: deterministic journal keys under concurrent dispatch -----
#
# A contextvar stack of scope segments, e.g. ("map#2[3]",) or
# ("aggregate#1/researcher",) -- joined with "/" and prepended to every
# journal key reserved while active. This is what makes journal keys
# deterministic for *every* stage kind (bare callables, @agent functions,
# Pipeline/Aggregate, not just @task) under compose.map/aggregate/parallel
# tool dispatch: instead of racing every concurrently-scheduled item/
# branch's calls to `next_key()` against each other (whichever one happens
# to run first wins that slot -- see the durability audit that added this),
# the dispatching combinator reserves one deterministic scope segment *per
# item/branch/call*, serially, in a fixed (input/declaration) order, on its
# own engine coroutine, *before* any item/branch is actually dispatched.
# Each item/branch's own coroutine then pushes its already-reserved segment
# for the duration of its call (via `push_scope`) -- so whatever journal
# keys that call's own body reserves are automatically prefixed by a scope
# that was assigned deterministically, regardless of which item/branch
# actually finishes first.
#
# Propagates the way `asyncio.Task` already does: each item/branch runs as
# its own coroutine (a native `asyncio.Task` for an async-native stage, or
# handed off to `_dispatch._run_sync_on_own_thread`'s own dedicated thread
# for a plain sync callable -- see that module), and `push_scope` is called
# *inside* that coroutine, so setting it there can never leak back to the
# dispatching coroutine or to sibling items/branches. A sync callable's
# dedicated thread sees the same ambient scope via that bridge's own
# `contextvars.copy_context()` snapshot, taken at the moment the coroutine
# reaches it -- no separate `tracing.propagate()` step needed here (that
# helper is for genuine standalone worker threads, e.g. `.stream()`'s
# background thread, not this coroutine-native dispatch).
_scope_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "_scope_stack", default=()
)


@contextmanager
def push_scope(segment: str) -> Iterator[None]:
    """Push ``segment`` onto the ambient scope stack for the duration of the block.

    Safe to call with no active :class:`RunContext` (e.g. outside a
    ``@flow``) -- it only affects keys reserved via
    :meth:`RunContext.next_key`/:meth:`RunContext.reserve_scope_segment`,
    so it's a harmless no-op when nothing reads the stack.
    """
    stack = _scope_stack.get()
    token = _scope_stack.set((*stack, segment))
    try:
        yield
    finally:
        _scope_stack.reset(token)


@runtime_checkable
class JournalScope(Protocol):
    """What :func:`current_run_context` returns: either the real
    :class:`RunContext` or a guard proxy wrapping one (see
    :class:`_AbandonGuardedRunContext`, installed only inside an abandoned
    ``@task(timeout=...)`` worker thread's own copied context).
    """

    @property
    def run_id(self) -> str: ...
    @property
    def preloaded(self) -> dict[str, str]: ...

    def next_key(self, name: str) -> str: ...
    def reserve_scope_segment(self, name: str) -> str: ...
    def qualify(self, segment: str) -> str: ...
    def journal_lookup(self, key: str) -> tuple[bool, Any]: ...
    def journal_record(self, key: str, value: Any) -> Any: ...
    async def ajournal_record(self, key: str, value: Any) -> Any: ...


@dataclass
class RunContext:
    """Ambient, per-flow-run state: the journal store, run id, and step counters.

    ``preloaded`` is the run's full journal (``{step_key: value_json}``)
    read from the store once at flow start -- an in-memory fast path for
    replay lookups so every journaled step doesn't need its own SELECT.
    ``counters`` is a per-*scoped*-step-name call counter (keyed by
    ``(*active_scope, name)``, not just ``name`` -- see the scope-stack
    module docstring above), incremented under ``_lock`` so concurrent
    callers sharing the same scope (there normally aren't any: each
    concurrently-dispatched item/branch/call gets its own scope segment)
    never corrupt it.
    """

    run_id: str
    store: RunStore
    counters: dict[tuple[str, ...], int] = field(default_factory=dict)
    preloaded: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def reserve_scope_segment(self, name: str) -> str:
        """Reserve the next call-order slot for ``name`` under the active scope.

        Returns the *local* segment only (e.g. ``"map#2"``), not prefixed by
        the active scope -- combinators use this (rather than
        :meth:`next_key`) when they need the raw segment to push as a new,
        deeper scope for concurrently-dispatched work (see
        ``composeai.combinators.map``/``Aggregate._run_branches`` and
        ``composeai.flow``'s nested-``@flow``-call journaling).
        """
        with self._lock:
            scope = _scope_stack.get()
            counter_key = (*scope, name)
            n = self.counters.get(counter_key, 0) + 1
            self.counters[counter_key] = n
        return f"{name}#{n}"

    def qualify(self, segment: str) -> str:
        """Prefix ``segment`` with the active scope stack, joined by ``"/"``."""
        scope = _scope_stack.get()
        return "/".join((*scope, segment)) if scope else segment

    def next_key(self, name: str) -> str:
        """Reserve and return the next fully-qualified journal key for ``name``.

        Equivalent to ``qualify(reserve_scope_segment(name))`` -- the one
        call most journaling call sites (``@task``, the ``@agent`` loop's
        journaled-run wrapper, nested ``@flow`` calls) need when they're not
        also opening a new scope for concurrent sub-work.
        """
        return self.qualify(self.reserve_scope_segment(name))

    def journal_lookup(self, key: str) -> tuple[bool, Any]:
        """``(True, decoded_value)`` on a journal hit, else ``(False, None)``."""
        value_json = self.preloaded.get(key)
        if value_json is None:
            return False, None
        return True, from_jsonable(json.loads(value_json))

    def journal_record(self, key: str, value: Any) -> Any:
        """Journal ``value`` under ``key`` (first-write-wins) and return the winning, decoded value.

        Raises :class:`~composeai.errors.SerializationError` naming
        ``key`` if ``value`` can't round-trip the journal encoder.
        """
        try:
            value_json = json.dumps(to_jsonable(value))
        except SerializationError as exc:
            raise SerializationError(f"step {key!r}: {exc}") from exc
        winner_json = self.store.journal_put(self.run_id, key, value_json)
        self.preloaded[key] = winner_json
        return from_jsonable(json.loads(winner_json))

    async def ajournal_record(self, key: str, value: Any) -> Any:
        """Async twin of :meth:`journal_record` (v0.4.0 Plan A, Task 8).

        The one blocking call ``journal_record`` makes -- ``self.store
        .journal_put(...)`` -- routes through
        ``composeai._storeasync.worker_for(self.store).call("journal_put", ...)``
        (the store's dedicated writer thread) instead of calling
        ``self.store`` directly, so a journaled ``@agent`` step reached from
        inside :mod:`composeai.combinators`'s async engine (``map``/
        ``aggregate``/``pipe`` fan-outs) never blocks the composeai runtime
        loop with SQLite I/O -- same reasoning as
        :func:`~composeai.agentfn._asnapshot_agent_state`.

        First-write-wins is preserved exactly: :meth:`RunStore.journal_put`'s
        ``INSERT OR IGNORE`` (falling back to a re-``SELECT`` of the row
        that won, on conflict) is still the one atomic operation that
        decides a race's winner -- this only changes WHICH THREAD executes
        that same SQL (the store's writer thread instead of whichever
        thread called ``ajournal_record``), never how the race itself is
        resolved. The return value is the winner's decoded value either
        way (never necessarily this caller's own ``value``), and
        ``self.preloaded`` is updated with the winner's JSON exactly like
        the sync path, so a subsequent ``journal_lookup`` (sync or async,
        on any thread sharing this ``RunContext``) sees it immediately.

        Deferred import: :mod:`composeai._storeasync` imports
        :class:`RunStore` from this module at module scope, so importing it
        back here at module scope would cycle -- this local import is the
        same workaround :mod:`composeai.agentfn` didn't need only because
        it is never imported *by* ``_storeasync``.
        """
        from composeai._storeasync import worker_for

        try:
            value_json = json.dumps(to_jsonable(value))
        except SerializationError as exc:
            raise SerializationError(f"step {key!r}: {exc}") from exc
        winner_json = await worker_for(self.store).call(
            "journal_put", self.run_id, key, value_json
        )
        self.preloaded[key] = winner_json
        return from_jsonable(json.loads(winner_json))


class _AbandonedTaskError(BaseException):
    """Internal control-flow signal: raised inside an abandoned
    ``@task(timeout=...)`` worker thread the instant it touches the journal
    after its timeout already fired (see :class:`_AbandonGuardedRunContext`).

    Never meant to be caught by user code, or even to be observed --
    ``_run_with_timeout``'s worker's own top-level ``except
    BaseException`` swallows it (the caller thread already has its
    ``TaskTimeoutError``; there's nothing further to report). A
    ``BaseException``, not an ``Exception``, specifically so a task's own
    retry loop (``except Exception``) or a tool body's error handling can't
    accidentally catch it and keep the zombie mutating the journal.
    """


@dataclass
class _AbandonGuardedRunContext:
    """Wraps a real :class:`RunContext` so a timed-out ``@task``'s abandoned
    worker thread loses journal write access the instant it's abandoned.

    Installed by ``_run_with_timeout`` *only* inside the
    worker thread's own copied context (via :func:`use_journal_scope`) --
    the caller thread (and its own copy of the ambient context, taken by
    ``tracing.propagate`` before the worker ever starts) keeps referencing
    the real, unwrapped :class:`RunContext` throughout, so an abandoned
    worker can never affect the flow that gave up on it, and the flow's own
    subsequent steps are never guarded.

    Every journal-touching call checks ``abandoned`` first; once it's set
    (by the caller, the instant the timeout fires), any further call raises
    :class:`_AbandonedTaskError` instead of touching the store. This only
    stops the zombie from journaling further steps (or reserving further
    scopes) -- it cannot stop the zombie's *pure* side effects (a network
    call, a file write, ...) already in flight, since there is no safe way
    to interrupt an arbitrary Python thread. That part is inherent, and
    already documented on :class:`~composeai.errors.TaskTimeoutError`.
    """

    _ctx: JournalScope
    _abandoned: threading.Event

    @property
    def run_id(self) -> str:
        return self._ctx.run_id

    @property
    def preloaded(self) -> dict[str, str]:
        return self._ctx.preloaded

    def _check(self) -> None:
        if self._abandoned.is_set():
            raise _AbandonedTaskError(
                "task abandoned by timeout -- journal write access revoked"
            )

    def next_key(self, name: str) -> str:
        self._check()
        return self._ctx.next_key(name)

    def reserve_scope_segment(self, name: str) -> str:
        self._check()
        return self._ctx.reserve_scope_segment(name)

    def qualify(self, segment: str) -> str:
        # A pure read of the ambient scope stack -- nothing to guard.
        return self._ctx.qualify(segment)

    def journal_lookup(self, key: str) -> tuple[bool, Any]:
        self._check()
        return self._ctx.journal_lookup(key)

    def journal_record(self, key: str, value: Any) -> Any:
        self._check()
        return self._ctx.journal_record(key, value)

    async def ajournal_record(self, key: str, value: Any) -> Any:
        # Mostly unreachable: this guard is only ever installed inside an
        # abandoned @task(timeout=...)'s own daemon THREAD copy of the
        # ambient context (see _run_with_timeout), not the composeai runtime
        # loop's own async engine -- but a nested async agent call launched
        # from that abandoned worker's thread CAN reach this await, so
        # ``self._check()`` below is the real defense, not the surrounding
        # structure. Implemented anyway for structural parity with every
        # other method on this class (all of which guard-check before
        # delegating).
        self._check()
        return await self._ctx.ajournal_record(key, value)


def abandon_guard(ctx: JournalScope | None, abandoned: threading.Event) -> JournalScope | None:
    """Wrap ``ctx`` in a guard that revokes journal access once ``abandoned`` is set.

    ``None`` in, ``None`` out -- nothing to guard when there's no active
    :class:`RunContext` (the task isn't inside a ``@flow``).
    """
    if ctx is None:
        return None
    return _AbandonGuardedRunContext(ctx, abandoned)


def interrupt_lookup(key: str) -> tuple[bool, Any]:
    """``(True, decoded_value)`` on a journal hit for the ambient run, else ``(False, None)``.

    The one lookup :mod:`composeai.hitl`'s ``approve``/``ask_human`` (and
    :mod:`composeai.agentfn`'s approval-gated-tool check) both use: inside an
    active ``@flow`` body, delegates to the ambient :class:`RunContext` (its
    preloaded-journal fast path); inside a durable ``@agent`` run (standalone
    or nested -- no ``RunContext`` of its own, but :func:`current_run_id`
    is set either way), reads the run's journal row directly. Raises
    :class:`~composeai.errors.ConfigError` if neither is active -- there is
    nowhere to journal an answer against.
    """
    ctx = current_run_context()
    if ctx is not None:
        return ctx.journal_lookup(key)
    run_id = current_run_id()
    if run_id is None:
        raise ConfigError(
            "approve()/ask_human() (and approval-gated @tool calls) only work "
            "inside an active @flow body or a durable @agent run -- none is "
            "active here"
        )
    store = open_default()
    raw = store.journal_get(run_id, key)
    if raw is None:
        return False, None
    return True, from_jsonable(json.loads(raw))


def persist_pending_interrupt(store: RunStore, run_id: str, interrupt: Any) -> None:
    """Persist ``interrupt`` (duck-typed: ``.id``/``.kind``/``.question``/``.payload``)
    as ``run_id``'s pending interrupt and mark its run row ``"paused"``.

    Takes a duck-typed object rather than a real
    :class:`~composeai.hitl.Interrupt` -- this module cannot import
    ``composeai.hitl``, which imports *this* module for :func:`interrupt_lookup`
    and :func:`current_run_context`/:func:`current_run_id` above (importing it
    back here would cycle). Used by this module's own :func:`settle_agent_run`
    for a standalone (non-flow) run's pause path; a flow's own pause goes
    through the async twin below, :func:`apersist_pending_interrupt`, from
    inside ``composeai.flow``'s already-async execution instead.

    Both store calls route through
    ``composeai._storeasync.worker_for(store).call_blocking(...)`` rather
    than calling ``store`` directly (v0.4.0 Plan B Task 1) -- the same FIFO
    queue and worker thread that ``runs._default_span_sink`` casts this
    run's span persistence onto, so by the time this (blocking) call
    returns, every span cast before it is guaranteed to have already run
    (persisted, or failed-and-warned -- see
    ``_storeasync._warn_once_on_cast_failure``), never left queued behind a
    ``.run()`` call that's already returned control to the caller. Deferred
    import: same import-cycle reason documented on
    :meth:`RunContext.ajournal_record` (``composeai._storeasync`` imports
    :class:`RunStore` from this module at module scope).
    """
    from composeai._storeasync import worker_for

    payload_json = json.dumps(to_jsonable(interrupt.payload))
    worker = worker_for(store)
    worker.call_blocking(
        "pending_interrupt_put",
        run_id=run_id,
        interrupt_id=interrupt.id,
        kind=interrupt.kind,
        question=interrupt.question,
        payload_json=payload_json,
        created_at=time.time(),
    )
    worker.call_blocking("update_run", run_id, status="paused")


async def apersist_pending_interrupt(store: RunStore, run_id: str, interrupt: Any) -> None:
    """Async twin of :func:`persist_pending_interrupt` (v0.4.0 Plan A, Task 9).

    Both underlying store calls (``pending_interrupt_put`` then
    ``update_run``) route through
    ``composeai._storeasync.worker_for(store).call(...)`` instead of calling
    ``store`` directly, one after the other in the same order the sync
    version uses -- so persisting a flow's pause from inside
    ``composeai.flow._aexecute_flow`` (running on the composeai runtime
    loop) never blocks that loop with SQLite I/O. Same reasoning as
    :meth:`RunContext.ajournal_record`; deferred import for the same
    import-cycle reason documented there (:mod:`composeai._storeasync`
    imports :class:`RunStore` from this module at module scope).
    """
    from composeai._storeasync import worker_for

    payload_json = json.dumps(to_jsonable(interrupt.payload))
    await worker_for(store).call(
        "pending_interrupt_put",
        run_id=run_id,
        interrupt_id=interrupt.id,
        kind=interrupt.kind,
        question=interrupt.question,
        payload_json=payload_json,
        created_at=time.time(),
    )
    await worker_for(store).call("update_run", run_id, status="paused")


# --- resume() answers: shared by composeai.flow.resume and --------------
# composeai.agentfn.resume_standalone_agent (moved here, from composeai.flow,
# so both can call it without a flow<->agentfn import cycle -- agentfn already
# imports this module).


def resolve_answer_key(pending_ids: set[str], raw_key: str) -> str:
    """Resolve one ``answers`` dict key to a full interrupt id.

    Exact matches (against currently-pending interrupt ids) pass through
    unchanged, as does any key that doesn't look like it was meant as
    ``tool:`` shorthand (see below) -- this is what lets a bare, caller-
    chosen ``approve("some_id")``/``ask_human("some_id", ...)`` id be
    supplied *before* the flow has even reached that call (a documented,
    tested pattern: "answer several future gates in one ``resume()``
    call"), since such an id is looked up verbatim later and doesn't need
    to be currently pending to be meaningful.

    A ``"tool_name"`` shorthand (see :mod:`composeai.agentfn`'s
    ``tool:{tool_name}:{call_id}`` convention) resolves when exactly one
    pending interrupt matches that tool name; matching more than one raises
    :class:`~composeai.errors.ConfigError` listing the full ids. Unlike a
    bare id, a tool-call answer can *never* be meaningfully pre-supplied
    ahead of time (the real id embeds a model-assigned ``call_id`` that
    doesn't exist yet) -- so when every currently-pending interrupt is
    itself a ``tool:`` id and none of them match ``raw_key`` either
    exactly or by tool-name prefix, ``raw_key`` can only be a stale/
    mistyped/wrong-tool shorthand that will never resolve to anything;
    this raises :class:`~composeai.errors.ConfigError` listing the pending
    ids rather than silently journaling a dead answer under
    ``__interrupt__:{raw_key}`` (previously: silently accepted, then
    never found once the real ``tool:{name}:{call_id}`` interrupt paused).
    """
    if raw_key in pending_ids:
        return raw_key
    matches = sorted(i for i in pending_ids if i.startswith(f"tool:{raw_key}:"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ConfigError(
            f"answer key {raw_key!r} is ambiguous -- it matches multiple pending "
            f"interrupts {matches!r}; use the full interrupt id to disambiguate"
        )
    if pending_ids and all(i.startswith("tool:") for i in pending_ids):
        raise ConfigError(
            f"answer key {raw_key!r} does not match any pending interrupt "
            f"-- currently pending: {sorted(pending_ids)!r} (a tool-call answer "
            "can't be supplied ahead of the tool actually being called, unlike a "
            "bare approve()/ask_human() id)"
        )
    return raw_key


def apply_resume_answers(store: RunStore, run_id: str, answers: dict[str, Any] | None) -> None:
    """Journal each ``answers`` entry under ``__interrupt__:{full_id}`` and clear its pending row.

    First-write-wins (``journal_put``'s ``INSERT OR IGNORE``), same as every
    other journaled value -- an answer can never be silently overwritten by
    a second, different ``resume()`` call for the same interrupt.

    Callers must only call this once resuming is definitely going ahead
    (run exists, is registered, fingerprint matches, isn't already
    completed, ...) -- journaling an answer is a durable, first-write-wins
    commit; doing it before those checks would permanently lock in an
    answer even for a ``resume()`` call that ends up raising and never
    actually re-executing anything.
    """
    if not answers:
        return
    pending_ids = {row["interrupt_id"] for row in store.pending_interrupts_all(run_id)}
    for raw_key, value in answers.items():
        full_id = resolve_answer_key(pending_ids, raw_key)
        try:
            value_json = json.dumps(to_jsonable(value))
        except SerializationError as exc:
            raise SerializationError(f"answer {raw_key!r}: {exc}") from exc
        store.journal_put(run_id, f"__interrupt__:{full_id}", value_json)
        store.pending_interrupt_delete(run_id, full_id)


async def aapply_resume_answers(
    store: RunStore, run_id: str, answers: dict[str, Any] | None
) -> None:
    """Async twin of :func:`apply_resume_answers` (v0.4.0 Plan B, Task 7).

    Identical first-write-wins/:func:`resolve_answer_key` logic -- read
    :func:`apply_resume_answers`'s docstring first, unchanged here. Every
    store call it makes (``pending_interrupts_all``, then one
    ``journal_put``/``pending_interrupt_delete`` pair per answer, same
    order as the sync version) routes through ``await
    worker_for(store).call(...)`` instead of calling ``store`` directly, so
    a durable resume driven through ``composeai.flow.aresume``/
    ``composeai.agentfn.aresume_standalone_agent`` (both running on the
    CALLER's own event loop, never a dedicated worker thread) never blocks
    that loop with SQLite I/O. Deferred import: same import-cycle reason
    documented on :meth:`RunContext.ajournal_record`
    (``composeai._storeasync`` imports :class:`RunStore` from this module
    at module scope).
    """
    if not answers:
        return
    from composeai._storeasync import worker_for

    pending_rows = await worker_for(store).call("pending_interrupts_all", run_id)
    pending_ids = {row["interrupt_id"] for row in pending_rows}
    for raw_key, value in answers.items():
        full_id = resolve_answer_key(pending_ids, raw_key)
        try:
            value_json = json.dumps(to_jsonable(value))
        except SerializationError as exc:
            raise SerializationError(f"answer {raw_key!r}: {exc}") from exc
        await worker_for(store).call(
            "journal_put", run_id, f"__interrupt__:{full_id}", value_json
        )
        await worker_for(store).call("pending_interrupt_delete", run_id, full_id)


_run_context: contextvars.ContextVar[JournalScope | None] = contextvars.ContextVar(
    "_run_context", default=None
)


def current_run_context() -> JournalScope | None:
    """The ambient :class:`JournalScope` if we're inside a ``@flow`` body, else ``None``.

    Consulted by ``@task``/``@agent`` (and ``compose.map``/``aggregate``) to
    decide whether a call should auto-journal as a step or just execute
    plainly. Normally the real :class:`RunContext`; inside an abandoned
    ``@task(timeout=...)`` worker thread, a guard proxy that revokes journal
    access (see :func:`abandon_guard`) -- callers only need the
    :class:`JournalScope` interface either way.
    """
    return _run_context.get()


@contextmanager
def use_run_context(ctx: RunContext) -> Iterator[RunContext]:
    """Make ``ctx`` ambient (for :func:`current_run_context`) and tag spans with its ``run_id``."""
    ctx_token = _run_context.set(ctx)
    id_token = _current_run_id.set(ctx.run_id)
    try:
        yield ctx
    finally:
        _run_context.reset(ctx_token)
        _current_run_id.reset(id_token)


@contextmanager
def use_journal_scope(scope: JournalScope | None) -> Iterator[None]:
    """Make ``scope`` the ambient :class:`JournalScope` for the block, without
    touching :func:`current_run_id` (unlike :func:`use_run_context`, which
    also tags spans with the run id).

    Used to install a per-thread journal-access guard: see
    ``_run_with_timeout``, which calls this (inside the
    timed-out task's own worker thread, on its own copied context -- see
    ``composeai.tracing.propagate``) to swap in an
    :func:`abandon_guard`-wrapped context that only that thread (and
    anything it itself spawns) ever sees.
    """
    token = _run_context.set(scope)
    try:
        yield
    finally:
        _run_context.reset(token)


def _error_payload(exc: BaseException) -> dict[str, str]:
    """Build the ``error_json`` payload for a failed run row.

    ``str(exc)`` is empty for a plain ``asyncio.CancelledError()`` (the
    common shape when an in-flight ``arun()``/``astream()`` is cancelled --
    e.g. via ``Task.cancel()`` or Ctrl-C), which would otherwise persist an
    empty message with nothing for ``compose runs``/``compose trace`` to
    render; synthesize ``"cancelled"`` for that case specifically. Every
    other exception's message passes through unchanged. Shared by
    :func:`settle_agent_run`, :func:`asettle_agent_run`, and
    ``composeai.flow._aexecute_flow``'s failed path -- the three sites that
    persist a run row's terminal failure.
    """
    message = str(exc) or ("cancelled" if isinstance(exc, asyncio.CancelledError) else "")
    return {"type": type(exc).__name__, "message": message}


def run_standalone_agent(
    name: str,
    args_json: str | None,
    thunk: Callable[[], Run[Any]],
    *,
    budget: Budget | None = None,
) -> Run[Any]:
    """Wrap a root-level ``agent.run()`` call with a durable ``runs`` row (kind ``"agent"``).

    Called by :mod:`composeai.agentfn` only when the agent call is a trace
    root (no enclosing span) -- never for an agent nested inside a
    pipe/aggregate/flow (not a trace root) nor for one auto-journaled
    inside a flow (that goes through :class:`RunContext` instead, as one
    step of the flow's *own* run row, not a row of its own). ``args_json``
    (the encoded call args/kwargs, or ``None``) lets a later ``resume()``
    reconstruct the original call if it's ever needed before any
    ``agent_state`` snapshot exists (see :func:`settle_agent_run``).
    ``budget`` is persisted on the row (see :func:`encode_budget`) so
    ``composeai.agentfn.resume_standalone_agent`` can reinstate the same cap
    on resume -- previously a resume ran with no budget enforcement at all.

    Pre-generates ``trace_id`` (mirroring ``composeai.flow.Flow.run``)
    and wraps ``thunk()`` in :func:`~composeai.tracing.use_trace` with it --
    this is what lets a pause be turned into a paused ``Run`` *with* its
    trace still attached (see :func:`settle_agent_run`), and what lets a
    later resume share the same trace_id.
    """
    store = open_default()
    run_id = new_ulid()
    trace_id = new_ulid()
    now = time.time()
    store.create_run(
        run_id=run_id,
        kind="agent",
        name=name,
        status="running",
        created_at=now,
        updated_at=now,
        trace_id=trace_id,
        fingerprint=None,
        args_json=args_json,
        budget_json=encode_budget(budget),
    )
    with use_run_id(run_id), tracing.use_trace(trace_id):
        return settle_agent_run(store, run_id, thunk)


def settle_agent_run(store: RunStore, run_id: str, thunk: Callable[[], Run[Any]]) -> Run[Any]:
    """Run ``thunk`` to completion or pause, persisting the outcome to ``run_id``'s row.

    Shared by a fresh standalone agent run (:func:`run_standalone_agent`) and
    a resumed one (``composeai.agentfn.resume_standalone_agent``) -- both
    wrap ``thunk`` in the same ambient ``use_run_id``/
    ``tracing.use_trace(...)`` context before calling this, so
    :func:`~composeai.tracing.current_trace` here is always the run's own
    trace (needed to build the returned ``Run`` either way).

    Detects a pause via ``getattr(exc, "_compose_pause", False)`` (duck-typed
    -- this module cannot import ``composeai.hitl``, which imports this
    module) rather than catching a specific exception type; a real failure
    marks the row ``"failed"`` and re-raises unchanged.

    Every finalization path -- completed, failed, and (via
    :func:`persist_pending_interrupt`) paused -- routes its ``update_run``
    through ``composeai._storeasync.worker_for(store).call_blocking(...)``
    rather than calling ``store`` directly (v0.4.0 Plan B Task 1 fix: this
    used to bypass the store's worker/queue entirely, so a standalone
    agent/pipe/aggregate run's ``.run()`` could return -- row marked
    ``"completed"``/``"failed"``/``"paused"`` and all -- while some of that
    same run's spans were still sitting in the worker's queue, cast but not
    yet actually persisted).
    ``call_blocking`` shares the same FIFO queue as the ``cast()`` jobs
    ``runs._default_span_sink`` enqueues for every span this run emitted,
    so its blocking wait doubles as a barrier: by the time this call
    returns (and thus by the time ``.run()`` returns to the caller), every
    span this run cast is guaranteed to have already been persisted or
    failed-and-warned (see ``_storeasync._warn_once_on_cast_failure``),
    never left queued behind a run that's already visibly done. A flow's
    own run already got this for free -- ``composeai.flow`` awaits its
    worker calls directly -- this closes the same gap for a run that never
    goes through a flow at all.
    """
    from composeai._storeasync import worker_for

    try:
        run = thunk()
    except BaseException as exc:
        if getattr(exc, "_compose_pause", False):
            # `exc.interrupt` (duck-typed -- see the module docstring's note
            # on why this module can't import `composeai.hitl.Interrupt` to
            # type-check the attribute access statically).
            interrupt = getattr(exc, "interrupt")  # noqa: B009
            persist_pending_interrupt(store, run_id, interrupt)
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
        worker_for(store).call_blocking(
            "update_run",
            run_id,
            status="failed",
            error_json=json.dumps(_error_payload(exc)),
        )
        raise
    worker_for(store).call_blocking(
        "update_run",
        run_id,
        status="completed",
        trace_id=run.trace.trace_id,
        output_json=_best_effort_output_json(run.output),
    )
    # The durable row's id IS the run's identity: `run.id` must match what
    # `compose runs` / `compose trace <id>` will find in the store.
    run.id = run_id
    return run


async def asettle_agent_run(
    store: RunStore, run_id: str, thunk: Callable[[], Coroutine[Any, Any, Run[Any]]]
) -> Run[Any]:
    """Async twin of :func:`settle_agent_run` (v0.4.0 Plan B, Task 4).

    Same status transitions/pause handling as the sync version -- read its
    docstring first -- but every store write goes through ``await
    composeai._storeasync.worker_for(store).call(...)`` rather than
    ``call_blocking(...)``: this runs on the CALLER's own asyncio loop (via
    ``AgentFunction.arun``/``.astream``), never a dedicated worker thread,
    so a *blocking* call here would freeze that loop instead of merely
    parking an idle thread the way ``call_blocking`` does on the sync path.

    ``thunk`` is a zero-arg coroutine *factory*
    (``Callable[[], Coroutine[Any, Any, Run]]``), not a coroutine itself --
    calling it produces the coroutine this function awaits, exactly once
    (mirrors the sync version calling its own zero-arg ``thunk``).

    The pause path reuses :func:`apersist_pending_interrupt` (the existing
    async twin of :func:`persist_pending_interrupt`, v0.4.0 Plan A) rather
    than duplicating its two awaited store calls here.
    """
    from composeai._storeasync import worker_for

    try:
        run = await thunk()
    except BaseException as exc:
        if getattr(exc, "_compose_pause", False):
            # `exc.interrupt` (duck-typed -- see the module docstring's note
            # on why this module can't import `composeai.hitl.Interrupt` to
            # type-check the attribute access statically).
            interrupt = getattr(exc, "interrupt")  # noqa: B009
            await apersist_pending_interrupt(store, run_id, interrupt)
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
        await worker_for(store).call(
            "update_run",
            run_id,
            status="failed",
            error_json=json.dumps(_error_payload(exc)),
        )
        raise
    await worker_for(store).call(
        "update_run",
        run_id,
        status="completed",
        trace_id=run.trace.trace_id,
        output_json=_best_effort_output_json(run.output),
    )
    # The durable row's id IS the run's identity: `run.id` must match what
    # `compose runs` / `compose trace <id>` will find in the store.
    run.id = run_id
    return run


async def arun_standalone_agent(
    name: str,
    args_json: str | None,
    thunk: Callable[[], Coroutine[Any, Any, Run[Any]]],
    *,
    budget: Budget | None = None,
) -> Run[Any]:
    """Async twin of :func:`run_standalone_agent` (v0.4.0 Plan B, Task 4).

    Same shape as the sync version -- read its docstring first -- pre-
    generating ``run_id``/``trace_id``, creating the durable row, wrapping
    ``thunk`` in :func:`use_run_id`/:func:`~composeai.tracing.use_trace`
    (both plain synchronous context managers: the contextvars they push
    stay correctly scoped across every ``await`` inside this same
    coroutine/task, so there is nothing async-specific needed there), then
    handing off to :func:`asettle_agent_run`.

    The one real difference from the sync version: row creation itself
    goes through ``await composeai._storeasync.worker_for(store)
    .call("create_run", ...)`` instead of calling ``store.create_run``
    directly -- this runs on the caller's own asyncio loop (never a
    dedicated worker thread), so a direct, blocking SQLite write here would
    freeze that loop instead of merely parking an idle thread.
    """
    from composeai._storeasync import worker_for

    store = open_default()
    run_id = new_ulid()
    trace_id = new_ulid()
    now = time.time()
    await worker_for(store).call(
        "create_run",
        run_id=run_id,
        kind="agent",
        name=name,
        status="running",
        created_at=now,
        updated_at=now,
        trace_id=trace_id,
        fingerprint=None,
        args_json=args_json,
        budget_json=encode_budget(budget),
    )
    with use_run_id(run_id), tracing.use_trace(trace_id):
        return await asettle_agent_run(store, run_id, thunk)


def _best_effort_output_json(value: Any) -> str | None:
    try:
        return json.dumps(to_jsonable(value))
    except SerializationError:
        return None
