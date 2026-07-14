"""The async engine's dual stage dispatcher.

``run_stage`` is the ONE way the async engine executes a unit of user code
(an ``@agent``/``@task``/``@tool`` body, ...), whether that code is a
coroutine function or a plain sync callable:

- A coroutine function is awaited natively on the caller's own running loop.
  A timeout wraps the await in ``asyncio.wait_for`` -- NOT ``asyncio.timeout``,
  which needs Python 3.11 and this codebase's floor is 3.10 -- so on timeout
  the coroutine is cancelled cooperatively (a real ``CancelledError`` thrown
  into it) rather than abandoned; there is no zombie thread for the async
  path.
- A sync callable runs off-loop on its own dedicated daemon thread (see
  ``_run_sync_on_own_thread`` below) so it never blocks the event loop.
  This is deliberately NOT ``asyncio.to_thread``: that helper hands the
  call to the loop's shared, bounded default executor
  (``min(32, cpu+4)`` threads), and a sync stage that itself calls a sync
  facade (``compose.map``, ``agent.run``, ...) blocks its executor slot
  while the nested work needs more slots of the *same* pool -- at
  fan-out >= pool size that is a permanent, silent deadlock. A dedicated
  thread per in-flight sync stage has no shared bound, so nesting can
  never starve it. Contextvars are propagated manually (``to_thread`` did
  this for us) so contextvars set by the caller are still visible inside
  ``fn``. WITH a timeout, a thread can't be safely interrupted, so this
  delegates to ``runs._run_with_timeout`` -- unchanged daemon-race-and-abandon
  semantics, just invoked from this dedicated thread instead of directly,
  preserving the exact "abandoned thread" wording callers already match on.
  Independent of any timeout, ``_run_sync_on_own_thread`` itself always
  installs the same journal abandon-guard around ``fn`` (v0.4.0 Plan B,
  Task 2): if the AWAITING side is cancelled instead (e.g. an enclosing
  ``wait_for`` timing out a *composite* stage while this nested sync stage
  is still running on its own thread), the guard's ``abandoned`` event is
  set the instant that happens, so the zombie thread's later journal writes
  are rejected rather than landing under the cancelled branch's keys.

``gather_settled`` runs a batch of coroutines concurrently and guarantees
every one of them settles (result or exception, never left pending), in
input order -- the shape combinators (map/aggregate/...) need to collect
partial results after one branch fails without losing the others.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
from collections.abc import Callable, Coroutine
from typing import Any

from . import runs
from .errors import TaskTimeoutError


def _resolve(future: asyncio.Future, result: Any, exc: BaseException | None) -> None:
    """Settle ``future`` with ``result`` or ``exc``, called on the loop thread.

    Guards against a future the awaiting task already gave up on (cancelled)
    so a late worker-thread completion never raises ``InvalidStateError``.
    """
    if future.cancelled():
        return
    if exc is not None:
        future.set_exception(exc)
    else:
        future.set_result(result)


async def _run_sync_on_own_thread(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Run a sync callable on its OWN daemon thread, awaiting the result.

    NOT ``asyncio.to_thread``: its shared, bounded default executor deadlocks
    when sync stages nest sync facades (each blocked caller pins a slot;
    fan-out >= pool size starves the nested work forever). A dedicated
    thread per in-flight sync stage restores the old engine's cost model
    (fresh pool per fan-out) where total threads scale with concurrent
    work and no shared bound exists. Contextvars are propagated manually
    (``to_thread`` did it for us); the reply future is resolved under a
    ``try``/``except BaseException`` so the awaiting task can never hang.

    Installs the same journal abandon-guard ``runs._run_with_timeout``
    installs for a timed-out task (v0.4.0 Plan B, Task 2 -- closing Plan
    A's remaining gap): if the AWAITING side (this coroutine, on the
    engine's loop) is cancelled -- e.g. an enclosing ``wait_for`` timing out
    a *composite* stage while this nested sync stage is still running --
    ``abandoned`` is set the instant that happens, and the worker thread's
    own copied context has a guard installed in its place so any further
    journal touch from inside ``fn`` raises instead of landing a zombie
    write under the cancelled branch's keys. Only the worker thread's own
    copied context is ever guarded -- never the caller's -- so a cancelled
    stage never affects sibling branches or the flow that gave up on it.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    context = contextvars.copy_context()
    abandoned = threading.Event()

    def _guarded_call() -> Any:
        ctx = runs.current_run_context()
        guarded = runs.abandon_guard(ctx, abandoned)
        with runs.use_journal_scope(guarded):
            return fn(*args, **kwargs)

    def _worker() -> None:
        try:
            result = context.run(_guarded_call)
        except BaseException as exc:  # resolved below; never swallowed
            try:
                loop.call_soon_threadsafe(_resolve, future, None, exc)
            except RuntimeError:
                pass  # loop already closed -- nothing to do, awaiter is gone
        else:
            try:
                loop.call_soon_threadsafe(_resolve, future, result, None)
            except RuntimeError:
                pass  # loop already closed -- nothing to do, awaiter is gone

    threading.Thread(target=_worker, daemon=True, name="composeai-stage").start()
    try:
        return await future
    except asyncio.CancelledError:
        abandoned.set()
        raise


async def run_stage(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    timeout: float | None,
    name: str,
    kind: str,
) -> Any:
    """Run ``fn(*args, **kwargs)`` as one dispatched stage of the async engine.

    ``fn`` may be a coroutine function or a plain sync callable; see the
    module docstring for the four resulting code paths.
    """
    if inspect.iscoroutinefunction(fn):
        coro: Coroutine[Any, Any, Any] = fn(*args, **kwargs)
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TaskTimeoutError(
                f"{kind} {name!r} exceeded timeout={timeout}s (cancelled cooperatively)"
            ) from exc

    if timeout is None:
        return await _run_sync_on_own_thread(fn, args, kwargs)
    return await _run_sync_on_own_thread(
        runs._run_with_timeout, (fn, args, kwargs, timeout, name), {"kind": kind}
    )


async def gather_settled(
    coros: list[Coroutine[Any, Any, Any]],
) -> list[tuple[Any, BaseException | None]]:
    """Run ``coros`` concurrently; every item settles, in input order.

    Reshapes ``asyncio.gather(..., return_exceptions=True)`` into this
    codebase's ``(output, exc)`` settled-pair convention.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    return [
        (None, result) if isinstance(result, BaseException) else (result, None)
        for result in results
    ]
