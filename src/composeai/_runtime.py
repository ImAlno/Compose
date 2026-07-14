"""The process-wide sync<->async runtime bridge.

One lazily-started daemon thread runs a persistent asyncio event loop
("composeai-runtime"). Every SYNC public entry point (agent_fn.run(),
flow.run(), resume(), map(), ...) drives the async-native engine by
submitting its coroutine here and blocking on the result. Because the
engine runs on this dedicated thread's loop -- never the caller's -- the
sync API keeps working even when the calling thread already has a running
event loop (Jupyter, ASGI handlers), the failure mode that bites
run_until_complete-style facades (see the 0.4.0 design spec's research
notes). A persistent loop (rather than asyncio.run per call) keeps
loop-affine state (async SDK clients, sessions) valid across calls.

Contextvars: the caller's context is snapshotted into the submitted
coroutine (contextvars.copy_context), matching the snapshot-at-spawn
semantics tracing.propagate() established for worker threads.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextvars
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

_lock = threading.Lock()
_runtime: _Runtime | None = None


class _Runtime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="composeai-runtime", daemon=True
        )
        self.thread.start()
        self.stopped = False
        atexit.register(self._shutdown)

    def _shutdown(self) -> None:
        # Interpreter exit: stop the loop and give the thread a bounded
        # chance to unwind. Never hang exit -- the thread is a daemon.
        #
        # `stopped` is set BEFORE we touch the loop so run_sync can fail
        # fast instead of scheduling work that will never run. This matters
        # for atexit-LIFO ordering: atexit hooks fire last-registered-first,
        # so any hook registered *before* ours (i.e. before this runtime was
        # lazily created) fires *after* our `_shutdown`. If that hook calls
        # a sync composeai API, it must see the RuntimeError below rather
        # than block forever on a loop/thread that has already stopped.
        self.stopped = True
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2)
        except Exception:
            pass


def _get_runtime() -> _Runtime:
    global _runtime
    if _runtime is None:
        with _lock:
            if _runtime is None:
                _runtime = _Runtime()
    return _runtime


def get_loop() -> asyncio.AbstractEventLoop:
    """The runtime loop (started lazily on first use)."""
    return _get_runtime().loop


def on_runtime_thread() -> bool:
    runtime = _runtime
    return runtime is not None and threading.current_thread() is runtime.thread


def _cancel_task_threadsafe(
    loop: asyncio.AbstractEventLoop, task_holder: list[asyncio.Task[Any]]
) -> None:
    """Cancel the task in ``task_holder`` (if any), threadsafe from any thread.

    ``task_holder`` is the one-slot list ``run_sync``'s ``_spawn`` appends
    the live task handle to, right after ``loop.create_task`` succeeds --
    read here (from whatever thread ``run_sync``'s ``KeyboardInterrupt``
    except-branch runs on) without a lock, since a list append/read is
    atomic under the GIL and this function only ever reads the slot, never
    mutates it. Empty when this races ahead of ``_spawn`` actually running
    on the loop thread -- a ``KeyboardInterrupt`` can land at any point on
    the calling thread, including before ``call_soon_threadsafe``'s
    callback has even been scheduled -- in which case this is a no-op:
    best-effort cancellation, not a guarantee. ``task.cancel()`` itself is
    scheduled via ``call_soon_threadsafe`` rather than called directly,
    since ``asyncio.Task.cancel`` is only safe to call from the loop's own
    thread.
    """
    if not task_holder:
        return
    task = task_holder[0]
    try:
        loop.call_soon_threadsafe(task.cancel)
    except RuntimeError:
        # The loop closed between `run_sync` reading `runtime.stopped`/
        # `loop.is_closed()` and the KeyboardInterrupt landing here (e.g.
        # interpreter shutdown racing a Ctrl-C) -- `call_soon_threadsafe`
        # raises RuntimeError on a closed loop. Cancellation here is
        # documented best-effort, not a guarantee (see this function's own
        # docstring); a closed loop can't run the cancel callback anyway,
        # so there is nothing left to do but let the original
        # KeyboardInterrupt still propagate to the caller unchanged.
        pass


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` on the runtime loop, blocking the calling thread.

    The one facade primitive. Snapshots the caller's contextvars into the
    coroutine. Refuses to run from the runtime thread itself: blocking
    there would deadlock the loop that must execute the work.

    KeyboardInterrupt semantics: ``done.result()`` blocks on a plain
    ``concurrent.futures.Future``, so Ctrl-C raises ``KeyboardInterrupt`` out
    of *this* wait on the calling thread. That exception is caught here
    (v0.4.0 Plan B, Task 2) and used to schedule ``task.cancel()``
    threadsafe on the runtime loop -- via :func:`_cancel_task_threadsafe`,
    reading the live ``asyncio.Task`` handle out of the one-slot
    ``task_holder`` list ``_spawn`` populates below -- before re-raising the
    original ``KeyboardInterrupt`` unchanged. So Ctrl-C now actually stops
    the engine attempt: the task is cancelled cooperatively (a real
    ``CancelledError`` thrown into ``coro``'s await chain, same as a
    ``wait_for`` timeout), and any nested own-thread sync stage still
    running loses journal write access via the same abandon-guard
    ``_dispatch._run_sync_on_own_thread`` installs for exactly this case.
    There is a race: a ``KeyboardInterrupt`` landing before ``_spawn`` has
    even run on the loop thread yet finds ``task_holder`` empty --
    cancellation is then a best-effort no-op (nothing to cancel yet), but
    the original ``KeyboardInterrupt`` is still re-raised to the caller
    regardless, exactly as before. A pure side effect already in flight
    (network call, file write, ...) still can't be safely interrupted --
    that part is inherent, unchanged by this.
    """
    if on_runtime_thread():
        coro.close()  # avoid the un-awaited coroutine warning
        raise RuntimeError(
            "composeai runtime cannot be re-entered from its own loop thread"
        )
    runtime = _get_runtime()
    loop = runtime.loop
    # Benign TOCTOU: `stopped` could flip true between this read and
    # `call_soon_threadsafe(_spawn)` below. Accepted today because only
    # atexit's `_shutdown` ever sets it (one-way, interpreter-exit-only) --
    # a future shutdown path triggered concurrently with live `run_sync`
    # callers (not just atexit) would need to close this race for real.
    if runtime.stopped or loop.is_closed():
        # Fail fast on a dead runtime instead of scheduling work that will
        # never run: after _shutdown, nothing will ever call
        # `done.set_result`/`set_exception`, so `done.result()` below would
        # block the caller forever. This is the atexit-LIFO scenario: an
        # atexit hook registered before this runtime was created fires
        # *after* our `_shutdown` (atexit is LIFO) and may call a sync
        # composeai API during interpreter exit -- it must get this error,
        # not a silent hang.
        coro.close()  # avoid the un-awaited coroutine warning
        raise RuntimeError(
            "composeai runtime is shut down (interpreter exit in progress?)"
        )
    context = contextvars.copy_context()
    done: concurrent.futures.Future[T] = concurrent.futures.Future()
    # One-slot holder for the live Task handle, populated by `_spawn` below
    # once `loop.create_task` succeeds -- read by the KeyboardInterrupt
    # except-branch (on the calling thread) via `_cancel_task_threadsafe`.
    # A list, not a plain variable: `_spawn` runs on the loop thread while
    # this is read from the calling thread, and a single append/index read
    # is atomic under the GIL without needing a lock.
    task_holder: list[asyncio.Task[T]] = []

    def _spawn() -> None:
        # The entire body is guarded: this callback runs on the runtime
        # loop via call_soon_threadsafe, so any exception it raises would
        # otherwise be swallowed by the loop's default exception handler
        # (just logged) while the caller sits blocked on done.result()
        # forever. Route every failure into `done` instead.
        try:
            # context.run(loop.create_task, coro) makes the Task copy the
            # CALLER's context snapshot -- run_coroutine_threadsafe alone
            # would create the Task in the RUNTIME thread's context
            # instead, silently dropping the caller's contextvars.
            task = context.run(loop.create_task, coro)
            task_holder.append(task)

            def _finish(task: asyncio.Task[T]) -> None:
                if task.cancelled():
                    done.set_exception(concurrent.futures.CancelledError())
                    return
                exception = task.exception()
                if exception is not None:
                    done.set_exception(exception)
                else:
                    done.set_result(task.result())

            task.add_done_callback(_finish)
        except BaseException as exc:
            done.set_exception(exc)

    loop.call_soon_threadsafe(_spawn)
    try:
        return done.result()
    except KeyboardInterrupt:
        _cancel_task_threadsafe(loop, task_holder)
        raise
