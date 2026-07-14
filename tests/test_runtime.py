"""The process-wide runtime loop bridge (composeai._runtime)."""

from __future__ import annotations

import asyncio

import pytest

from composeai._runtime import get_loop, on_runtime_thread, run_sync


async def _double(x: int) -> int:
    await asyncio.sleep(0)
    return x * 2


def test_run_sync_returns_result():
    assert run_sync(_double(21)) == 42


def test_run_sync_propagates_exception():
    async def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_sync(boom())


def test_run_sync_works_with_running_loop_in_caller_thread():
    """The Jupyter/ASGI case: run_sync is called from inside a coroutine that
    is itself running on a loop in the CALLING thread's call stack.

    A naive facade (asyncio.run / loop.run_until_complete) would raise
    RuntimeError("... already running") here, since you can't nest a second
    run_until_complete inside a loop that's already driving the current
    frame. This implementation works anyway because the actual coroutine
    executes on the *runtime* thread's own loop, not the caller's -- run_sync
    just blocks the caller's thread (parking its loop) on a
    concurrent.futures.Future until that result comes back.
    """
    async def caller_side():
        # THIS coroutine is running on a loop in the current thread's call
        # stack right now. run_sync must still work.
        return run_sync(_double(5))

    assert asyncio.run(caller_side()) == 10


def test_loop_is_persistent_and_singleton():
    loop_a = get_loop()
    loop_b = get_loop()
    assert loop_a is loop_b
    assert run_sync(_double(1)) == 2  # still usable after prior tests


def test_reentry_from_runtime_thread_raises():
    async def tries_reentry():
        # on the runtime thread now; calling run_sync here must fail fast
        assert on_runtime_thread()
        with pytest.raises(RuntimeError, match="re-entered"):
            run_sync(_double(1))
        return "ok"

    assert run_sync(tries_reentry()) == "ok"


def test_contextvars_flow_into_runtime():
    import contextvars

    var: contextvars.ContextVar[str] = contextvars.ContextVar("t1var", default="unset")
    var.set("caller-value")

    async def read_var() -> str:
        return var.get()

    assert run_sync(read_var()) == "caller-value"


def test_keyboard_interrupt_cancels_engine_task():
    """Simulate the caller thread's blocking wait being interrupted: the
    engine-side task must get cancelled, not keep running.

    Drives the cancel SEAM directly (`_runtime._cancel_task_threadsafe`) --
    the same helper `run_sync`'s KeyboardInterrupt except-branch calls --
    rather than real signal delivery, mirroring exactly what `_spawn`/
    `run_sync` do internally (schedule task creation via
    `call_soon_threadsafe`, hold the task handle in a one-slot list, cancel
    threadsafe from another thread).
    """
    import threading
    import time as _time

    from composeai import _runtime

    started = threading.Event()
    state: dict = {}

    async def engine():
        started.set()
        try:
            await asyncio.sleep(30)
            state["finished"] = True
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    loop = _runtime.get_loop()
    task_holder: list = []

    def _spawn() -> None:
        task_holder.append(loop.create_task(engine()))

    loop.call_soon_threadsafe(_spawn)
    assert started.wait(timeout=2)

    _runtime._cancel_task_threadsafe(loop, task_holder)

    deadline = _time.monotonic() + 2
    while state.get("cancelled") is not True and _time.monotonic() < deadline:
        _time.sleep(0.01)

    assert state == {"cancelled": True}


def test_run_sync_after_shutdown_raises_not_hangs():
    from composeai import _runtime

    runtime = _runtime._get_runtime()
    # simulate interpreter-exit ordering without killing the loop for later
    # tests: flip the flag, assert fail-fast, flip back
    runtime.stopped = True
    try:
        with pytest.raises(RuntimeError, match="shut down"):
            _runtime.run_sync(_double(1))
    finally:
        runtime.stopped = False
