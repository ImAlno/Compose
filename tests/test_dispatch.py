"""The async engine's stage dispatcher (composeai._dispatch)."""

from __future__ import annotations

import asyncio
import time

import pytest

from composeai._dispatch import gather_settled, run_stage
from composeai.errors import TaskTimeoutError


def test_sync_fn_runs_via_to_thread():
    def work(x: int) -> int:
        return x + 1

    assert asyncio.run(run_stage(work, (1,), {}, timeout=None, name="w", kind="@task")) == 2


def test_async_fn_awaited_natively():
    async def work(x: int) -> int:
        await asyncio.sleep(0)
        return x * 3

    assert asyncio.run(run_stage(work, (2,), {}, timeout=None, name="w", kind="@task")) == 6


def test_async_fn_timeout_cancels_cooperatively():
    started = time.monotonic()

    async def slow() -> str:
        await asyncio.sleep(10)
        return "never"

    with pytest.raises(TaskTimeoutError, match="cancelled cooperatively"):
        asyncio.run(run_stage(slow, (), {}, timeout=0.2, name="slow", kind="@task"))
    assert time.monotonic() - started < 2


def test_sync_fn_timeout_keeps_daemon_race_semantics():
    started = time.monotonic()

    def stuck() -> str:
        time.sleep(10)
        return "never"

    with pytest.raises(TaskTimeoutError, match="abandoned thread"):
        asyncio.run(run_stage(stuck, (), {}, timeout=0.2, name="stuck", kind="@task"))
    assert time.monotonic() - started < 2


def test_contextvars_reach_sync_stage():
    import contextvars

    var: contextvars.ContextVar[str] = contextvars.ContextVar("dvar", default="unset")

    def read() -> str:
        return var.get()

    async def drive() -> str:
        var.set("from-async")
        return await run_stage(read, (), {}, timeout=None, name="r", kind="@task")

    assert asyncio.run(drive()) == "from-async"


def test_cancelled_stage_zombie_loses_journal_access(tmp_path):
    """Cancel the awaiting task while a sync stage runs; the abandoned
    stage's later journal write must be rejected by the abandon guard.

    Closes Plan A's remaining gap: `wait_for`-cancelling a composite stage
    used to leave an in-flight nested sync stage running on its own thread
    WITH journal access -- a zombie write could land under the cancelled
    branch's keys. Cancelling the awaiting side must set the same abandon
    guard `_run_with_timeout` already installs for timeouts.
    """
    import threading

    from composeai import runs
    from composeai._dispatch import run_stage

    store = runs.RunStore(tmp_path / "runs.db")
    started = threading.Event()
    release = threading.Event()
    zombie_result: dict = {}

    def slow_writer() -> str:
        started.set()
        release.wait(timeout=10)  # keep running past the cancel
        ctx = runs.current_run_context()
        assert ctx is not None
        try:
            ctx.journal_record("zombie#1", "should-be-blocked")
            zombie_result["wrote"] = True
        except BaseException as exc:  # noqa: BLE001 -- the abandon guard raises
            # a BaseException (never an Exception) by design, precisely so a
            # zombie stage's own error handling can't swallow it.
            zombie_result["error"] = type(exc).__name__
        return "done"

    async def drive() -> None:
        ctx = runs.RunContext(run_id="r-zombie", store=store)
        with runs.use_run_context(ctx):
            task = asyncio.ensure_future(
                run_stage(slow_writer, (), {}, timeout=None, name="w", kind="@task")
            )
            await asyncio.get_event_loop().run_in_executor(None, started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(drive())
    release.set()

    deadline = time.monotonic() + 5
    while "wrote" not in zombie_result and "error" not in zombie_result:
        if time.monotonic() > deadline:
            break
        time.sleep(0.01)

    assert zombie_result == {"error": "_AbandonedTaskError"}


def test_gather_settled_every_item_settles_in_order():
    async def ok(i):
        return i

    async def bad(i):
        raise ValueError(f"bad-{i}")

    async def drive():
        return await gather_settled([ok(0), bad(1), ok(2)])

    settled = asyncio.run(drive())
    assert settled[0] == (0, None)
    assert settled[1][0] is None and isinstance(settled[1][1], ValueError)
    assert settled[2] == (2, None)
