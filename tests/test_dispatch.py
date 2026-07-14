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
