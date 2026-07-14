"""The dedicated SQLite writer thread (composeai._storeasync)."""

from __future__ import annotations

import time

import pytest

from composeai import _runtime
from composeai._storeasync import StoreWorker, worker_for
from composeai.runs import open_default


def test_call_executes_store_method_and_returns_result():
    store = open_default()
    worker = StoreWorker(store)
    try:
        run_id = "storeasync-t1"
        _runtime.run_sync(
            worker.call(
                "create_run",
                run_id=run_id,
                kind="flow",
                name="w",
                status="running",
                created_at=time.time(),
                updated_at=time.time(),
                trace_id="t",
                fingerprint=None,
                args_json=None,
            )
        )
        row = _runtime.run_sync(worker.call("get_run", run_id))
        assert row is not None and row["run_id"] == run_id
    finally:
        worker.close()


def test_exceptions_propagate():
    store = open_default()
    worker = StoreWorker(store)
    try:
        with pytest.raises(Exception):  # noqa: B017 -- store-side ValueError, any exception is fine
            _runtime.run_sync(worker.call("update_run", "nope", not_a_column="x"))
    finally:
        worker.close()


def test_call_blocking_same_queue():
    store = open_default()
    worker = StoreWorker(store)
    try:
        assert worker.call_blocking("get_run", "definitely-missing") is None
    finally:
        worker.close()


def test_close_idempotent_and_call_after_close_raises():
    store = open_default()
    worker = StoreWorker(store)
    worker.close()
    worker.close()
    with pytest.raises(RuntimeError, match="closed"):
        worker.call_blocking("get_run", "x")


def test_worker_for_returns_one_worker_per_store():
    store = open_default()
    worker = worker_for(store)
    try:
        assert worker_for(store) is worker
    finally:
        worker.close()
    # close() removes itself from the worker_for registry, so a later
    # lookup for the same store mints a fresh (live) worker rather than
    # handing back the one whose thread has already exited.
    fresh = worker_for(store)
    try:
        assert fresh is not worker
    finally:
        fresh.close()


def test_close_drains_already_enqueued_jobs():
    import threading

    store = open_default()
    worker = StoreWorker(store)
    results = []

    def _fetch() -> None:
        results.append(worker.call_blocking("get_run", "missing-drain"))

    threads = [threading.Thread(target=_fetch) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    worker.close()
    assert results == [None] * 5
