"""The dedicated SQLite writer thread.

Each :class:`RunStore` gets exactly one background thread ("composeai-store")
that owns every call into it, FIFO, off a ``queue.Queue`` of jobs. This is
deliberately NOT ``asyncio.to_thread``: that helper submits to
``concurrent.futures``' shared *default* executor, which is joined at
interpreter exit (``atexit``) with no timeout. One SQLite call wedged behind
the 30s busy-timeout would pile up on that shared executor and reintroduce
exactly the shutdown-hang class ``runs._run_with_timeout`` was designed to
avoid. A single owned thread sidesteps that: it can be closed on its own
schedule (bounded join, idempotent), and -- because ``RunStore`` keeps its
connection in a ``threading.local`` -- pinning all calls to one thread also
guarantees exactly one SQLite connection and serializes writes exactly like
today's single-caller behavior.

Two ways to submit a job, sharing the same queue and worker thread:

- ``call()`` is async: it requires a running loop (``asyncio.get_running_loop``),
  builds an ``asyncio.Future`` on it, and the worker resolves that future via
  ``loop.call_soon_threadsafe`` (the only safe way to touch a loop from another
  thread).
- ``call_blocking()`` is sync: it hands the worker a ``concurrent.futures.Future``
  and blocks the calling thread on ``.result()``. For facade code that is
  already off the event loop.

Every job runs inside a try/except on the worker thread, and the reply --
whichever kind it is -- is resolved either with the result or with the
exception, never left untouched. Combined with rejecting new jobs once
``close()`` has run, no caller can be left blocked on a future that nothing
will ever resolve.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import queue
import threading
from typing import Any

from .runs import RunStore

_STOP = object()

# A job's reply is either half of a `call()` pair -- the asyncio.Future to
# resolve plus the loop it belongs to, so the worker thread can hop back via
# call_soon_threadsafe -- or a plain concurrent.futures.Future for
# `call_blocking()`.


def _set_result(future: asyncio.Future[Any], result: Any) -> None:
    if not future.done():
        future.set_result(result)


def _set_exception(future: asyncio.Future[Any], exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


class StoreWorker:
    """One dedicated writer thread wrapping a single :class:`RunStore`."""

    def __init__(self, store: RunStore) -> None:
        self._store = store
        self._queue: queue.Queue[Any] = queue.Queue()
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="composeai-store", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is _STOP:
                return
            method_name, args, kwargs, reply = job
            # Guarded so a broken store method (or a bug here) can never
            # leave the awaiting side blocked forever -- the reply is
            # always resolved, one way or the other, and the worker loop
            # itself never dies mid-job.
            try:
                result = getattr(self._store, method_name)(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 -- propagated to caller, not swallowed
                self._resolve(reply, exc=exc)
            else:
                self._resolve(reply, result=result)

    @staticmethod
    def _resolve(reply: Any, *, result: Any = None, exc: BaseException | None = None) -> None:
        if isinstance(reply, tuple):
            future, loop = reply
            callback = _set_exception if exc is not None else _set_result
            value = exc if exc is not None else result
            try:
                loop.call_soon_threadsafe(callback, future, value)
            except RuntimeError:
                # Loop already closed underneath us -- nothing left to
                # resolve; nobody can still be awaiting it.
                pass
        else:
            cf_future: concurrent.futures.Future[Any] = reply
            if not cf_future.done():
                if exc is not None:
                    cf_future.set_exception(exc)
                else:
                    cf_future.set_result(result)

    def _enqueue(
        self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any], reply: Any
    ) -> None:
        # Check-and-put must be atomic with close()'s flip-and-put-STOP,
        # under the same lock -- otherwise a job enqueued between the read
        # of `_closed` and the `put` could land *after* `_STOP`, where the
        # worker thread will never reach it: it exits at `_STOP` without
        # draining what comes after, leaving the caller's future (or
        # `.result()`) waiting forever. Serializing both operations on
        # `_close_lock` means an enqueue either lands before `_STOP` (and
        # gets drained) or is rejected outright -- never silently stranded.
        # This lock is uncontended in the common case, so it's cheap.
        with self._close_lock:
            if self._closed:
                raise RuntimeError("StoreWorker is closed")
            self._queue.put((method_name, args, kwargs, reply))

    async def call(self, method_name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Enqueue ``store.<method_name>(*args, **kwargs)``; await its result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._enqueue(method_name, args, kwargs, (future, loop))
        return await future

    def call_blocking(self, method_name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Same job queue as :meth:`call`, but block the caller synchronously."""
        cf_future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._enqueue(method_name, args, kwargs, cf_future)
        return cf_future.result()

    def close(self) -> None:
        """Drain the queue, stop the thread, join (<=2s). Idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STOP)
        self._thread.join(timeout=2)
        _forget_worker(self)


_workers: dict[int, StoreWorker] = {}
_workers_lock = threading.Lock()


def _forget_worker(worker: StoreWorker) -> None:
    """Remove ``worker`` from the ``worker_for`` registry, if it's there.

    Called from :meth:`StoreWorker.close`, so a closed worker is never
    handed back out: the next ``worker_for(store)`` for the same store
    mints a fresh one instead of returning a dead worker whose thread has
    already exited.
    """
    with _workers_lock:
        for key, registered in list(_workers.items()):
            if registered is worker:
                del _workers[key]


def close_worker_if_registered(store: RunStore) -> None:
    """Close and forget ``store``'s :class:`StoreWorker`, if one was ever created.

    Unlike :func:`worker_for`, this never creates a worker just to close it
    -- a ``store`` nothing ever routed async work through has no entry here,
    and this is a no-op for it. Used by
    :func:`composeai.runs.reset_default` so resetting the process-wide
    default store doesn't orphan its "composeai-store" writer thread until
    interpreter exit (the test suite's autouse fixture calls
    ``reset_default`` once per test).
    """
    key = id(store)
    with _workers_lock:
        worker = _workers.get(key)
    if worker is not None:
        worker.close()


def worker_for(store: RunStore) -> StoreWorker:
    """The one live :class:`StoreWorker` for this ``store`` instance.

    Keyed by ``id(store)`` in a plain dict rather than a
    ``WeakKeyDictionary``. A ``StoreWorker`` holds a strong reference to
    its ``store``, so a registered store is never garbage collected while
    its worker is registered -- the dict does not rely on GC to stay
    bounded. It stays bounded in practice because the only store
    registered here today is the process-wide singleton from
    ``runs.open_default()``, and because ``close()`` removes its own entry
    (see :func:`_forget_worker`), so a closed worker's slot is freed for a
    fresh one rather than accumulating dead entries. Any worker still
    registered is swept and closed at interpreter exit.
    """
    key = id(store)
    with _workers_lock:
        worker = _workers.get(key)
        if worker is None:
            worker = StoreWorker(store)
            _workers[key] = worker
        return worker


@atexit.register
def _close_all_workers() -> None:
    with _workers_lock:
        workers = list(_workers.values())
    for worker in workers:
        worker.close()
