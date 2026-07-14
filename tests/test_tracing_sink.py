"""Tests for Phase 7's tracing additions: ``set_span_sink`` and ``use_trace``.

``set_span_sink`` is a module-global hook (not a contextvar -- persistence
must observe every span regardless of which thread/context created it)
called with every finished span, from ``span()``'s ``finally`` block.
Sink failures must never crash user code: caught here in ``tracing.py``
(which must not import ``composeai.runs`` -- the dependency runs the other
way, ``runs`` installs the sink into ``tracing``), with one process-wide
``RuntimeWarning`` on the first failure and silence after that.

``use_trace`` lets flow resume force a specific ``trace_id`` as ambient so
resumed spans join the original run's trace instead of starting a new one.
"""

import warnings

import pytest

from composeai import tracing
from composeai.tracing import current_trace, reset_span_sink_state, set_span_sink, span, use_trace


@pytest.fixture(autouse=True)
def _reset_sink():
    yield
    set_span_sink(None)
    reset_span_sink_state()


def test_no_sink_by_default_is_a_noop():
    with span("task", "t"):
        pass  # must not raise with no sink installed


def test_set_span_sink_called_with_finished_span():
    received = []
    set_span_sink(received.append)
    with span("task", "t") as s:
        pass
    assert len(received) == 1
    assert received[0] is s
    assert received[0].status == "ok"
    assert received[0].ended_at is not None


def test_set_span_sink_called_for_every_span_including_nested():
    received = []
    set_span_sink(received.append)
    with span("flow", "root"):
        with span("task", "child"):
            pass
    assert len(received) == 2
    assert received[0].name == "child"
    assert received[1].name == "root"


def test_set_span_sink_none_uninstalls():
    received = []
    set_span_sink(received.append)
    set_span_sink(None)
    with span("task", "t"):
        pass
    assert received == []


def test_sink_failure_warns_once_and_does_not_crash_run():
    calls = {"n": 0}

    def bad_sink(finished_span):
        calls["n"] += 1
        raise RuntimeError("boom")

    set_span_sink(bad_sink)

    with pytest.warns(RuntimeWarning):
        with span("task", "t1"):
            pass
    assert calls["n"] == 1

    # Second failure: no warning (already warned once this process), but
    # still doesn't crash and the sink is still invoked.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with span("task", "t2"):
            pass
    assert calls["n"] == 2


def test_use_trace_forces_specific_trace_id_for_new_spans():
    with use_trace("forced-trace-id"):
        with span("flow", "root") as root:
            assert root.trace_id == "forced-trace-id"
        trace = current_trace()
        assert trace is not None
        assert trace.trace_id == "forced-trace-id"
    assert current_trace() is None


def test_use_trace_restores_previous_trace_after_block():
    with span("flow", "outer") as outer:
        with use_trace("other-trace"):
            with span("task", "inner") as inner:
                assert inner.trace_id == "other-trace"
        restored = current_trace()
        assert restored is not None
        assert restored.trace_id == outer.trace_id


def test_span_sink_persists_via_worker_and_is_drained_by_close():
    """Spans now persist through the store worker (off the emitting thread).

    Uses the REAL default sink installed by ``composeai.runs`` (not this
    file's custom ``received.append`` sinks the other tests install) --
    ``runs.reset_default()`` first so it installs fresh regardless of
    whatever ran before this test in the same session. After running a
    traced call and closing the worker (drain), the span row must be
    present -- same net effect as the old synchronous sink, just off the
    emitting thread.
    """
    import sqlite3

    from composeai import runs
    from composeai._storeasync import worker_for

    runs.reset_default()
    try:
        store = runs.open_default()  # installs runs._default_span_sink
        with span("task", "sink-worker-t1") as s:
            pass
        worker_for(store).close()  # drain: blocks until the cast persist lands
        conn = sqlite3.connect(store._path)
        row = conn.execute(
            "SELECT span_id FROM spans WHERE span_id = ?", (s.span_id,)
        ).fetchone()
        assert row is not None
    finally:
        runs.reset_default()


def test_module_does_not_import_runs():
    # tracing.py must stay ignorant of composeai.runs -- the sink is
    # injected, not imported. A regression here would reintroduce the
    # circular-import risk the design docs explicitly warn against.
    assert "runs" not in vars(tracing) or not hasattr(tracing, "RunStore")
    import ast
    import inspect

    source = inspect.getsource(tracing)
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
    assert not any(name.endswith("runs") or name == "runs" for name in imported_names)
