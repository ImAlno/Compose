"""Tests for ``composeai.runs.RunStore`` (Phase 7: durable flow runtime).

All tests use an explicit ``RunStore(path)`` pointed at a ``tmp_path`` --
never the real ``./.compose`` -- except the handful that specifically
exercise ``open_default()``/``reset_default()`` and ``COMPOSE_DIR``, which
still redirect via ``monkeypatch.setenv`` + ``reset_default()`` so nothing
ever touches the repo (the session ``conftest.py`` fixture does this
globally too, but these tests are explicit about it since they're testing
that exact mechanism).
"""

import json
import sqlite3
import threading
import warnings
from typing import Any

import pytest

from composeai import runs, tracing
from composeai._encoding import from_jsonable
from composeai.messages import Usage
from composeai.runs import RunStore


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore(tmp_path / "sub" / "runs.db")


# --- construction / WAL --------------------------------------------------


def test_creates_db_file_and_parent_dir_lazily(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "runs.db"
    assert not db_path.parent.exists()
    RunStore(db_path)
    assert db_path.exists()
    assert db_path.parent.exists()


def test_wal_mode_active(store):
    conn = sqlite3.connect(store._path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode.lower() == "wal"


def test_schema_init_is_idempotent(tmp_path):
    path = tmp_path / "runs.db"
    RunStore(path)
    RunStore(path)  # must not raise re-opening an existing db


# --- runs table CRUD -------------------------------------------------------


def _run_kwargs(**overrides):
    base = dict(
        run_id="r1",
        kind="flow",
        name="myflow",
        status="running",
        created_at=1.0,
        updated_at=1.0,
        trace_id="t1",
        fingerprint="abc123",
        args_json="{}",
        output_json=None,
        error_json=None,
    )
    base.update(overrides)
    return base


def test_create_and_get_run_round_trip(store):
    store.create_run(**_run_kwargs())
    row = store.get_run("r1")
    assert row is not None
    assert row["run_id"] == "r1"
    assert row["kind"] == "flow"
    assert row["name"] == "myflow"
    assert row["status"] == "running"
    assert row["trace_id"] == "t1"
    assert row["fingerprint"] == "abc123"


def test_get_run_missing_returns_none(store):
    assert store.get_run("nope") is None


def test_update_run_changes_fields(store):
    store.create_run(**_run_kwargs())
    store.update_run("r1", status="completed", output_json='{"x": 1}')
    row = store.get_run("r1")
    assert row["status"] == "completed"
    assert row["output_json"] == '{"x": 1}'


def test_update_run_rejects_unknown_column_name(store):
    """Regression: `update_run`'s SET clause is built from `fields`' *keys*,
    which can't be parameterized -- an allow-list is what stops a key like
    `"status = (SELECT ...), error_json"` from being interpolated as SQL
    instead of a column name. Verified experimentally in the original
    finding that this exact shape rewrote the generated UPDATE statement
    into a cross-row data exfiltration primitive; this asserts the
    allow-list now rejects it outright, before any SQL is built."""
    from composeai.errors import ConfigError

    store.create_run(**_run_kwargs())
    malicious_key = "status = (SELECT group_concat(run_id) FROM runs), error_json"
    with pytest.raises(ConfigError):
        store.update_run("r1", **{malicious_key: "x"})
    # Nothing was mutated -- the row is untouched.
    assert store.get_run("r1")["status"] == "running"


def test_update_run_rejects_plain_unknown_column_too(store):
    from composeai.errors import ConfigError

    store.create_run(**_run_kwargs())
    with pytest.raises(ConfigError):
        store.update_run("r1", not_a_real_column="x")


def test_list_runs_orders_newest_first(store):
    store.create_run(**_run_kwargs(run_id="r1", created_at=1.0))
    store.create_run(**_run_kwargs(run_id="r2", created_at=2.0))
    store.create_run(**_run_kwargs(run_id="r3", created_at=3.0))
    rows = store.list_runs()
    assert [r["run_id"] for r in rows] == ["r3", "r2", "r1"]


def test_list_runs_filters_by_kind_and_status(store):
    store.create_run(**_run_kwargs(run_id="r1", kind="flow", status="completed", created_at=1.0))
    store.create_run(**_run_kwargs(run_id="r2", kind="agent", status="completed", created_at=2.0))
    store.create_run(**_run_kwargs(run_id="r3", kind="flow", status="failed", created_at=3.0))

    assert [r["run_id"] for r in store.list_runs(kind="flow")] == ["r3", "r1"]
    assert [r["run_id"] for r in store.list_runs(status="completed")] == ["r2", "r1"]
    assert [r["run_id"] for r in store.list_runs(kind="flow", status="failed")] == ["r3"]


# --- journal: get/put/all, first-write-wins --------------------------------


def test_journal_get_missing_returns_none(store):
    assert store.journal_get("r1", "task#1") is None


def test_journal_put_then_get_round_trip(store):
    winner = store.journal_put("r1", "task#1", '{"v": 1}')
    assert winner == '{"v": 1}'
    assert store.journal_get("r1", "task#1") == '{"v": 1}'


def test_journal_all_returns_every_key_for_run(store):
    store.journal_put("r1", "task#1", "1")
    store.journal_put("r1", "task#2", "2")
    store.journal_put("r2", "task#1", "99")  # different run, must not leak in
    assert store.journal_all("r1") == {"task#1": "1", "task#2": "2"}


def test_journal_put_first_write_wins_same_store(store):
    winner1 = store.journal_put("r1", "k", "first")
    winner2 = store.journal_put("r1", "k", "second")
    assert winner1 == "first"
    assert winner2 == "first"  # loser reads back the winner, not its own value
    assert store.journal_get("r1", "k") == "first"


def test_journal_put_first_write_wins_across_two_store_handles(tmp_path):
    """Two ``RunStore`` handles on the same file, racing on the same key.

    Simulates two threads/processes writing the same journal key
    concurrently (e.g. two branches of a ``compose.map`` somehow computing
    the same step, or a crash-and-resume race) -- exactly one value must
    survive, and the loser must read back *that* value, never its own.
    """
    path = tmp_path / "runs.db"
    store_a = RunStore(path)
    store_b = RunStore(path)

    results: list[str] = [""] * 2
    barrier = threading.Barrier(2)

    def write(idx: int, store_handle: RunStore, value: str) -> None:
        barrier.wait()
        results[idx] = store_handle.journal_put("shared-run", "k", value)

    t1 = threading.Thread(target=write, args=(0, store_a, "value-a"))
    t2 = threading.Thread(target=write, args=(1, store_b, "value-b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results[0] == results[1]  # both threads agree on one winner
    assert results[0] in ("value-a", "value-b")
    assert store_a.journal_get("shared-run", "k") == results[0]
    assert store_b.journal_get("shared-run", "k") == results[0]


# --- span persistence: hot/cold split + FTS --------------------------------


def _make_span(**overrides: Any) -> tracing.Span:
    base: dict[str, Any] = dict(
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        kind="task",
        name="do-thing",
        started_at=1.0,
        ended_at=2.0,
        status="ok",
        input={"a": 1},
        output={"b": 2},
        usage=None,
        attributes={"x": "y"},
        replayed=False,
    )
    base.update(overrides)
    return tracing.Span(**base)


def test_persist_span_writes_hot_row(store):
    span = _make_span()
    store.persist_span(span, "r1")
    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM spans WHERE span_id = ?", ("s1",)).fetchone()
    conn.close()
    assert row is not None
    assert row["run_id"] == "r1"
    assert row["trace_id"] == "t1"
    assert row["kind"] == "task"
    assert row["name"] == "do-thing"
    assert row["status"] == "ok"
    assert bool(row["replayed"]) is False
    assert json.loads(row["attributes_json"]) == {"x": "y"}


def test_persist_span_writes_cold_payload_row(store):
    span = _make_span()
    store.persist_span(span, "r1")
    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM span_payloads WHERE span_id = ?", ("s1",)).fetchone()
    conn.close()
    assert row is not None
    assert json.loads(row["input_json"]) == {"a": 1}
    assert json.loads(row["output_json"]) == {"b": 2}


def test_persist_span_run_id_none_is_allowed(store):
    span = _make_span(span_id="s-noroot")
    store.persist_span(span, None)  # spans outside any run persist with run_id NULL
    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT run_id FROM spans WHERE span_id = ?", ("s-noroot",)).fetchone()
    conn.close()
    assert row["run_id"] is None


def test_persist_span_records_usage_and_cost(store):
    span = _make_span(
        span_id="s-usage",
        kind="llm",
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.02),
    )
    store.persist_span(span, "r1")
    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM spans WHERE span_id = ?", ("s-usage",)).fetchone()
    conn.close()
    assert row["cost_usd"] == pytest.approx(0.02)
    usage = from_jsonable(json.loads(row["usage_json"]))
    assert usage.input_tokens == 10


def test_persist_span_records_error(store):
    span = _make_span(
        span_id="s-err",
        status="error",
        error=tracing.ErrorInfo(type="ValueError", message="boom", stacktrace="tb..."),
    )
    store.persist_span(span, "r1")
    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT error_json FROM spans WHERE span_id = ?", ("s-err",)).fetchone()
    conn.close()
    error = json.loads(row["error_json"])
    assert error == {"type": "ValueError", "message": "boom", "stacktrace": "tb..."}


def test_persist_span_replayed_flag_stored(store):
    span = _make_span(span_id="s-replay", replayed=True)
    store.persist_span(span, "r1")
    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT replayed FROM spans WHERE span_id = ?", ("s-replay",)).fetchone()
    conn.close()
    assert bool(row["replayed"]) is True


def test_persist_span_fts_row_when_available(store):
    span = _make_span(span_id="s-fts")
    store.persist_span(span, "r1")
    if not store._fts_available:
        pytest.skip("FTS5 not compiled into this sqlite3 build -- feature degrades cleanly")
    conn = sqlite3.connect(store._path)
    row = conn.execute("SELECT * FROM span_fts WHERE span_id = ?", ("s-fts",)).fetchone()
    conn.close()
    assert row is not None


def test_persist_span_transient_fts_insert_failure_warns_once_and_stays_available(store):
    """Regression: a *single* failed FTS insert (lock contention, or any
    other transient OperationalError unrelated to FTS5 availability) must
    not permanently flip `_fts_available` to False the way a genuine
    "FTS5 isn't compiled in" failure does at schema-init time -- that would
    silently and permanently disable search after one blip, with no
    warning. Simulates the transient failure the way the original finding
    reproduced it: drop `span_fts` out from under a store that successfully
    detected it as available."""
    if not store._fts_available:
        pytest.skip("FTS5 not compiled into this sqlite3 build")

    store.persist_span(_make_span(span_id="s1"), "r1")  # succeeds normally
    assert store._fts_available is True

    conn = sqlite3.connect(store._path)
    conn.execute("DROP TABLE span_fts")
    conn.commit()
    conn.close()

    with pytest.warns(RuntimeWarning):
        store.persist_span(_make_span(span_id="s2"), "r1")
    # Still considered available -- only a genuine "FTS5 not compiled in"
    # failure (at schema-init time) should ever flip this permanently.
    assert store._fts_available is True
    # The non-FTS rows still persisted despite the FTS insert failing.
    assert store.get_run  # sanity: store object still usable
    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM spans WHERE span_id = ?", ("s2",)).fetchone()
    conn.close()
    assert row is not None

    # Second (and any further) failure: no additional warning -- warns once.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store.persist_span(_make_span(span_id="s3"), "r1")


def test_schema_version_mismatch_raises_config_error(tmp_path):
    """Regression: opening a `runs.db` some other schema version wrote must
    fail loud (ConfigError) rather than silently breaking the first time
    code touches a column that version's tables lack (the original finding
    reproduced this concretely: an older-shape `spans` table missing
    `cost_usd`/`replayed` raised a raw sqlite3.OperationalError deep inside
    `persist_span`, which then got converted into "tracing silently goes
    dark forever" by the span-sink's one-time-warning handling)."""
    path = tmp_path / "old.db"
    # Hand-build a pre-versioning-shaped `runs` table (no PRAGMA user_version
    # ever set -> defaults to 0) so RunStore sees a `runs` table already
    # exists but at an unexpected version.
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, kind TEXT)")
    conn.commit()
    conn.close()

    from composeai.errors import ConfigError

    with pytest.raises(ConfigError, match="schema"):
        RunStore(path)


def test_schema_version_set_on_fresh_store(tmp_path):
    path = tmp_path / "fresh.db"
    store = RunStore(path)
    conn = sqlite3.connect(path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert version == store._SCHEMA_VERSION == 1


def test_persist_span_never_crashes_on_unserializable_payload(store):
    class Weird:
        def __repr__(self):
            return "<weird>"

    span = _make_span(span_id="s-weird", input=Weird(), output=Weird())
    store.persist_span(span, "r1")  # must not raise
    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM span_payloads WHERE span_id = ?", ("s-weird",)).fetchone()
    conn.close()
    assert row is not None


# --- open_default / reset_default / COMPOSE_DIR ----------------------------


@pytest.fixture
def _reset_default_around():
    """Guarantee ``runs.reset_default()`` runs even if the test body fails.

    Tests below reach directly into the module-global default-store cache
    (via ``monkeypatch.setenv("COMPOSE_DIR", ...)``); an assertion failure
    partway through a bare `reset_default()`-at-the-end test would leave a
    stale cached store pointed at that test's temp dir for whatever runs
    next -- this fixture's `yield`/`finally` shape can't skip its cleanup
    that way.
    """
    yield
    runs.reset_default()


def test_open_default_uses_compose_dir_env(tmp_path, monkeypatch, _reset_default_around):
    monkeypatch.setenv("COMPOSE_DIR", str(tmp_path / "mycompose"))
    runs.reset_default()
    store1 = runs.open_default()
    assert (tmp_path / "mycompose" / "runs.db").exists()
    store2 = runs.open_default()
    assert store1 is store2  # cached singleton


def test_reset_default_drops_cached_store(tmp_path, monkeypatch, _reset_default_around):
    monkeypatch.setenv("COMPOSE_DIR", str(tmp_path / "c1"))
    runs.reset_default()
    store1 = runs.open_default()
    monkeypatch.setenv("COMPOSE_DIR", str(tmp_path / "c2"))
    runs.reset_default()
    store2 = runs.open_default()
    assert store1 is not store2
    assert (tmp_path / "c2" / "runs.db").exists()


# --- span sink wiring -------------------------------------------------------


def test_open_default_installs_span_sink_and_persists_spans(
    tmp_path, monkeypatch, _reset_default_around
):
    from composeai._storeasync import worker_for

    monkeypatch.setenv("COMPOSE_DIR", str(tmp_path / "sinktest"))
    runs.reset_default()
    store = runs.open_default()
    with tracing.span("task", "standalone"):
        pass
    # The default sink now casts `persist_span` onto the store's worker
    # thread (off the emitting thread -- see `runs._default_span_sink` /
    # `StoreWorker.cast`), so the row lands asynchronously; draining the
    # worker (close() blocks until its queue is empty) makes the write
    # observable before this reads it back directly.
    worker_for(store).close()
    conn = sqlite3.connect(store._path)
    row = conn.execute("SELECT span_id FROM spans WHERE name = ?", ("standalone",)).fetchone()
    conn.close()
    assert row is not None


def test_sink_failure_warns_once_from_worker_thread_and_run_continues(
    tmp_path, monkeypatch, _reset_default_around
):
    """Regression, restored for async span persistence (v0.4.0 Plan B Task 1):
    `persist_span` failures used to be caught synchronously in
    ``tracing._notify_span_sink`` (which warned once per process) because
    the old sink called ``store.persist_span`` directly on the emitting
    thread. The sink now casts the write onto the store's worker thread
    (``StoreWorker.cast``, no reply) -- a genuine `persist_span` failure
    happens asynchronously, well after ``_default_span_sink`` already
    returned, so ``_notify_span_sink``'s try/except can no longer observe
    it (``cast()`` is fire-and-forget by design: no reply channel exists to
    carry an exception back). A fire-and-forget cast that just swallowed
    the exception outright would silently break the documented contract
    (``tracing.set_span_sink``'s docstring, ``RunStore._init_schema``'s
    comment) that the *first* persistence failure per process is never
    silent -- so ``StoreWorker._resolve`` now emits that one
    ``RuntimeWarning`` itself, from the worker thread, the first time a
    cast job's store method raises (see
    ``_storeasync._warn_once_on_cast_failure``), with every failure after
    that staying silent. Never crashes (or even warns synchronously in) the
    run either way.
    """
    from composeai import _storeasync
    from composeai._storeasync import worker_for

    monkeypatch.setenv("COMPOSE_DIR", str(tmp_path / "failtest"))
    runs.reset_default()  # also clears the "already warned" latch below
    store = runs.open_default()

    def boom(self, span, run_id):
        raise RuntimeError("disk full")

    monkeypatch.setattr(RunStore, "persist_span", boom)

    # `warnings.catch_warnings`'s bookkeeping (the filter list, `showwarning`)
    # is process-global, not thread-local, so a warning raised on the
    # worker thread is captured here too -- AS LONG AS it actually runs
    # before this `with` block exits. `worker_for(store).close()` blocks
    # until the queue (including the failing cast job) is fully drained,
    # so it's called *inside* the block to force that ordering.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with tracing.span("task", "t1"):
            pass
        worker_for(store).close()

    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert "persisting a span failed" in str(caught[0].message)
    assert _storeasync._cast_warned is True

    # Second failure (a fresh worker -- the previous one is now closed):
    # the latch is process-wide, not per-worker, so this stays silent.
    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always")
        with tracing.span("task", "t2"):
            pass
        worker_for(store).close()
    assert len(caught_again) == 0

    # Minor: the worker loop itself survived both failing casts -- a
    # subsequent call_blocking still completes normally rather than hanging
    # or raising, proving the dedicated writer thread never died mid-job.
    assert worker_for(store).call_blocking("get_run", "does-not-exist") is None


def test_standalone_agent_run_id_matches_store_row():
    """Regression: Run.id must equal the durable runs-table row id."""
    from composeai import agent, runs
    from composeai.testing import FakeModel

    @agent(model=FakeModel(script=["done"]))
    def solo(text: str) -> str:
        """Solo."""
        return text

    run = solo.run("hi")
    store = runs.open_default()
    row = store.get_run(run.id)
    assert row is not None
    assert row["kind"] == "agent"
    assert row["status"] == "completed"


# --- budget_json round trip --------------------------------------------------


def test_create_run_persists_and_reads_back_budget_json(store):
    store.create_run(**_run_kwargs(budget_json='{"usd": 0.5, "tokens": null}'))
    row = store.get_run("r1")
    assert row["budget_json"] == '{"usd": 0.5, "tokens": null}'


def test_encode_decode_budget_round_trip():
    b = runs.Budget(usd=0.5, tokens=1000)
    encoded = runs.encode_budget(b)
    decoded = runs.decode_budget(encoded)
    assert decoded == b
    assert runs.encode_budget(None) is None
    assert runs.decode_budget(None) is None
    assert runs.decode_budget("") is None


# --- persist_pause: atomic snapshot + pending interrupts + status ----------


class _FakeInterrupt:
    def __init__(self, id: str, payload: Any = None) -> None:
        self.id = id
        self.kind = "approval"
        self.question = None
        self.payload = payload


def test_persist_pause_writes_snapshot_interrupts_and_status_atomically(store):
    store.create_run(**_run_kwargs(run_id="r1", status="running"))

    store.persist_pause(
        run_id="r1",
        interrupts=[_FakeInterrupt("tool:a:call_1"), _FakeInterrupt("tool:b:call_2")],
        scope_key="",
        messages_json='{"messages": true}',
        partial_results_json='{"partial": true}',
        turn=3,
    )

    row = store.get_run("r1")
    assert row["status"] == "paused"

    pending = {p["interrupt_id"] for p in store.pending_interrupts_all("r1")}
    assert pending == {"tool:a:call_1", "tool:b:call_2"}

    state = store.agent_state_get("r1", "")
    assert state is not None
    assert state["messages_json"] == '{"messages": true}'
    assert state["turn"] == 3


def test_persist_pause_with_no_interrupts_still_snapshots_and_pauses(store):
    """A pause propagating from a nested ask_human() (not an approval gate)
    has nothing of its own to add to `pending_interrupts` here -- the
    snapshot and status transition must still happen."""
    store.create_run(**_run_kwargs(run_id="r1", status="running"))

    store.persist_pause(
        run_id="r1",
        interrupts=[],
        scope_key="",
        messages_json='{"messages": true}',
        partial_results_json="{}",
        turn=1,
    )

    assert store.get_run("r1")["status"] == "paused"
    assert store.pending_interrupts_all("r1") == []
    assert store.agent_state_get("r1", "") is not None


# --- delete_run: full purge --------------------------------------------------


def test_delete_run_purges_every_associated_table(store):
    store.create_run(**_run_kwargs(run_id="r1"))
    store.journal_put("r1", "step#1", "1")
    store.pending_interrupt_put(
        run_id="r1",
        interrupt_id="go",
        kind="approval",
        question=None,
        payload_json="{}",
        created_at=1.0,
    )
    store.agent_state_put(
        run_id="r1",
        scope_key="",
        messages_json="[]",
        partial_results_json="{}",
        turn=1,
        updated_at=1.0,
    )
    trace = tracing.Trace(trace_id="t1")
    span = tracing.Span(trace_id="t1", kind="task", name="x", started_at=1.0)
    trace.add(span)
    store.persist_span(span, "r1")

    store.delete_run("r1")

    assert store.get_run("r1") is None
    assert store.journal_all("r1") == {}
    assert store.pending_interrupts_all("r1") == []
    assert store.agent_state_get("r1", "") is None
    conn = store._connect()
    assert conn.execute("SELECT * FROM spans WHERE run_id = ?", ("r1",)).fetchall() == []


def test_delete_run_purges_span_payloads(store):
    span = _make_span(span_id="s-task8")
    store.persist_span(span, "run-task8")
    conn = store._connect()
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM span_payloads WHERE span_id = ?", ("s-task8",)
        ).fetchone()[0]
        == 1
    )

    store.delete_run("run-task8")

    assert (
        conn.execute(
            "SELECT COUNT(*) FROM span_payloads WHERE span_id = ?", ("s-task8",)
        ).fetchone()[0]
        == 0
    )


# --- RunContext scope stack: deterministic keys under concurrent dispatch --


def test_run_context_next_key_scoped_by_active_scope_stack(store):
    ctx = runs.RunContext(run_id="r1", store=store)
    assert ctx.next_key("t") == "t#1"
    assert ctx.next_key("t") == "t#2"  # unscoped counter keeps incrementing

    with runs.push_scope("map#1[0]"):
        assert ctx.next_key("t") == "map#1[0]/t#1"  # independent counter in this scope
        with runs.push_scope("inner"):
            assert ctx.next_key("t") == "map#1[0]/inner/t#1"
    with runs.push_scope("map#1[1]"):
        assert ctx.next_key("t") == "map#1[1]/t#1"  # a sibling scope, unaffected by [0]'s count

    assert ctx.next_key("t") == "t#3"  # back to top level, unaffected by any scope's count


def test_reserve_scope_segment_returns_local_segment_only(store):
    ctx = runs.RunContext(run_id="r1", store=store)
    with runs.push_scope("outer"):
        segment = ctx.reserve_scope_segment("map")
        assert segment == "map#1"  # local, not prefixed by "outer"
        assert ctx.qualify(segment) == "outer/map#1"


# --- abandon_guard: revokes journal access once the run is abandoned -------


def test_abandon_guard_blocks_journal_access_only_after_abandoned(store):
    ctx = runs.RunContext(run_id="r1", store=store)
    abandoned = threading.Event()
    guarded = runs.abandon_guard(ctx, abandoned)
    assert guarded is not None

    # Before abandonment: behaves exactly like the wrapped RunContext.
    key = guarded.next_key("t")
    assert key == "t#1"
    assert guarded.journal_record(key, 42) == 42

    abandoned.set()
    from composeai.runs import _AbandonedTaskError

    with pytest.raises(_AbandonedTaskError):
        guarded.next_key("t")
    with pytest.raises(_AbandonedTaskError):
        guarded.journal_record("t#2", 1)
    with pytest.raises(_AbandonedTaskError):
        guarded.journal_lookup("t#1")
    with pytest.raises(_AbandonedTaskError):
        guarded.reserve_scope_segment("t")


def test_abandon_guard_of_none_context_is_none():
    assert runs.abandon_guard(None, threading.Event()) is None


# --- resolve_answer_key / apply_resume_answers ------------------------------


def test_resolve_answer_key_no_match_against_all_tool_pending_raises_config_error():
    from composeai.errors import ConfigError

    pending = {"tool:dangerous:call_1"}
    with pytest.raises(ConfigError):
        runs.resolve_answer_key(pending, "send_email")


def test_resolve_answer_key_bare_id_not_yet_pending_passes_through():
    """A bare, caller-chosen approve()/ask_human() id may be supplied before
    the flow has even reached it -- must not raise, since it's looked up
    verbatim later regardless of current pending state."""
    pending = {"first"}
    assert runs.resolve_answer_key(pending, "second") == "second"


def test_resolve_answer_key_no_pending_at_all_passes_through_as_bare_id():
    assert runs.resolve_answer_key(set(), "anything") == "anything"
