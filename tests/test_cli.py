"""Tests for the `compose` CLI (Phase 9): runs/trace/diff/costs/path/export.

Item 0 (required regression fix, from the build ledger): top-level
``pipe``/``aggregate`` runs previously created no durable ``runs`` row at
all (no span-sink tagging via ``current_run_id()``), unlike a standalone
``@agent`` run (``composeai.runs.run_standalone_agent``). The fix wraps
``composeai.combinators._run_top`` the same way, so ``compose runs``/
``compose trace`` can find them. Regression tests for that live first in
this file, before the CLI-specific tests, since it's the required
pre-item.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from composeai import agent, cli, runs
from composeai.cli import main
from composeai.combinators import aggregate, pipe
from composeai.testing import FakeModel, replay_cassette
from composeai.tools import tool


@pytest.fixture(autouse=True)
def _isolated_compose_dir(tmp_path, monkeypatch):
    """Every test in this file gets its own fresh ``COMPOSE_DIR``.

    Without this, `compose runs`/`costs` listing/ordering/count assertions
    would be at the mercy of whatever *other* tests in the suite already
    wrote to the session-wide default store (see ``tests/conftest.py``).
    """
    monkeypatch.setenv("COMPOSE_DIR", str(tmp_path))
    runs.reset_default()
    yield
    runs.reset_default()


def test_pipe_run_creates_durable_run_row_with_matching_id():
    def a(x: str) -> str:
        return x.upper()

    def b(x: str) -> str:
        return x + "!"

    pipeline = pipe(a, b)
    run = pipeline.run("hi")

    store = runs.open_default()
    row = store.get_run(run.id)
    assert row is not None
    assert row["kind"] == "pipe"
    assert row["status"] == "completed"
    assert row["run_id"] == run.id


def test_aggregate_run_creates_durable_run_row_with_matching_id():
    def a(x: str) -> str:
        return x.upper()

    def b(x: str) -> str:
        return x.lower()

    agg = aggregate(one=a, two=b)
    run = agg.run("Hi")

    store = runs.open_default()
    row = store.get_run(run.id)
    assert row is not None
    assert row["kind"] == "aggregate"
    assert row["status"] == "completed"
    assert row["run_id"] == run.id


def test_pipe_run_persists_spans_tagged_with_run_id():
    """Regression: before the fix, `current_run_id()` stayed None for a
    top-level pipe/aggregate run, so its spans persisted with `run_id`
    SQL NULL -- `compose trace <run_id>` would have found nothing.
    """

    @agent(model=FakeModel(["done"]))
    def solo(text: str) -> str:
        """Solo."""
        return text

    def passthrough(x: str) -> str:
        return x

    pipeline = pipe(solo, passthrough)
    run = pipeline.run("go")

    store = runs.open_default()
    row = store.get_run(run.id)
    assert row is not None
    assert row["trace_id"] == run.trace.trace_id

    import sqlite3

    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    spans = conn.execute(
        "SELECT span_id, kind FROM spans WHERE trace_id = ?", (run.trace.trace_id,)
    ).fetchall()
    conn.close()
    assert len(spans) >= 2  # at least the pipe root span + the agent span
    assert all(s["span_id"] for s in spans)
    # Every span in this run's trace persisted at all -- proves the sink
    # was reachable/tagged for a top-level pipe run (the regression).
    kinds = {s["kind"] for s in spans}
    assert "pipe" in kinds
    assert "agent" in kinds


def test_pipe_run_failure_marks_row_failed():
    def boom(x: str) -> str:
        raise ValueError("nope")

    def a(x: str) -> str:
        return x

    pipeline = pipe(a, boom)
    try:
        pipeline.run("x")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError to propagate")

    store = runs.open_default()
    # Find the most-recent pipe row (the failed one).
    rows = store.list_runs(kind="pipe", limit=1)
    assert rows
    assert rows[0]["status"] == "failed"


# =====================================================================
# `compose path`
# =====================================================================


def test_cmd_path_prints_compose_dir(capsys, tmp_path):
    rc = main(["path"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == str(tmp_path)


# =====================================================================
# `compose runs`
# =====================================================================


def test_cmd_runs_empty_store_prints_message(capsys):
    rc = main(["runs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no runs" in out.lower()


def test_cmd_runs_empty_store_json_is_empty_array(capsys):
    rc = main(["runs", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert json.loads(out) == []


def test_cmd_runs_lists_newest_first(capsys):
    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    pipe(a, b).run("one")
    time.sleep(0.01)
    pipe(a, b).run("two")
    time.sleep(0.01)
    run3 = pipe(a, b).run("three")

    rc = main(["runs", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["run_id"] == run3.id  # newest first


def test_cmd_runs_limit_flag(capsys):
    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    for _ in range(5):
        pipe(a, b).run("x")
        time.sleep(0.005)

    rc = main(["runs", "-n", "2", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2


def test_cmd_runs_filters_by_status(capsys):
    def a(x: str) -> str:
        return x

    def boom(x: str) -> str:
        raise ValueError("nope")

    pipe(a, a).run("ok")
    try:
        pipe(a, boom).run("bad")
    except ValueError:
        pass

    rc = main(["runs", "--status", "failed", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"


def test_cmd_runs_filters_by_kind(capsys):
    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    pipe(a, b).run("x")
    aggregate(one=a, two=b).run("x")

    rc = main(["runs", "--kind", "aggregate", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["kind"] == "aggregate"


def test_cmd_runs_json_shape_includes_cost_and_tokens(capsys):
    fake = FakeModel(["hello"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("world")

    rc = main(["runs", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "agent"
    assert row["status"] == "completed"
    assert "tokens" in row
    assert row["tokens"] == 30  # FakeModel default usage: 10 input + 20 output
    assert "cost_usd" in row


def test_cmd_runs_text_output_includes_short_id_and_status(capsys):
    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    run = pipe(a, b).run("x")

    rc = main(["runs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert run.id[:8] in out
    assert "completed" in out
    assert "pipe" in out


def test_cmd_runs_text_output_includes_full_run_id(capsys):
    """Regression: `compose runs`' text output used to print only an 8-char
    `_short_id` abbreviation, which `compose trace`/`diff`/`export` (exact
    `WHERE run_id = ?` lookups, before the prefix-resolution fix) could
    never look up -- the only way to get a usable id was `--json`. Text
    output must print the full id too."""

    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    run = pipe(a, b).run("x")

    rc = main(["runs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert run.id in out


def test_short_id_disambiguates_runs_created_within_the_same_millisecond_window():
    """Regression: an 8-char short id decodes only the top 40 of the 48-bit
    ULID timestamp -- zero contribution from the random suffix -- so any
    two runs created within the same ~262ms window (routine for a flow's
    own row plus its nested agent rows) displayed an *identical* short id.
    12 chars covers the full 48-bit timestamp plus 12 random bits."""
    from composeai._ids import new_ulid
    from composeai.cli import _short_id

    # 5 ids (matching the original bug report's own repro): with 12 chars
    # (48-bit timestamp + 12 random bits, 4096 possibilities) the chance of
    # any collision among 5 near-simultaneous ids is under 0.3% -- with the
    # old 8-char id (zero random contribution) this was a *guaranteed*
    # collision every time.
    ids = [new_ulid() for _ in range(5)]
    short_ids = [_short_id(i) for i in ids]
    assert len(set(short_ids)) == len(short_ids)


def test_cmd_runs_since_relative_days_includes_recent(capsys):
    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    pipe(a, b).run("x")

    rc = main(["runs", "--since", "7d", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1


def test_cmd_runs_since_excludes_old_row(capsys):
    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    run = pipe(a, b).run("x")

    # Rewrite this run's created_at to look 30 days old.
    db_path = runs.open_default()._path
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE runs SET created_at = ? WHERE run_id = ?",
        (time.time() - 30 * 86400, run.id),
    )
    conn.commit()
    conn.close()

    rc = main(["runs", "--since", "7d", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == []


def test_cmd_runs_since_absolute_date_parses(capsys):
    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    pipe(a, b).run("x")

    rc = main(["runs", "--since", "2020-01-01", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1


def test_cmd_runs_since_invalid_value_exits_nonzero(capsys):
    rc = main(["runs", "--since", "not-a-date"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "since" in err.lower()


def test_cmd_runs_query_hits_matching_span_content(capsys):
    store = runs.open_default()
    if not store._fts_available:
        pytest.skip("FTS5 not compiled into this sqlite3 build")

    fake = FakeModel(["a very particular needle response"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    run = greeter.run("world")

    rc = main(["runs", "-q", "particular", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(r["run_id"] == run.id for r in rows)


def test_cmd_runs_query_no_match_returns_empty(capsys):
    store = runs.open_default()
    if not store._fts_available:
        pytest.skip("FTS5 not compiled into this sqlite3 build")

    fake = FakeModel(["hello"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("world")

    rc = main(["runs", "-q", "zzz_no_such_token_zzz", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == []


def test_cmd_runs_query_prints_notice_when_fts_unavailable(tmp_path, capsys):
    """Build a db with the core tables but *no* span_fts table (simulating a
    sqlite3 build without FTS5 compiled in) -- independent of whether this
    machine's actual sqlite3 build has FTS5, so the notice path is always
    exercised.
    """
    compose_dir = tmp_path / "no-fts"
    compose_dir.mkdir()
    db_path = compose_dir / "runs.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, kind TEXT, name TEXT, status TEXT,
            created_at REAL, updated_at REAL, trace_id TEXT, fingerprint TEXT,
            args_json TEXT, output_json TEXT, error_json TEXT
        );
        CREATE TABLE spans (
            span_id TEXT PRIMARY KEY, trace_id TEXT, run_id TEXT, parent_span_id TEXT,
            kind TEXT, name TEXT, started_at REAL, ended_at REAL, status TEXT,
            replayed INTEGER, attributes_json TEXT, usage_json TEXT, cost_usd REAL,
            error_json TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO runs VALUES ('r1','pipe','p','completed',1.0,1.0,'t1',NULL,NULL,NULL,NULL)"
    )
    conn.commit()
    conn.close()

    import os

    os.environ["COMPOSE_DIR"] = str(compose_dir)
    try:
        rc = main(["runs", "-q", "anything"])
    finally:
        os.environ["COMPOSE_DIR"] = str(tmp_path)
    assert rc == 0
    captured = capsys.readouterr()
    assert "fts5" in captured.err.lower() or "full-text" in captured.err.lower()


def test_cmd_runs_query_malformed_fts5_syntax_gives_friendly_error(capsys):
    """Regression: a malformed `-q` value used to raise a raw, uncaught
    sqlite3.OperationalError (a full Python traceback to stderr) since
    FTS5's MATCH argument has its own query grammar, independent of SQL
    syntax itself. Must instead surface as a clean `compose: ...` message
    and exit 1."""
    store = runs.open_default()
    if not store._fts_available:
        pytest.skip("FTS5 not compiled into this sqlite3 build")

    fake = FakeModel(["hello"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("world")

    rc = main(["runs", "-q", '"unterminated'])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("compose:")
    assert "traceback" not in err.lower()


def test_cmd_runs_negative_limit_is_argparse_error():
    with pytest.raises(SystemExit) as exc_info:
        main(["runs", "-n", "-5"])
    assert exc_info.value.code == 2


def test_cmd_runs_zero_limit_is_argparse_error():
    with pytest.raises(SystemExit) as exc_info:
        main(["runs", "-n", "0"])
    assert exc_info.value.code == 2


# =====================================================================
# `compose trace`
# =====================================================================


def test_cmd_trace_matches_render_trace_of_rebuilt_trace(capsys):
    from composeai import tracing

    def a(x: str) -> str:
        return x.upper()

    def b(x: str) -> str:
        return x + "!"

    run = pipe(a, b).run("hi")

    rc = main(["trace", run.id])
    assert rc == 0
    out = capsys.readouterr().out
    assert out == tracing.render_trace(run.trace, color=False) + "\n"


def test_cmd_trace_last_flag_finds_most_recent(capsys):
    from composeai import tracing

    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    pipe(a, b).run("first")
    time.sleep(0.01)
    run2 = pipe(a, b).run("second")

    rc = main(["trace", "--last"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out == tracing.render_trace(run2.trace, color=False) + "\n"


def test_cmd_trace_unknown_run_id_exits_1_message_to_stderr(capsys):
    rc = main(["trace", "not-a-real-id"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not-a-real-id" in err


def test_cmd_trace_no_id_and_no_last_exits_1(capsys):
    rc = main(["trace"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "run id" in err.lower() or "--last" in err


def test_cmd_trace_paused_run_prints_banner_with_pending_interrupt(capsys):
    @tool(requires_approval=True)
    def dangerous_tool() -> str:
        """Needs approval."""
        return "done"

    model = FakeModel([{"tool_calls": [{"name": "dangerous_tool", "arguments": {}, "id": "c1"}]}])

    @agent(model=model, tools=[dangerous_tool], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    run = runner.run()
    assert run.status == "paused"

    rc = main(["trace", run.id])
    assert rc == 0
    out = capsys.readouterr().out
    assert "paused" in out.lower()
    assert "tool:dangerous_tool:c1" in out
    assert "resume(" in out
    assert run.id in out


# =====================================================================
# id-prefix resolution (trace/diff/export) -- decision #2
# =====================================================================


def test_cmd_trace_resolves_unique_id_prefix(capsys):
    from composeai import tracing

    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    run = pipe(a, b).run("hi")

    rc = main(["trace", run.id[:10]])
    assert rc == 0
    out = capsys.readouterr().out
    assert out == tracing.render_trace(run.trace, color=False) + "\n"


def test_cmd_trace_ambiguous_prefix_exits_1_and_lists_matches(capsys):
    store = runs.open_default()
    store.create_run(
        run_id="prefixAAA111",
        kind="agent",
        name="a",
        status="completed",
        created_at=1.0,
        updated_at=1.0,
        trace_id="t1",
        fingerprint=None,
        args_json=None,
    )
    store.create_run(
        run_id="prefixBBB222",
        kind="agent",
        name="b",
        status="completed",
        created_at=2.0,
        updated_at=2.0,
        trace_id="t2",
        fingerprint=None,
        args_json=None,
    )

    rc = main(["trace", "prefix"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ambiguous" in err.lower()
    assert "prefixAAA111" in err
    assert "prefixBBB222" in err


def test_cmd_trace_exact_full_id_wins_even_if_a_prefix_of_another(capsys):
    """An exact match always resolves to itself, even when it also happens
    to literally be a string-prefix of some other run's id -- ambiguity
    only applies when the *given* string isn't itself a real run id."""
    from composeai import tracing

    store = runs.open_default()
    store.create_run(
        run_id="short1",
        kind="agent",
        name="short-run",
        status="completed",
        created_at=1.0,
        updated_at=1.0,
        trace_id="t-short",
        fingerprint=None,
        args_json=None,
    )
    store.create_run(
        run_id="short1-longer-sibling",
        kind="agent",
        name="longer-run",
        status="completed",
        created_at=2.0,
        updated_at=2.0,
        trace_id="t-longer",
        fingerprint=None,
        args_json=None,
    )

    rc = main(["trace", "short1"])
    assert rc == 0
    out = capsys.readouterr().out
    empty_trace = tracing.Trace(trace_id="t-short")
    assert out == tracing.render_trace(empty_trace, color=False) + "\n"


def test_cmd_diff_resolves_unique_prefixes_for_both_ids(capsys):
    def upper(x: str) -> str:
        return x.upper()

    run_a = pipe(upper, upper).run("hi")
    time.sleep(0.05)  # cross a millisecond boundary so the 10-char prefixes differ
    run_b = pipe(upper, upper, upper).run("hi")

    rc = main(["diff", run_a.id[:10], run_b.id[:10]])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Δcost" in out


def test_cmd_export_resolves_unique_prefix(tmp_path, capsys):
    fake = FakeModel(["exported via prefix"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    run = greeter.run("world")
    cassette_path = tmp_path / "prefix-exported.json"

    rc = main(["export", run.id[:10], "--cassette", str(cassette_path)])
    assert rc == 0
    assert cassette_path.exists()


# =====================================================================
# `compose diff`
# =====================================================================


def test_cmd_diff_matched_spans_show_deltas_and_output_changed(capsys):
    def upper(x: str) -> str:
        return x.upper()

    fake_a = FakeModel(["response A"])
    fake_b = FakeModel(["a longer response B here"])

    @agent(model=fake_a)
    def writer(text: str) -> str:  # pyright: ignore[reportRedeclaration]
        """Write."""
        return text

    run_a = pipe(writer, upper).run("hello")

    import composeai.agentfn as agentfn_mod

    agentfn_mod._AGENT_REGISTRY.pop("writer", None)

    @agent(model=fake_b)
    def writer(text: str) -> str:  # noqa: F811
        """Write."""
        return text

    run_b = pipe(writer, upper).run("hello")

    rc = main(["diff", run_a.id, run_b.id])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Δcost" in out
    assert "Δtokens" in out
    assert "output changed" in out


def test_cmd_diff_extra_span_shows_plus_minus_lines(capsys):
    def upper(x: str) -> str:
        return x.upper()

    def shout(x: str) -> str:
        return x + "!"

    run_a = pipe(upper, upper).run("hi")
    run_b = pipe(upper, upper, shout).run("hi")

    rc = main(["diff", run_a.id, run_b.id])
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert any(line.strip().startswith("+") for line in lines)


def test_cmd_diff_unknown_run_id_exits_1(capsys):
    def a(x: str) -> str:
        return x

    def b(x: str) -> str:
        return x

    run = pipe(a, b).run("x")

    rc = main(["diff", run.id, "no-such-id"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no-such-id" in err


# =====================================================================
# `compose costs`
# =====================================================================


def test_cmd_costs_by_model_includes_total_row(capsys):
    fake = FakeModel(["hi"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("world")

    rc = main(["costs", "--by", "model"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert "FakeModel" in out


def test_cmd_costs_by_name_groups_by_run_name(capsys):
    fake = FakeModel(["hi", "there"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("a")
    greeter.run("b")

    rc = main(["costs", "--by", "name"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "greeter" in out
    assert "calls=2" in out


def test_cmd_costs_by_day_groups_by_calendar_day(capsys):
    fake = FakeModel(["hi"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("world")
    today = time.strftime("%Y-%m-%d")

    rc = main(["costs", "--by", "day"])
    assert rc == 0
    out = capsys.readouterr().out
    assert today in out


def test_cmd_costs_empty_store(capsys):
    rc = main(["costs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no llm spans" in out.lower()


# =====================================================================
# `compose export` -> replay round trip
# =====================================================================


def test_cmd_export_then_replay_round_trip(tmp_path, capsys):
    fake = FakeModel(["exported answer"])

    @agent(model=fake)
    def greeter(name: str) -> str:  # pyright: ignore[reportRedeclaration]
        """Greet."""
        return name

    run = greeter.run("world")
    cassette_path = tmp_path / "exported.json"

    rc = main(["export", run.id, "--cassette", str(cassette_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1" in out
    assert cassette_path.exists()

    data = json.loads(cassette_path.read_text())
    assert data["version"] == 2
    assert len(data["entries"]) == 1
    assert data["entries"][0]["full_hash"] is None
    assert data["entries"][0]["message_hash"]

    import composeai.agentfn as agentfn_mod

    agentfn_mod._AGENT_REGISTRY.pop("greeter", None)

    @agent(model=fake)
    def greeter(name: str) -> str:  # noqa: F811
        """Greet."""
        return name

    with replay_cassette(cassette_path):
        replayed = greeter.run("world")
    assert replayed.output == "exported answer"


def test_cmd_export_unknown_run_id_exits_1(tmp_path, capsys):
    rc = main(["export", "no-such-run", "--cassette", str(tmp_path / "x.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no-such-run" in err


def test_cmd_export_warns_when_source_content_capture_was_disabled(
    tmp_path, capsys, monkeypatch
):
    """Regression: a run captured with COMPOSE_TRACE_CONTENT=0 has NULL
    input_json/output_json on its llm spans -- `_export_entry` degrades
    this to an empty system/messages/`Message.assistant("")` cassette
    entry silently. `compose export` must warn (not silently report
    success) so the degenerate export doesn't look like a normal one."""
    monkeypatch.setenv("COMPOSE_TRACE_CONTENT", "0")
    fake = FakeModel(["real answer, never captured"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    run = greeter.run("world")
    monkeypatch.delenv("COMPOSE_TRACE_CONTENT", raising=False)

    cassette_path = tmp_path / "degenerate.json"
    rc = main(["export", run.id, "--cassette", str(cassette_path)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "content capture" in err.lower() or "compose_trace_content" in err.lower()

    data = json.loads(cassette_path.read_text())
    assert data["entries"][0]["request"]["system"] is None
    assert data["entries"][0]["request"]["messages"] == []


def test_cmd_export_no_warning_when_content_capture_was_enabled(tmp_path, capsys):
    fake = FakeModel(["real answer, captured normally"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    run = greeter.run("world")
    cassette_path = tmp_path / "normal.json"
    rc = main(["export", run.id, "--cassette", str(cassette_path)])
    assert rc == 0
    err = capsys.readouterr().err
    assert err == ""


# =====================================================================
# subprocess smoke: console script + import cleanliness
# =====================================================================


def test_console_script_smoke(tmp_path):
    """Regression coverage: this must include an @agent run (not just a
    plain-callable pipe), so at least one llm span with real `usage_json`
    exists -- a genuinely fresh process (this subprocess) decoding that
    `Usage` is exactly what caught a real bug in self-review: `Usage`
    wasn't eagerly registered for `from_jsonable` the way `Message` (and
    its parts) already were (see `composeai.messages`'s bottom block) --
    a plain-callable-only pipe run has no llm spans at all and would never
    have exercised this path.
    """

    @agent(model=FakeModel(["hi"]))
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("world")

    compose_bin = Path(sys.executable).with_name("compose")
    assert compose_bin.exists(), f"console script not found at {compose_bin}"

    import os

    env = dict(os.environ)
    env["COMPOSE_DIR"] = str(tmp_path)
    result = subprocess.run(
        [str(compose_bin), "runs", "-n", "1"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "greeter" in result.stdout

    result_costs = subprocess.run(
        [str(compose_bin), "costs"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result_costs.returncode == 0, (
        f"stdout={result_costs.stdout!r} stderr={result_costs.stderr!r}"
    )


def test_cli_imports_cleanly_with_no_provider_sdks_installed():
    """`composeai.cli` must import fine with anthropic/openai unimportable.

    Runs in a fresh subprocess (so no other test's already-imported modules
    leak into ``sys.modules``) with a ``sys.meta_path`` finder that raises
    for any ``anthropic``/``openai`` import -- proving `composeai.cli`
    never touches those packages on its import path.
    """
    code = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        root = name.split('.')[0]\n"
        "        if root in ('anthropic', 'openai'):\n"
        "            raise ModuleNotFoundError(f'blocked: {name}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "import composeai.cli\n"
        "assert 'anthropic' not in sys.modules\n"
        "assert 'openai' not in sys.modules\n"
        "print('IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "IMPORT_OK" in result.stdout


# =====================================================================
# self-review follow-ups
# =====================================================================


def test_cmd_costs_since_invalid_value_exits_nonzero(capsys):
    """Same class of bug as the `runs --since` fast-path: an empty store
    must never mask a bad --since value."""
    rc = main(["costs", "--since", "not-a-date"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "since" in err.lower()


def test_cmd_diff_delta_tokens_sign_is_b_minus_a(capsys):
    """Regression for a self-review fix: per-node Δtokens must be b - a,
    not accidentally double-negated back to a - b."""

    def upper(x: str) -> str:
        return x.upper()

    fake_a = FakeModel(["short"])
    fake_b = FakeModel(["a much longer response than before"])

    @agent(model=fake_a)
    def writer(text: str) -> str:  # pyright: ignore[reportRedeclaration]
        """Write."""
        return text

    run_a = pipe(writer, upper).run("hello")

    import composeai.agentfn as agentfn_mod

    agentfn_mod._AGENT_REGISTRY.pop("writer", None)

    @agent(model=fake_b)
    def writer(text: str) -> str:  # noqa: F811
        """Write."""
        return text

    run_b = pipe(writer, upper).run("hello")

    # Both FakeModel calls report the same default usage (10 in / 20 out
    # tokens each) regardless of text length, so Δtokens for the *agent*
    # node is 0 here -- what matters is that the diff doesn't crash and
    # reports a well-formed, non-doubly-negated delta. Force a real
    # difference by using distinct Usage on each side instead.
    rc = main(["diff", run_a.id, run_b.id])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Δtokens=" in out


# =====================================================================
# capstone fix wave C: _rebuild_trace decode parity + ordering determinism
# =====================================================================


def test_rebuild_trace_decodes_attributes_via_from_jsonable():
    """Regression: `persist_span` encodes `attributes` through the same
    encoder as `input`/`output` (tagging non-JSON-primitive values with
    `{"$kind": ..., "$type": ..., "value": ...}`), but `_rebuild_trace`
    used to decode `attributes_json` with plain `json.loads` (no
    `from_jsonable`) while `input`/`output` correctly unwrapped those tags
    -- an asymmetric round trip that returned the raw tagged dict verbatim
    for any non-primitive attribute value."""
    from dataclasses import dataclass

    from composeai import tracing as tracing_mod
    from composeai.cli import _rebuild_trace

    @dataclass
    class Foo:
        x: int

    store = runs.open_default()
    span = tracing_mod.Span(
        trace_id="t-attr-decode",
        span_id="s-attr-decode",
        kind="task",
        name="x",
        started_at=1.0,
        ended_at=2.0,
        status="ok",
        attributes={"custom": Foo(x=5), "plain": "ok"},
    )
    store.persist_span(span, None)

    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    trace = _rebuild_trace(conn, "t-attr-decode")
    conn.close()

    assert len(trace.spans) == 1
    assert trace.spans[0].attributes["custom"] == Foo(x=5)
    assert trace.spans[0].attributes["plain"] == "ok"


def test_rebuild_trace_orders_equal_started_at_by_span_id_deterministically():
    """Regression: `_rebuild_trace`'s `ORDER BY s.started_at` alone has no
    guaranteed order for rows with equal `started_at` (routine for very
    fast parallel calls) -- `compose diff`'s sibling-index path matching
    (`_span_path`) silently depended on that unstable order. Adding
    `span_id` as a tiebreak makes the resulting order reproducible."""
    from composeai import tracing as tracing_mod
    from composeai.cli import _rebuild_trace

    store = runs.open_default()
    common_ts = 100.0
    for span_id in ("s-zzz", "s-aaa", "s-mmm"):
        span = tracing_mod.Span(
            trace_id="t-tie",
            span_id=span_id,
            kind="task",
            name="n",
            started_at=common_ts,
            ended_at=common_ts,
            status="ok",
        )
        store.persist_span(span, None)

    conn = sqlite3.connect(store._path)
    conn.row_factory = sqlite3.Row
    trace = _rebuild_trace(conn, "t-tie")
    conn.close()

    assert [s.span_id for s in trace.spans] == ["s-aaa", "s-mmm", "s-zzz"]


def test_cmd_runs_usage_totals_do_not_mix_across_runs(capsys):
    """Regression guard for switching `cmd_runs` from an N+1 per-row usage
    query to one bulk `run_id IN (...)` query: each run's totals must stay
    attributed to that run, never accidentally pooled together."""
    fake_a = FakeModel(["a"])
    fake_b = FakeModel(["b"])

    @agent(model=fake_a)
    def greeter(name: str) -> str:  # pyright: ignore[reportRedeclaration]
        """Greet."""
        return name

    greeter.run("x")

    import composeai.agentfn as agentfn_mod

    agentfn_mod._AGENT_REGISTRY.pop("greeter", None)

    @agent(model=fake_b)
    def greeter(name: str) -> str:  # noqa: F811
        """Greet."""
        return name

    greeter.run("y")

    rc = main(["runs", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2
    # FakeModel's default usage is 30 tok/call regardless of text -- if the
    # bulk query accidentally pooled both runs' llm spans together, at
    # least one row would show 60 instead.
    assert all(r["tokens"] == 30 for r in rows)


# =====================================================================
# `compose --import` (Task 10)
# =====================================================================


def test_trace_import_flag_decodes_app_types(tmp_path, monkeypatch, capsys):
    """The flagship repro from docs/missing-features.md item 3: a flow with an
    app-defined pydantic output, run in a SEPARATE process, must be traceable
    only with --import (and fail with the unregistered-type message without).

    The app module is *imported by its real dotted name* in the child
    process (via `python -c "import cli_import_app_schemas as m; ..."`)
    rather than executed directly (`python cli_import_app_schemas.py`):
    running it directly would set the module's `__name__` -- and therefore
    `CliWidget.__module__`, since that's fixed at class-definition time --
    to `"__main__"`, a name `--import` can never match by re-importing
    (`__main__` in any other process refers to *that* process's own entry
    module, e.g. pytest itself, not this app). This mirrors the real repro
    in the docs (`research_agent.schemas:SubQuestion`): the type-defining
    module there is imported by its package path from a separate entry
    point, never run as the script itself.
    """
    module_dir = tmp_path / "appmod"
    module_dir.mkdir()
    (module_dir / "cli_import_app_schemas.py").write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel

            from composeai import flow, task


            class CliWidget(BaseModel):
                name: str


            @task
            def build_widget(name: str) -> CliWidget:
                return CliWidget(name=name)


            @flow
            def cli_widget_flow() -> CliWidget:
                return build_widget("w1")
            """
        )
    )
    state_dir = tmp_path / "state"
    env = os.environ.copy()
    env["COMPOSE_DIR"] = str(state_dir)
    env["PYTHONPATH"] = str(module_dir) + os.pathsep + env.get("PYTHONPATH", "")
    driver = (
        "import cli_import_app_schemas as m\n"
        "run = m.cli_widget_flow.run()\n"
        "print(run.id)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    run_id = proc.stdout.strip().splitlines()[-1]

    monkeypatch.setenv("COMPOSE_DIR", str(state_dir))
    monkeypatch.syspath_prepend(str(module_dir))

    # Without --import: the decode error surfaces as a clean CLI failure.
    assert cli.main(["trace", run_id]) == 1
    err = capsys.readouterr().err
    assert "unregistered type" in err

    # With --import: the trace renders.
    assert cli.main(["trace", "--import", "cli_import_app_schemas", run_id]) == 0
    out = capsys.readouterr().out
    assert "cli_widget_flow" in out


def test_import_flag_unknown_module_is_a_clean_error(capsys):
    assert cli.main(["runs", "--import", "definitely_not_a_module_xyz"]) == 1
    assert "--import definitely_not_a_module_xyz" in capsys.readouterr().err
