"""``compose``: a stdlib-only CLI for inspecting composeai's durable ``RunStore``.

Subcommands: ``runs`` (list), ``trace`` (rebuild + render one run's trace),
``diff`` (structural diff of two traces), ``costs`` (group-by spend
report), ``path`` (print the state directory), and ``export`` (turn a
run's stored llm spans into a replayable cassette -- see
:mod:`composeai.testing`).

Deliberately reads the ``runs``/``spans``/``span_payloads``/
``pending_interrupts`` tables (see :mod:`composeai.runs`'s ``RunStore``)
with its own plain :mod:`sqlite3` connections rather than going through
``RunStore``/``open_default()`` -- this is a read-mostly reporting tool,
not a run-producing one, so it has no reason to install the span-
persistence sink or hold a long-lived cached connection the way the
runtime does. The schema itself is owned by :mod:`composeai.runs` and
duplicated here only as the column names each query needs.

Import-path constraint (verified by ``tests/test_cli.py``'s subprocess
test): stdlib + pydantic only. This module must never import
``composeai.models.anthropic``/``composeai.models.openai`` (or the
``anthropic``/``openai`` packages themselves) at import time, so
``compose --help`` (and every other subcommand) works with no provider
SDK installed -- both ``composeai.runs``/``composeai.tracing`` and
``composeai.testing`` (used here only for ``compute_message_hash``, in
``compose export``) already satisfy this (see their own module docs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import tracing
from ._encoding import from_jsonable, to_jsonable
from .errors import ComposeError
from .messages import Message, StopReason, Usage
from .models.base import ModelResponse
from .testing import _CASSETTE_VERSION, compute_message_hash

# --- state directory --------------------------------------------------------


def _compose_dir() -> Path:
    """``{COMPOSE_DIR or ./.compose}`` -- read lazily, same convention as
    ``composeai.runs.open_default``'s ``_default_db_path``."""
    return Path(os.environ.get("COMPOSE_DIR") or "./.compose")


def _db_path() -> Path:
    return _compose_dir() / "runs.db"


def _open_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fts_available(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'span_fts'"
    ).fetchone()
    return row is not None


# --- small formatting helpers ------------------------------------------------


def _parse_since(value: str) -> float:
    """Parse ``--since``: ``"<N>d"``, ``"<N>h"``, or an ``"YYYY-MM-DD"`` date."""
    match = re.fullmatch(r"(\d+)([dh])", value)
    if match:
        count, unit = int(match.group(1)), match.group(2)
        seconds = count * 86400 if unit == "d" else count * 3600
        return datetime.now().timestamp() - seconds
    try:
        return datetime.strptime(value, "%Y-%m-%d").timestamp()
    except ValueError as exc:
        raise ComposeError(
            f"invalid --since value {value!r}: expected '<N>d', '<N>h', or 'YYYY-MM-DD'"
        ) from exc


def _relative_age(started_at: float, *, now: float | None = None) -> str:
    now = now if now is not None else time.time()
    delta = max(0.0, now - started_at)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _decode_usage(usage_json: str | None) -> Usage:
    if usage_json is None:
        return Usage()
    return from_jsonable(json.loads(usage_json))


def _short_id(run_id: str) -> str:
    """A display-only abbreviation of ``run_id`` -- never used to look a run up.

    ``run_id`` is a ULID: a 48-bit millisecond timestamp in the top bits
    followed by 80 random bits (see ``composeai._ids``). 8 Crockford-Base32
    characters decode only 40 bits -- entirely inside that timestamp, with
    zero contribution from the random suffix -- so any two runs created
    within the same ~262ms window (routine for a ``@flow``'s own row plus
    its nested ``@agent``/``pipe``/``aggregate`` rows, or any quick burst of
    runs) would display an *identical* 8-char id, making them visually
    indistinguishable in ``compose runs``. 12 chars (60 bits: the full
    48-bit timestamp plus 12 random bits) keeps two such runs distinct with
    only a 1-in-4096 chance of collision, while staying short enough to
    read at a glance.
    """
    return run_id[:12]


def _fmt_cost(cost_usd: float | None, cost_complete: bool) -> str:
    if cost_usd is None:
        return "-"
    prefix = "" if cost_complete else "≥"
    return f"{prefix}${cost_usd:.4f}"


def _run_usage_totals_bulk(
    conn: sqlite3.Connection, run_ids: list[str]
) -> dict[str, tuple[int, float | None, bool]]:
    """``{run_id: (tokens, cost_usd, cost_complete)}`` for every id in ``run_ids``.

    One query over every requested run's llm spans (``run_id IN (...)``),
    grouped and summed in Python -- rather than ``cmd_runs`` calling a
    per-run usage query once per displayed row (an N+1 query pattern: one
    query to list the runs, then one more per row just to total its usage).
    A run with no llm spans (or not present in ``run_ids`` at all) is
    simply absent from the returned dict -- callers should default missing
    ids to ``(0, None, True)`` ("nothing spent").
    """
    if not run_ids:
        return {}
    placeholders = ",".join("?" * len(run_ids))
    rows = conn.execute(
        f"SELECT run_id, usage_json FROM spans WHERE kind = 'llm' AND run_id IN ({placeholders})",  # noqa: S608 -- placeholders only, values parameterized
        run_ids,
    ).fetchall()
    by_run: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    return {run_id: _sum_usage_rows(rs) for run_id, rs in by_run.items()}


def _sum_usage_rows(rows: list[sqlite3.Row]) -> tuple[int, float | None, bool]:
    tokens = 0
    cost = 0.0
    any_cost = False
    cost_complete = True
    for row in rows:
        usage = _decode_usage(row["usage_json"])
        tokens += usage.input_tokens + usage.output_tokens
        if usage.cost_usd is not None:
            cost += usage.cost_usd
            any_cost = True
        cost_complete = cost_complete and usage.cost_complete
    return tokens, (cost if any_cost else None), cost_complete


# --- rebuilding a Trace from spans + span_payloads ---------------------------


def _rebuild_trace(conn: sqlite3.Connection, trace_id: str) -> tracing.Trace:
    """Rebuild a :class:`~composeai.tracing.Trace` from the ``spans``/``span_payloads`` tables.

    Ordered by ``started_at, span_id`` (insertion order into the in-memory
    ``Trace.spans`` list doesn't matter for ``render_trace``, which walks
    parent/child links -- but a stable, reproducible order matters to
    ``compose diff``'s ``_span_path``, which numbers same-``(kind, name)``
    siblings by their position in this list: two spans with equal
    ``started_at`` -- routine for very fast parallel task/tool calls, or
    just ``time.time()``'s finite resolution -- would otherwise sort in
    whatever order SQLite happens to return them, which isn't guaranteed
    stable across queries/processes, silently pairing up the wrong
    siblings when diffing two runs. ``span_id`` (a ULID -- ``composeai.
    _ids``) is itself time-ordered and unique, so it's a deterministic
    tiebreaker, not an arbitrary one.
    """
    trace = tracing.Trace(trace_id=trace_id)
    rows = conn.execute(
        "SELECT s.*, p.input_json, p.output_json FROM spans s "
        "LEFT JOIN span_payloads p ON p.span_id = s.span_id "
        "WHERE s.trace_id = ? ORDER BY s.started_at, s.span_id",
        (trace_id,),
    ).fetchall()
    for row in rows:
        error = None
        if row["error_json"]:
            error_data = json.loads(row["error_json"])
            error = tracing.ErrorInfo(**error_data)
        span = tracing.Span(
            trace_id=row["trace_id"],
            span_id=row["span_id"],
            parent_span_id=row["parent_span_id"],
            kind=row["kind"],
            name=row["name"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            status=row["status"],
            error=error,
            input=from_jsonable(json.loads(row["input_json"])) if row["input_json"] else None,
            output=from_jsonable(json.loads(row["output_json"])) if row["output_json"] else None,
            usage=_decode_usage(row["usage_json"]) if row["usage_json"] else None,
            attributes=(
                from_jsonable(json.loads(row["attributes_json"]))
                if row["attributes_json"]
                else {}
            ),
            replayed=bool(row["replayed"]),
        )
        trace.add(span)
    return trace


def _get_run_row(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row is not None else None


def _get_last_run_row(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
    return dict(row) if row is not None else None


def _resolve_run_row(conn: sqlite3.Connection, id_or_prefix: str) -> dict[str, Any] | None:
    """Resolve a run id -- full, or any unique prefix of one -- to its ``runs`` row.

    ``compose runs``' text output prints the full run id, but a user
    copying just the leading handful of characters (or the whole thing)
    should still be able to paste it into ``trace``/``diff``/``export``.
    An exact match always wins outright, even if it also happens to be a
    literal prefix of some other run's id. Otherwise this matches every run
    id sharing ``id_or_prefix`` as a prefix: zero matches returns ``None``
    (callers report "no run found"); exactly one match resolves to it;
    more than one raises :class:`~composeai.errors.ComposeError` listing
    every match, since silently picking one would be a data-loss footgun
    (e.g. accidentally ``diff``-ing or ``export``-ing the wrong run).

    Uses ``substr(run_id, 1, ?) = ?`` rather than ``LIKE ... || '%'`` so an
    id/prefix is matched by exact character comparison, not SQL ``LIKE``
    pattern syntax (which would treat a literal ``%``/``_`` in the input as
    a wildcard rather than a character to match -- moot for a real ULID's
    Crockford-Base32 alphabet, but this way there's no wildcard-injection
    surface to think about at all).
    """
    if not id_or_prefix:
        return None
    exact = _get_run_row(conn, id_or_prefix)
    if exact is not None:
        return exact
    matches = conn.execute(
        "SELECT * FROM runs WHERE substr(run_id, 1, ?) = ?",
        (len(id_or_prefix), id_or_prefix),
    ).fetchall()
    if not matches:
        return None
    if len(matches) > 1:
        ids = sorted(row["run_id"] for row in matches)
        raise ComposeError(
            f"run id prefix {id_or_prefix!r} is ambiguous -- it matches "
            f"{len(ids)} runs: {', '.join(ids)} -- use a longer prefix or "
            "the full id"
        )
    return dict(matches[0])


# --- compose runs ------------------------------------------------------------


def cmd_runs(args: argparse.Namespace) -> int:
    # Validate --since up front (raises ComposeError -> caught by main() and
    # reported with exit 1) even when there's no store yet to query --
    # a bad flag should never be silently swallowed by an empty result.
    since_ts = _parse_since(args.since) if args.since else None

    db_path = _db_path()
    if not db_path.exists():
        _print_runs([], args)
        return 0

    conn = _open_ro(db_path)
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if args.status:
            clauses.append("status = ?")
            params.append(args.status)
        if args.kind:
            clauses.append("kind = ?")
            params.append(args.kind)
        if since_ts is not None:
            clauses.append("created_at >= ?")
            params.append(since_ts)

        if args.query:
            if not _fts_available(conn):
                print(
                    "compose: full-text search unavailable (FTS5 not compiled into "
                    "this sqlite3 build) -- ignoring -q",
                    file=sys.stderr,
                )
            else:
                try:
                    match_rows = conn.execute(
                        "SELECT DISTINCT spans.run_id FROM span_fts "
                        "JOIN spans ON spans.span_id = span_fts.span_id "
                        "WHERE span_fts MATCH ? AND spans.run_id IS NOT NULL",
                        (args.query,),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    # FTS5's MATCH argument has its own query grammar (quotes,
                    # NEAR(), column filters, -/^ prefixes, boolean operators)
                    # independent of SQL syntax -- a malformed `-q` value (e.g.
                    # an unterminated quote) raises this, uncaught, as a raw
                    # traceback if not turned into a ComposeError here (`main()`
                    # only catches ComposeError).
                    raise ComposeError(
                        f"invalid -q/--query syntax {args.query!r}: {exc}"
                    ) from exc
                run_id_filter = sorted({row["run_id"] for row in match_rows})
                if not run_id_filter:
                    _print_runs([], args)
                    return 0
                clauses.append(
                    "run_id IN (" + ",".join("?" * len(run_id_filter)) + ")"
                )
                params.extend(run_id_filter)

        query = "SELECT * FROM runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(args.limit)
        rows = [dict(row) for row in conn.execute(query, params).fetchall()]

        totals = _run_usage_totals_bulk(conn, [row["run_id"] for row in rows])
        enriched = []
        for row in rows:
            tokens, cost, cost_complete = totals.get(row["run_id"], (0, None, True))
            enriched.append(
                {**row, "tokens": tokens, "cost_usd": cost, "cost_complete": cost_complete}
            )
    finally:
        conn.close()

    _print_runs(enriched, args)
    return 0


def _print_runs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.json_output:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print("no runs found")
        return
    now = time.time()
    for row in rows:
        age = _relative_age(row["created_at"], now=now)
        cost_str = _fmt_cost(row["cost_usd"], row["cost_complete"])
        print(
            f"{row['run_id']:<26}  {row['kind']:<9} {row['name']:<20} {row['status']:<10} "
            f"{age:<9} {cost_str:>10} {row['tokens']:>8} tok"
        )


# --- compose trace -----------------------------------------------------------


def cmd_trace(args: argparse.Namespace) -> int:
    # Check the arguments themselves before ever touching the store -- an
    # empty/missing store must not mask "you forgot to pass a run id".
    if not args.last and not args.run_id:
        print("compose trace: provide a run id or --last", file=sys.stderr)
        return 1

    db_path = _db_path()
    if not db_path.exists():
        if args.last:
            print("compose: no runs found", file=sys.stderr)
        else:
            print(f"compose: no run found with id {args.run_id!r}", file=sys.stderr)
        return 1

    conn = _open_ro(db_path)
    try:
        if args.last:
            row = _get_last_run_row(conn)
            if row is None:
                print("compose: no runs found", file=sys.stderr)
                return 1
        else:
            row = _resolve_run_row(conn, args.run_id)
            if row is None:
                print(f"compose: no run found with id {args.run_id!r}", file=sys.stderr)
                return 1

        trace = _rebuild_trace(conn, row["trace_id"])
        print(tracing.render_trace(trace))

        if row["status"] == "paused":
            pending = conn.execute(
                "SELECT * FROM pending_interrupts WHERE run_id = ?", (row["run_id"],)
            ).fetchall()
            _print_paused_banner(row["run_id"], pending)
    finally:
        conn.close()
    return 0


def _print_paused_banner(run_id: str, pending: list[sqlite3.Row]) -> None:
    print()
    print(f"⏸  run {run_id} is paused with {len(pending)} pending interrupt(s):")
    for p in pending:
        payload = json.loads(p["payload_json"]) if p["payload_json"] else None
        print(f"  - id={p['interrupt_id']!r} kind={p['kind']!r} question={p['question']!r}")
        print(f"    payload={payload!r}")
    print()
    print("To resume:")
    print("    from composeai import resume")
    answers = {}
    for p in pending:
        answers[p["interrupt_id"]] = True if p["kind"] == "approval" else "<answer>"
    print(f"    resume({run_id!r}, answers={answers!r})")


# --- compose diff -------------------------------------------------------------


def _span_path(trace: tracing.Trace, span: tracing.Span) -> tuple[str, ...]:
    """The path from root to ``span``: ``"kind:name#i"`` per level, ``i`` the
    index of ``span`` among its siblings sharing the same ``(kind, name)``.
    """
    components: list[str] = []
    current: tracing.Span | None = span
    while current is not None:
        siblings = (
            trace.children_of(current.parent_span_id)
            if current.parent_span_id is not None
            else trace.roots()
        )
        same = [s for s in siblings if s.kind == current.kind and s.name == current.name]
        # Identity, not `==` -- `Span` is a plain (non-frozen) dataclass with
        # an auto-generated structural `__eq__` over every field, including
        # arbitrary `input`/`output` payloads; `current` is always literally
        # one of `trace`'s own span objects, so `is` is both correct and
        # cheaper (and can't ever raise from a payload whose `==` is weird).
        index = next(i for i, s in enumerate(same) if s is current)
        components.append(f"{current.kind}:{current.name}#{index}")
        if current.parent_span_id is None:
            current = None
        else:
            current = next((s for s in trace.spans if s.span_id == current.parent_span_id), None)
    return tuple(reversed(components))


def _effective_usage(trace: tracing.Trace, span: tracing.Span) -> Usage:
    if span.kind == "llm" and span.usage is not None:
        return span.usage
    return trace.rollup_usage(span)


def _hash_output(value: Any) -> str:
    try:
        payload = to_jsonable(value)
    except Exception:  # noqa: BLE001 -- diffing must never crash on odd payloads
        payload = repr(value)
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt_signed_tokens(n: int) -> str:
    return f"{'+' if n >= 0 else ''}{n}"


def _fmt_signed_ms(ms: float) -> str:
    return f"{'+' if ms >= 0 else ''}{ms:.0f}ms"


def _fmt_signed_cost(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{'+' if delta >= 0 else ''}${delta:.4f}"


def cmd_diff(args: argparse.Namespace) -> int:
    db_path = _db_path()
    if not db_path.exists():
        print("compose: no runs found", file=sys.stderr)
        return 1

    conn = _open_ro(db_path)
    try:
        row_a = _resolve_run_row(conn, args.run_a)
        if row_a is None:
            print(f"compose: no run found with id {args.run_a!r}", file=sys.stderr)
            return 1
        row_b = _resolve_run_row(conn, args.run_b)
        if row_b is None:
            print(f"compose: no run found with id {args.run_b!r}", file=sys.stderr)
            return 1

        trace_a = _rebuild_trace(conn, row_a["trace_id"])
        trace_b = _rebuild_trace(conn, row_b["trace_id"])
    finally:
        conn.close()

    _print_diff(row_a["run_id"], row_b["run_id"], trace_a, trace_b)
    return 0


def _print_diff(run_a: str, run_b: str, trace_a: tracing.Trace, trace_b: tracing.Trace) -> None:
    paths_a = {_span_path(trace_a, s): s for s in trace_a.spans}
    paths_b = {_span_path(trace_b, s): s for s in trace_b.spans}

    usage_a = trace_a.total_usage()
    usage_b = trace_b.total_usage()
    duration_a = tracing._root_duration_ms(trace_a) or 0.0
    duration_b = tracing._root_duration_ms(trace_b) or 0.0
    delta_tokens = (usage_b.input_tokens + usage_b.output_tokens) - (
        usage_a.input_tokens + usage_a.output_tokens
    )
    delta_cost: float | None
    if usage_a.cost_usd is None and usage_b.cost_usd is None:
        delta_cost = None
    else:
        delta_cost = (usage_b.cost_usd or 0.0) - (usage_a.cost_usd or 0.0)

    print(f"diff {_short_id(run_a)} -> {_short_id(run_b)}")
    print(f"  Δcost: {_fmt_signed_cost(delta_cost)}")
    print(f"  Δtokens: {_fmt_signed_tokens(delta_tokens)}")
    print(f"  Δduration: {_fmt_signed_ms(duration_b - duration_a)}")
    print()

    for path in sorted(set(paths_a) | set(paths_b)):
        depth = len(path)
        indent = "  " * (depth - 1)
        in_a = path in paths_a
        in_b = path in paths_b
        if in_a and in_b:
            span_a = paths_a[path]
            span_b = paths_b[path]
            glyph = tracing._GLYPHS.get(span_b.kind, "?")
            ua = _effective_usage(trace_a, span_a)
            ub = _effective_usage(trace_b, span_b)
            dtok = (ub.input_tokens + ub.output_tokens) - (ua.input_tokens + ua.output_tokens)
            dcost: float | None
            if ua.cost_usd is None and ub.cost_usd is None:
                dcost = None
            else:
                dcost = (ub.cost_usd or 0.0) - (ua.cost_usd or 0.0)
            ddur = (span_b.duration_ms or 0.0) - (span_a.duration_ms or 0.0)
            changed = _hash_output(span_a.output) != _hash_output(span_b.output)
            line = (
                f"{indent}  {glyph} {span_b.name}  "
                f"Δcost={_fmt_signed_cost(dcost)} Δtokens={_fmt_signed_tokens(dtok)} "
                f"Δdur={_fmt_signed_ms(ddur)}"
            )
            if changed:
                line += "  (output changed)"
            print(line)
        elif in_a:
            span_a = paths_a[path]
            glyph = tracing._GLYPHS.get(span_a.kind, "?")
            print(f"{indent}- {glyph} {span_a.name}")
        else:
            span_b = paths_b[path]
            glyph = tracing._GLYPHS.get(span_b.kind, "?")
            print(f"{indent}+ {glyph} {span_b.name}")


# --- compose costs ------------------------------------------------------------


def cmd_costs(args: argparse.Namespace) -> int:
    # Same reasoning as cmd_runs: validate --since up front, even when
    # there's no store yet -- a bad flag must never be masked by an empty
    # "no llm spans found" result.
    since_ts = _parse_since(args.since) if args.since else None

    db_path = _db_path()
    if not db_path.exists():
        _print_costs([], args.by)
        return 0

    conn = _open_ro(db_path)
    try:
        clauses = ["kind = 'llm'"]
        params: list[Any] = []
        if since_ts is not None:
            clauses.append("started_at >= ?")
            params.append(since_ts)
        query = "SELECT usage_json, attributes_json, run_id, name, started_at FROM spans"
        query += " WHERE " + " AND ".join(clauses)
        rows = conn.execute(query, params).fetchall()

        buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "any_cost": False, "cost_complete": True}
        )
        run_name_cache: dict[str, str] = {}
        for row in rows:
            key = _cost_bucket_key(conn, row, args.by, run_name_cache)
            bucket = buckets[key]
            bucket["calls"] += 1
            usage = _decode_usage(row["usage_json"])
            bucket["tokens"] += usage.input_tokens + usage.output_tokens
            if usage.cost_usd is not None:
                bucket["cost"] += usage.cost_usd
                bucket["any_cost"] = True
            bucket["cost_complete"] = bucket["cost_complete"] and usage.cost_complete
    finally:
        conn.close()

    _print_costs(
        sorted(
            (
                {
                    "key": key,
                    "calls": b["calls"],
                    "tokens": b["tokens"],
                    "cost_usd": b["cost"] if b["any_cost"] else None,
                    "cost_complete": b["cost_complete"],
                }
                for key, b in buckets.items()
            ),
            key=lambda r: r["key"],
        ),
        args.by,
    )
    return 0


def _cost_bucket_key(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    by: str,
    run_name_cache: dict[str, str],
) -> str:
    if by == "model":
        attrs = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
        return attrs.get("model") or row["name"]
    if by == "day":
        return datetime.fromtimestamp(row["started_at"]).strftime("%Y-%m-%d")
    # by == "name": group by the owning run's name.
    run_id = row["run_id"]
    if run_id is None:
        return "(none)"
    if run_id not in run_name_cache:
        run_row = conn.execute("SELECT name FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        run_name_cache[run_id] = run_row["name"] if run_row is not None else "(unknown)"
    return run_name_cache[run_id]


def _print_costs(rows: list[dict[str, Any]], by: str) -> None:
    print(f"costs by {by}")
    if not rows:
        print("no llm spans found")
        return
    total_calls = 0
    total_tokens = 0
    total_cost = 0.0
    any_cost = False
    cost_complete = True
    for row in rows:
        cost_str = _fmt_cost(row["cost_usd"], row["cost_complete"])
        print(
            f"  {row['key']:<24} calls={row['calls']:<6} "
            f"tokens={row['tokens']:<10} cost={cost_str}"
        )
        total_calls += row["calls"]
        total_tokens += row["tokens"]
        if row["cost_usd"] is not None:
            total_cost += row["cost_usd"]
            any_cost = True
        cost_complete = cost_complete and row["cost_complete"]
    total_cost_str = _fmt_cost(total_cost if any_cost else None, cost_complete)
    print(f"  {'TOTAL':<24} calls={total_calls:<6} tokens={total_tokens:<10} cost={total_cost_str}")


# --- compose path --------------------------------------------------------------


def cmd_path(args: argparse.Namespace) -> int:
    print(str(_compose_dir()))
    return 0


# --- compose export ------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    db_path = _db_path()
    if not db_path.exists():
        print(f"compose: no run found with id {args.run_id!r}", file=sys.stderr)
        return 1

    conn = _open_ro(db_path)
    try:
        row = _resolve_run_row(conn, args.run_id)
        if row is None:
            print(f"compose: no run found with id {args.run_id!r}", file=sys.stderr)
            return 1

        llm_rows = conn.execute(
            "SELECT s.*, p.input_json, p.output_json FROM spans s "
            "LEFT JOIN span_payloads p ON p.span_id = s.span_id "
            "WHERE s.run_id = ? AND s.kind = 'llm' ORDER BY s.started_at, s.span_id",
            (row["run_id"],),
        ).fetchall()

        entries = [_export_entry(r) for r in llm_rows]
        degenerate = sum(1 for r in llm_rows if r["input_json"] is None)
    finally:
        conn.close()

    if degenerate:
        print(
            f"compose: warning -- {degenerate} of {len(entries)} exported entries came "
            "from spans persisted with content capture disabled (COMPOSE_TRACE_CONTENT=0): "
            "their cassette entries have no system/messages (system=None, messages=[]) and "
            "can only ever replay-match on that equally-empty message_hash -- re-capture "
            "this run with COMPOSE_TRACE_CONTENT unset/1 for a realistic cassette",
            file=sys.stderr,
        )

    cassette_path = Path(args.cassette)
    cassette_path.parent.mkdir(parents=True, exist_ok=True)
    cassette_payload = {"version": _CASSETTE_VERSION, "entries": entries}
    cassette_path.write_text(json.dumps(cassette_payload, indent=2))
    print(f"wrote {len(entries)} entries to {cassette_path}")
    return 0


def _export_entry(row: sqlite3.Row) -> dict[str, Any]:
    """Build one cassette entry from a persisted ``llm`` span.

    ``full_hash`` is always ``None`` here -- the persisted span input only
    ever stored ``{"system", "messages"}`` (see
    ``composeai.agentfn._call_llm``), never the tools/output_schema/
    max_tokens/temperature a live request also hashes -- so an exported
    cassette can only ever match on ``message_hash`` at replay time (see
    :class:`~composeai.testing.ReplayModel`'s fallback).

    ``stop_reason`` isn't persisted either (only ``response.message`` is --
    see the same function): inferred as ``TOOL_USE`` when the message
    contains a ``ToolCallPart``, else ``END_TURN``. ``parsed`` is left
    ``None``: a structured-output agent's ``_extract_output`` already falls
    back to ``json.loads(message.text)`` when ``parsed`` is unset, which is
    exactly what an exported cassette's message text is.
    """
    input_data = json.loads(row["input_json"]) if row["input_json"] else {}
    decoded_input = from_jsonable(input_data) if input_data else {}
    is_dict = isinstance(decoded_input, dict)
    system = decoded_input.get("system") if is_dict else None
    messages: list[Message] = decoded_input.get("messages", []) if is_dict else []

    output_data = json.loads(row["output_json"]) if row["output_json"] else None
    message: Message = (
        from_jsonable(output_data) if output_data is not None else Message.assistant("")
    )

    attrs = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
    model_id = attrs.get("model", row["name"])

    usage = _decode_usage(row["usage_json"])
    has_tool_call = any(getattr(part, "type", None) == "tool_call" for part in message.parts)
    stop_reason = StopReason.TOOL_USE if has_tool_call else StopReason.END_TURN

    response = ModelResponse(
        message=message,
        stop_reason=stop_reason,
        raw_stop_reason=stop_reason.value,
        usage=usage,
        model_id=model_id,
        parsed=None,
    )

    return {
        "full_hash": None,
        "message_hash": compute_message_hash(system, messages),
        "request": {
            "model": model_id,
            "system": system,
            "messages": [m.model_dump(mode="json") for m in messages],
        },
        "response": to_jsonable(response),
    }


# --- argparse wiring -----------------------------------------------------------


def _positive_int(value: str) -> int:
    """``argparse`` ``type=`` for ``-n/--limit``: a plain positive integer.

    A bare ``type=int`` accepts 0/negative values, which then hit
    ``rows[: args.limit]``'s Python slice semantics -- ``-5`` silently
    means "all rows except the last 5", not "unlimited"/an error, and
    disagrees with ``RunStore.list_runs``'s own ``LIMIT ?``, where SQLite
    treats a negative ``LIMIT`` as unlimited. Rejecting non-positive values
    here, loudly, at parse time (``argparse``'s own usage-error path, exit
    code 2) avoids that whole class of surprise.
    """
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from None
    if n <= 0:
        raise argparse.ArgumentTypeError(
            f"--limit must be a positive integer, got {value!r}"
        )
    return n


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compose", description="Inspect composeai's durable run store."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_runs = sub.add_parser("runs", help="List recent runs.")
    p_runs.add_argument("-n", "--limit", type=_positive_int, default=10)
    p_runs.add_argument("--json", action="store_true", dest="json_output")
    p_runs.add_argument("--status", choices=["completed", "failed", "paused", "running"])
    p_runs.add_argument("--kind", choices=["agent", "flow", "pipe", "aggregate"])
    p_runs.add_argument("--since")
    p_runs.add_argument("-q", "--query", dest="query")
    p_runs.set_defaults(func=cmd_runs)

    p_trace = sub.add_parser("trace", help="Render one run's trace.")
    p_trace.add_argument("run_id", nargs="?")
    p_trace.add_argument("--last", action="store_true")
    p_trace.set_defaults(func=cmd_trace)

    p_diff = sub.add_parser("diff", help="Structurally diff two runs' traces.")
    p_diff.add_argument("run_a")
    p_diff.add_argument("run_b")
    p_diff.set_defaults(func=cmd_diff)

    p_costs = sub.add_parser("costs", help="Group-by spend report over llm spans.")
    p_costs.add_argument("--since")
    p_costs.add_argument("--by", choices=["model", "name", "day"], default="model")
    p_costs.set_defaults(func=cmd_costs)

    p_path = sub.add_parser("path", help="Print the state directory.")
    p_path.set_defaults(func=cmd_path)

    p_export = sub.add_parser("export", help="Export a run's llm spans as a cassette.")
    p_export.add_argument("run_id")
    p_export.add_argument("--cassette", required=True)
    p_export.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ComposeError as exc:
        print(f"compose: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
