"""Always-on tracing: spans, trace trees, and a console renderer.

Every ``@agent``/``@task``/adapter call opens a :class:`Span` via
:func:`span`, which nests under whatever span is active on the current
contextvar stack and publishes ``span_started``/``span_finished`` events
on the ambient :class:`~composeai.events.EventBus` (see
``composeai.events`` -- streaming and tracing share that one bus). The
resulting :class:`Trace` is an in-memory tree that can be rendered to a
console (:func:`render_trace`) or exported to a JSON-safe dict
(:meth:`Trace.to_dict`) for persistence.
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, ParamSpec, TextIO, TypeVar

from . import events
from ._encoding import to_jsonable
from ._ids import new_ulid
from .errors import SerializationError
from .messages import Usage

SpanKind = Literal["flow", "task", "agent", "llm", "tool", "pause", "pipe", "aggregate"]
SpanStatus = Literal["running", "ok", "error", "paused"]
TraceStatus = Literal["ok", "error", "paused", "running"]

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class ErrorInfo:
    """A captured exception, flattened for storage/rendering."""

    type: str
    message: str
    stacktrace: str


@dataclass(kw_only=True)
class Span:
    """One node in a :class:`Trace`. Mutable -- updated in place as work runs."""

    trace_id: str
    span_id: str = field(default_factory=new_ulid)
    parent_span_id: str | None = None
    kind: SpanKind
    name: str
    started_at: float
    ended_at: float | None = None
    status: SpanStatus = "running"
    error: ErrorInfo | None = None
    input: Any = None
    output: Any = None
    usage: Usage | None = None
    """Set only on spans that directly consumed an LLM (adapters set it on
    ``llm`` spans; nothing else). Rollups are always computed from
    descendant ``llm`` spans -- never stored on parents -- so usage is
    never double-counted."""
    attributes: dict[str, Any] = field(default_factory=dict)
    replayed: bool = False

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000

    def set_input(self, value: Any) -> None:
        """Capture ``value`` as this span's input, honoring the content-capture gate."""
        self.input = value if content_capture_enabled() else None

    def set_output(self, value: Any) -> None:
        """Capture ``value`` as this span's output, honoring the content-capture gate."""
        self.output = value if content_capture_enabled() else None


def content_capture_enabled() -> bool:
    """Whether *trace spans* should capture ``input``/``output`` payloads.

    Reads the environment on every call (cheap, and monkeypatch-friendly
    for tests): disabled only when ``COMPOSE_TRACE_CONTENT`` is exactly
    ``"0"``. Usage/status/timing are always recorded regardless.

    Scope: this gates ``Span.input``/``Span.output`` only (via
    :meth:`Span.set_input`/:meth:`Span.set_output`) -- observability data
    that exists purely for ``compose trace``/``compose diff``/``-q``
    search. It does **not** extend to anything composeai needs as
    *functional* state to actually work: a paused agent's ``agent_state``
    snapshot (needed for ``resume()`` to continue the conversation), the
    ``@flow`` journal, and ``composeai.testing``'s cassette recording
    (:func:`~composeai.testing.record_cassette`) and
    ``@agent(cache=True)`` response cache
    (:class:`~composeai.testing.CachingModel`) are all written in full
    regardless of this flag -- each needs its real content to do its job
    (resume a conversation, replay a call, serve a cache hit) and has no
    degraded-but-functional mode. See the README's "Rules of the road"
    section for the full list.
    """
    return os.environ.get("COMPOSE_TRACE_CONTENT") != "0"


@dataclass
class Trace:
    """An in-memory tree of spans sharing one ``trace_id``.

    No synchronization of its own: ``spans`` is a plain, unlocked ``list``,
    even though ``aggregate()``/``map()``/parallel tool execution routinely
    call :meth:`add` (via :func:`span`) from several branches' threads at
    once, concurrently with :meth:`rollup_usage`/:meth:`children_of` reads
    from :func:`~composeai.runs.check_budgets` on yet another thread. This
    relies on CPython's GIL making ``list.append``/iteration individually
    atomic (true today, on every version this codebase targets) -- there is
    no stronger guarantee, unlike :class:`~composeai.runs.RunStore`, which
    explicitly documents its own thread-safety contract. A renderer/CLI/sink
    reading ``run.trace`` while a background branch is still appending spans
    could observe a partial (but never corrupted) snapshot, and this design
    would need real locking to stay correct on a free-threaded (no-GIL)
    CPython build.
    """

    trace_id: str
    spans: list[Span] = field(default_factory=list)

    def add(self, span: Span) -> None:
        self.spans.append(span)

    def roots(self) -> list[Span]:
        return [s for s in self.spans if s.parent_span_id is None]

    def children_of(self, span_id: str) -> list[Span]:
        return [s for s in self.spans if s.parent_span_id == span_id]

    def _subtree(self, span: Span) -> list[Span]:
        result = [span]
        for child in self.children_of(span.span_id):
            result.extend(self._subtree(child))
        return result

    def rollup_usage(self, span: Span) -> Usage:
        """Sum of ``usage`` over ``span``'s subtree (including itself)."""
        total = Usage()
        for s in self._subtree(span):
            if s.usage is not None:
                total = total + s.usage
        return total

    def total_usage(self) -> Usage:
        """Sum of ``usage`` over every span in the trace."""
        total = Usage()
        for s in self.spans:
            if s.usage is not None:
                total = total + s.usage
        return total

    @property
    def status(self) -> TraceStatus:
        roots = self.roots()
        if any(r.status == "error" for r in roots):
            return "error"
        if any(s.status == "paused" for s in self.spans):
            return "paused"
        if any(r.status == "running" for r in roots):
            return "running"
        return "ok"

    def print(self, *, file: TextIO | None = None, color: bool | None = None) -> None:
        target = file if file is not None else sys.stdout
        target.write(render_trace(self, color=color) + "\n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": [_span_to_dict(s) for s in self.spans],
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _error_to_dict(error: ErrorInfo | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {"type": error.type, "message": error.message, "stacktrace": error.stacktrace}


def _safe_jsonable(value: Any) -> Any:
    # Tracing is observability: exporting a trace must never crash the
    # program because some payload isn't encodable. (The journal encoder
    # stays strict; this fallback is for trace export only.)
    try:
        return to_jsonable(value)
    except SerializationError:
        return repr(value)


def _span_to_dict(span: Span) -> dict[str, Any]:
    input_value, input_truncated = truncate_payload(_safe_jsonable(span.input))
    output_value, output_truncated = truncate_payload(_safe_jsonable(span.output))

    attributes = dict(span.attributes)
    if span.usage is not None:
        attributes["gen_ai.usage.input_tokens"] = span.usage.input_tokens
        attributes["gen_ai.usage.output_tokens"] = span.usage.output_tokens
        attributes["gen_ai.usage.cache_read.input_tokens"] = span.usage.cache_read_tokens
        attributes["gen_ai.usage.cache_creation.input_tokens"] = span.usage.cache_creation_tokens
        if "provider" in span.attributes:
            attributes["gen_ai.provider.name"] = span.attributes["provider"]
        if "model" in span.attributes:
            attributes["gen_ai.request.model"] = span.attributes["model"]

    result: dict[str, Any] = {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "kind": span.kind,
        "name": span.name,
        "started_at": span.started_at,
        "ended_at": span.ended_at,
        "status": span.status,
        "error": _error_to_dict(span.error),
        "replayed": span.replayed,
        "input": input_value,
        "output": output_value,
        "attributes": attributes,
        "cost_usd": span.usage.cost_usd if span.usage is not None else None,
    }
    if input_truncated or output_truncated:
        result["_truncated"] = True
    return result


def truncate_payload(value: Any, limit: int = 8192) -> tuple[Any, bool]:
    """Recursively truncate long strings inside ``value``.

    Any ``str`` longer than ``limit`` becomes ``head + "…[N chars
    truncated]…" + tail``, with ``head``/``tail`` each ``limit // 2``
    characters. Dicts and lists are walked recursively; non-string
    scalars pass through unchanged. Returns ``(new_value, was_truncated)``.
    """
    truncated = False

    def _walk(v: Any) -> Any:
        nonlocal truncated
        if isinstance(v, str):
            if len(v) > limit:
                truncated = True
                half = limit // 2
                head = v[:half]
                tail = v[-half:] if half > 0 else ""
                dropped = len(v) - len(head) - len(tail)
                return f"{head}…[{dropped} chars truncated]…{tail}"
            return v
        if isinstance(v, dict):
            return {k: _walk(item) for k, item in v.items()}
        if isinstance(v, list):
            return [_walk(item) for item in v]
        return v

    return _walk(value), truncated


# --- Collector ---

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_current_span", default=None
)
_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "_current_trace", default=None
)


def current_span() -> Span | None:
    return _current_span.get()


def current_trace() -> Trace | None:
    return _current_trace.get()


@contextmanager
def use_trace(trace_id: str) -> Iterator[Trace]:
    """Force a fresh :class:`Trace` with a specific ``trace_id`` as ambient for the block.

    Used by flow resume (:mod:`composeai.flow`) so spans opened while
    re-executing a flow's body join the *original* run's trace instead of
    starting a new one -- the persisted ``spans`` rows for a resumed run
    then all share one ``trace_id`` with the spans from before the crash.
    Only meaningful when called with no trace already active (nesting
    inside an existing ``span()`` block is not a supported use).
    """
    trace = Trace(trace_id=trace_id)
    token = _current_trace.set(trace)
    try:
        yield trace
    finally:
        _current_trace.reset(token)


# --- Span persistence sink (Phase 7: durable flows) -----------------------
#
# A module-global hook (deliberately NOT a contextvar: persistence must
# observe every span regardless of which thread/context produced it, with
# no per-context opt-in/out). ``composeai.runs`` installs this lazily the
# first time any `.run()` executes (agent or flow) -- this module must
# never import `runs` itself (the dependency runs the other way: the sink
# is injected here, not looked up there) to avoid a tracing<->runs import
# cycle (tracing is a foundational module; runs is a consumer of it).

_span_sink: Callable[[Span], None] | None = None
_span_sink_warned = False


def set_span_sink(fn: Callable[[Span], None] | None) -> None:
    """Install (``fn``) or remove (``None``) the module-global span-persistence sink.

    Called with every finished :class:`Span` from ``span()``'s ``finally``
    block, after the span's terminal ``status``/``ended_at`` are set.
    Exceptions raised by ``fn`` itself are caught here -- persistence
    failures must never crash user code -- with one ``RuntimeWarning``
    emitted on the first failure per process and silence after that (see
    :func:`reset_span_sink_state` for the test-only way to clear that
    latch). That only covers a sink that fails *synchronously*, though:
    ``composeai.runs``'s default sink (installed by
    :func:`~composeai.runs.open_default`) never raises here at all -- it
    hands the write off to the store's writer thread
    (``composeai._storeasync.StoreWorker.cast``) and returns immediately,
    so a genuine persistence failure surfaces from the store writer thread
    on the first failed span persist instead (one ``RuntimeWarning``,
    then silence -- see ``_storeasync._warn_once_on_cast_failure``), not
    from here.
    """
    global _span_sink
    _span_sink = fn


def reset_span_sink_state() -> None:
    """Test-only: clear the "already warned about a sink failure" latch."""
    global _span_sink_warned
    _span_sink_warned = False


def _notify_span_sink(finished_span: Span) -> None:
    global _span_sink_warned
    sink = _span_sink
    if sink is None:
        return
    try:
        sink(finished_span)
    except Exception as exc:  # persistence must never crash user code
        if not _span_sink_warned:
            _span_sink_warned = True
            warnings.warn(
                f"composeai: span persistence failed ({type(exc).__name__}: {exc}); "
                "continuing without persisting this (and silently any further) span "
                "until the process restarts",
                RuntimeWarning,
                stacklevel=2,
            )


@contextmanager
def span(
    kind: SpanKind,
    name: str,
    *,
    input: Any = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """Open a new :class:`Span`, nesting under the current span if any.

    On normal exit the span is marked ``"ok"``. On exception it is marked
    ``"error"`` with a populated :class:`ErrorInfo`, and the exception is
    re-raised unchanged (same object, no wrapping/chaining) so the user's
    traceback stays pristine.
    """
    trace = _current_trace.get()
    parent = _current_span.get()

    trace_token: contextvars.Token[Trace | None] | None = None
    if trace is None:
        trace = Trace(trace_id=new_ulid())
        trace_token = _current_trace.set(trace)

    new_span = Span(
        trace_id=trace.trace_id,
        parent_span_id=parent.span_id if parent is not None else None,
        kind=kind,
        name=name,
        started_at=time.time(),
        input=input if content_capture_enabled() else None,
        attributes=dict(attributes) if attributes is not None else {},
    )
    trace.add(new_span)
    span_token = _current_span.set(new_span)

    events.emit(
        events.Event(
            kind="span_started",
            trace_id=new_span.trace_id,
            span_id=new_span.span_id,
            name=new_span.name,
            data={"kind": new_span.kind},
        )
    )
    try:
        yield new_span
    except BaseException as exc:
        # Phase 8 (human-in-the-loop): a pause is control flow, not a failure --
        # `composeai.hitl._Pause` (and anything else tagged `_compose_pause`)
        # marks the span "paused" with no `ErrorInfo`, rather than "error".
        # Duck-typed rather than importing `composeai.hitl` (which imports this
        # module for `span()` itself -- an import the other way would cycle).
        if getattr(exc, "_compose_pause", False):
            new_span.status = "paused"
        else:
            new_span.status = "error"
            new_span.error = ErrorInfo(
                type=type(exc).__name__,
                message=str(exc),
                stacktrace=traceback.format_exc(),
            )
        raise
    else:
        new_span.status = "ok"
    finally:
        new_span.ended_at = time.time()
        events.emit(
            events.Event(
                kind="span_finished",
                trace_id=new_span.trace_id,
                span_id=new_span.span_id,
                name=new_span.name,
                data={"kind": new_span.kind, "status": new_span.status},
            )
        )
        _notify_span_sink(new_span)
        _current_span.reset(span_token)
        if trace_token is not None:
            _current_trace.reset(trace_token)


def emit_run_finished(run_span: Span, *, status: str, error_type: str | None = None) -> None:
    """Publish the terminal ``run_finished`` event for a finished run — root runs only.

    A trace has exactly one ``run_finished``: the outermost run's. Nested
    runs (an agent inside a pipe/flow) stay silent here — their completion
    is already signaled by their span's ``span_finished`` — so bus consumers
    can safely treat ``run_finished`` as "the whole thing is done".
    """
    if run_span.parent_span_id is not None:
        return
    data: dict[str, Any] = {"status": status}
    if error_type is not None:
        data["error"] = error_type
    events.emit(
        events.Event(
            kind="run_finished",
            trace_id=run_span.trace_id,
            span_id=run_span.span_id,
            data=data,
        )
    )


def propagate(fn: Callable[P, R]) -> Callable[P, R]:
    """Wrap ``fn`` so it runs with the current contextvars (span/bus/etc).

    ``copy_context()`` is called at wrap time, in the parent thread, so
    later phases can hand the result to a worker thread and have it see
    the same active span/bus that was current when ``propagate`` was
    called.
    """
    ctx = contextvars.copy_context()

    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        return ctx.run(fn, *args, **kwargs)

    return _wrapped


# --- Console renderer ---

_GLYPHS: dict[str, str] = {
    "flow": "▶",
    "task": "•",
    "agent": "◆",
    "llm": "▸",
    "tool": "⚙",
    "pause": "⏸",
    "pipe": "→",
    "aggregate": "⇉",
}


def _resolve_color(color: bool | None) -> bool:
    if color is not None:
        return color
    # NO_COLOR (https://no-color.org/): color must be disabled whenever the
    # variable is *present*, "regardless of its value" -- including an empty
    # string, which `not os.environ.get("NO_COLOR")` would otherwise treat
    # identically to the variable being unset entirely (`"" ` is falsy, same
    # as `None`). `"NO_COLOR" in os.environ` is the presence check the spec
    # actually asks for.
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _colorize(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _humanize_tokens(n: int) -> str:
    if n < 1000:
        return f"{n} tok"
    if n < 1_000_000:
        return f"{n / 1000:.1f}k tok"
    return f"{n / 1_000_000:.1f}M tok"


def _humanize_duration(ms: float) -> str:
    if ms < 1000:
        return f"{round(ms)}ms"
    return f"{ms / 1000:.1f}s"


def _format_cost(usage: Usage) -> str | None:
    if usage.cost_usd is None:
        return None
    prefix = "≥" if not usage.cost_complete else ""
    return f"{prefix}${usage.cost_usd:.4f}"


def _usage_bracket(usage: Usage | None, duration_ms: float | None) -> str:
    segments: list[str] = []
    if usage is not None:
        cost = _format_cost(usage)
        if cost is not None:
            segments.append(cost)
        if usage.input_tokens or usage.output_tokens:
            segments.append(_humanize_tokens(usage.input_tokens + usage.output_tokens))
    if duration_ms is not None:
        segments.append(_humanize_duration(duration_ms))
    if not segments:
        return ""
    return "[" + " · ".join(segments) + "]"


def _root_duration_ms(trace: Trace) -> float | None:
    roots = trace.roots()
    if not roots:
        return None
    if any(r.ended_at is None for r in roots):
        return None
    started = min(r.started_at for r in roots)
    ended = max(r.ended_at for r in roots if r.ended_at is not None)
    return (ended - started) * 1000


def _header_line(trace: Trace) -> str:
    usage = trace.total_usage()
    duration_ms = _root_duration_ms(trace)
    bracket = _usage_bracket(usage, duration_ms)
    header = f"trace {trace.trace_id} — {trace.status}"
    if bracket:
        header += f" — {bracket}"
    return header


def _status_decoration(node: Span, use_color: bool) -> str:
    decoration = ""
    if node.status == "error" and node.error is not None:
        first_line = node.error.message.splitlines()[0] if node.error.message else ""
        text = f" ✗ {node.error.type}: {first_line}"
        decoration += _colorize(text, "31", use_color)
    elif node.status == "paused":
        decoration += _colorize(" ⏸ paused", "33", use_color)
    if node.replayed:
        decoration += _colorize(" (replayed)", "2", use_color)
    return decoration


def _node_line(trace: Trace, node: Span, use_color: bool) -> str:
    glyph = _GLYPHS[node.kind]
    usage = node.usage if node.kind == "llm" else trace.rollup_usage(node)
    bracket = _usage_bracket(usage, node.duration_ms)
    text = f"{glyph} {node.name}"
    if bracket:
        text += f" {bracket}"
    text += _status_decoration(node, use_color)
    return text


def _render_node(
    trace: Trace,
    node: Span,
    prefix: str,
    is_last: bool,
    lines: list[str],
    use_color: bool,
) -> None:
    connector = "└─ " if is_last else "├─ "
    lines.append(prefix + connector + _node_line(trace, node, use_color))
    children = trace.children_of(node.span_id)
    child_prefix = prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(children):
        _render_node(trace, child, child_prefix, i == len(children) - 1, lines, use_color)


def render_trace(trace: Trace, *, color: bool | None = None) -> str:
    """Render ``trace`` as a console tree.

    With ``color=False`` the output is byte-stable (no ANSI escapes,
    suitable for snapshot tests). ``color=None`` auto-detects from
    ``sys.stdout.isatty()`` and the ``NO_COLOR`` environment variable.
    """
    use_color = _resolve_color(color)
    lines = [_header_line(trace)]
    roots = trace.roots()
    for i, root in enumerate(roots):
        _render_node(trace, root, "", i == len(roots) - 1, lines, use_color)
    return "\n".join(lines)


# --- Mermaid renderer ---


def _mermaid_label(node: Span) -> str:
    # Mermaid's renderer decodes ANY `#\w+;`-shaped substring in a label as
    # an HTML entity -- not just the escapes we ourselves emit -- so a
    # user-settable span name containing e.g. literal text "#65;" would
    # otherwise round-trip through Mermaid's parser as the character "A".
    # Escaping "#" to "#35;" FIRST neutralizes any such entity already
    # present in the name; only then do we escape '"' to Mermaid's own
    # quote-escape "#quot;". Order matters: doing it the other way around
    # would leave a user's raw "#65;" as a live entity, and would also let
    # the "#" our own quote-escape step just introduced get re-escaped by
    # a later hash pass. Escaping "#" first means "#quot;"/"#35;" are the
    # only entities present in the final label.
    name = node.name.replace("#", "#35;").replace('"', "#quot;")
    return f"{name} [{node.kind}]"


def _walk_mermaid(
    trace: Trace,
    node: Span,
    parent_id: str | None,
    ids: dict[str, str],
    node_lines: list[str],
    edge_lines: list[str],
) -> None:
    node_id = f"s{len(ids)}"
    ids[node.span_id] = node_id
    node_lines.append(f'{node_id}["{_mermaid_label(node)}"]')
    if parent_id is not None:
        edge_lines.append(f"{parent_id} --> {node_id}")
    for child in trace.children_of(node.span_id):
        _walk_mermaid(trace, child, node_id, ids, node_lines, edge_lines)


def render_trace_mermaid(trace: Trace) -> str:
    """Render ``trace`` as a Mermaid ``flowchart TD`` document.

    Walks the same ``trace.roots()``/``children_of()`` structure
    :func:`render_trace`/``_render_node`` does -- one depth-first,
    roots-first pass -- assigning each span a node id ``s0``, ``s1``, ...
    in that same walk order. Each span becomes one node line
    (``sN["name [kind]"]``, with any ``#`` in ``name`` escaped to ``#35;``
    *before* any ``"`` is escaped to Mermaid's own ``#quot;`` -- see
    :func:`_mermaid_label` for why the order matters: Mermaid decodes any
    ``#\\w+;``-shaped substring as an HTML entity, so escaping ``#`` first
    is what keeps ``#quot;``/``#35;`` the only entities present in the
    label) followed, after all node lines, by one ``parent --> child`` edge
    line per parent/child pair, in the order children were discovered.

    Purely a renderer over the in-memory :class:`Trace` tree: reads only
    ``Span.kind``/``name``/parent-child links already loaded (by the CLI,
    from persisted rows) -- never imports or executes anything from the
    run's own payloads.
    """
    ids: dict[str, str] = {}
    node_lines: list[str] = []
    edge_lines: list[str] = []
    for root in trace.roots():
        _walk_mermaid(trace, root, None, ids, node_lines, edge_lines)
    return "\n".join(["flowchart TD", *node_lines, *edge_lines])
