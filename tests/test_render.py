import sys

from composeai.messages import Usage
from composeai.tracing import ErrorInfo, Span, Trace, render_trace


def _build_trace() -> tuple[Trace, Span]:
    trace = Trace(trace_id="TRACE01")
    root = Span(
        trace_id="TRACE01",
        span_id="ROOT0001",
        kind="flow",
        name="research_flow",
        started_at=0.0,
        ended_at=2.5,
        status="ok",
    )
    agent = Span(
        trace_id="TRACE01",
        span_id="AGENT001",
        parent_span_id="ROOT0001",
        kind="agent",
        name="researcher",
        started_at=0.0,
        ended_at=2.3,
        status="ok",
    )
    llm_a = Span(
        trace_id="TRACE01",
        span_id="LLM00001",
        parent_span_id="AGENT001",
        kind="llm",
        name="sonnet-5",
        started_at=0.0,
        ended_at=0.6,
        status="ok",
        usage=Usage(input_tokens=500, output_tokens=312, cost_usd=0.0123, cost_complete=True),
        attributes={"provider": "anthropic", "model": "claude-sonnet-5"},
    )
    tool_a = Span(
        trace_id="TRACE01",
        span_id="TOOL0001",
        parent_span_id="AGENT001",
        kind="tool",
        name="web_search",
        started_at=0.6,
        ended_at=0.9,
        status="ok",
    )
    llm_b = Span(
        trace_id="TRACE01",
        span_id="LLM00002",
        parent_span_id="AGENT001",
        kind="llm",
        name="sonnet-5",
        started_at=0.9,
        ended_at=1.1,
        status="error",
        usage=Usage(input_tokens=50, output_tokens=0, cost_usd=None, cost_complete=False),
        error=ErrorInfo(
            type="ProviderError",
            message="rate limited\nretry in 30s",
            stacktrace="Traceback (most recent call last):\n  ...\nProviderError: rate limited",
        ),
    )
    pipe_a = Span(
        trace_id="TRACE01",
        span_id="PIPE0001",
        parent_span_id="ROOT0001",
        kind="pipe",
        name="normalize",
        started_at=2.3,
        ended_at=2.35,
        status="ok",
    )
    aggregate_a = Span(
        trace_id="TRACE01",
        span_id="AGGR0001",
        parent_span_id="ROOT0001",
        kind="aggregate",
        name="combine",
        started_at=2.35,
        ended_at=2.4,
        status="ok",
    )
    task_a = Span(
        trace_id="TRACE01",
        span_id="TASK0001",
        parent_span_id="ROOT0001",
        kind="task",
        name="cache_lookup",
        started_at=2.4,
        ended_at=2.45,
        status="ok",
        replayed=True,
    )
    for s in [root, agent, llm_a, tool_a, llm_b, pipe_a, aggregate_a, task_a]:
        trace.add(s)
    return trace, root


EXPECTED_SNAPSHOT = (
    "trace TRACE01 — ok — [≥$0.0123 · 862 tok · 2.5s]\n"
    "└─ ▶ research_flow [≥$0.0123 · 862 tok · 2.5s]\n"
    "   ├─ ◆ researcher [≥$0.0123 · 862 tok · 2.3s]\n"
    "   │  ├─ ▸ sonnet-5 [$0.0123 · 812 tok · 600ms]\n"
    "   │  ├─ ⚙ web_search [300ms]\n"
    "   │  └─ ▸ sonnet-5 [50 tok · 200ms] ✗ ProviderError: rate limited\n"
    "   ├─ → normalize [50ms]\n"
    "   ├─ ⇉ combine [50ms]\n"
    "   └─ • cache_lookup [50ms] (replayed)"
)


def test_render_trace_snapshot_color_false():
    trace, _root = _build_trace()
    output = render_trace(trace, color=False)
    assert output == EXPECTED_SNAPSHOT


def test_render_trace_color_true_includes_ansi_codes():
    trace, _root = _build_trace()
    output = render_trace(trace, color=True)
    assert "\x1b[" in output
    # color=False must never contain ANSI escapes
    assert "\x1b[" not in render_trace(trace, color=False)


def test_render_trace_color_auto_detects_from_tty_and_no_color(monkeypatch):
    trace, _root = _build_trace()

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert "\x1b[" in render_trace(trace, color=None)

    monkeypatch.setenv("NO_COLOR", "1")
    assert "\x1b[" not in render_trace(trace, color=None)

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert "\x1b[" not in render_trace(trace, color=None)


def test_render_trace_no_color_empty_string_still_disables_color(monkeypatch):
    """Regression: the NO_COLOR convention (https://no-color.org/) disables
    color whenever the variable is *present*, "regardless of its value" --
    including an empty string. `not os.environ.get("NO_COLOR")` treated
    NO_COLOR="" identically to NO_COLOR being unset entirely (both are
    falsy), so only a non-empty value like "1" actually worked."""
    trace, _root = _build_trace()

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("NO_COLOR", "")
    assert "\x1b[" not in render_trace(trace, color=None)

    # A NO_COLOR="0" must disable color too -- the convention says "any value".
    monkeypatch.setenv("NO_COLOR", "0")
    assert "\x1b[" not in render_trace(trace, color=None)


def test_render_trace_paused_status_and_decoration():
    trace = Trace(trace_id="T1")
    root = Span(
        trace_id="T1", span_id="R1", kind="flow", name="root", started_at=0.0, status="running"
    )
    pause = Span(
        trace_id="T1",
        span_id="P1",
        parent_span_id="R1",
        kind="pause",
        name="await",
        started_at=0.5,
        status="paused",
    )
    trace.add(root)
    trace.add(pause)
    assert trace.status == "paused"

    output = render_trace(trace, color=False)
    assert "trace T1 — paused" in output
    assert "⏸ await" in output
    assert "⏸ paused" in output


def test_render_node_omits_bracket_when_no_usage_and_no_duration():
    trace = Trace(trace_id="T1")
    root = Span(trace_id="T1", span_id="R1", kind="flow", name="root", started_at=0.0)
    trace.add(root)
    output = render_trace(trace, color=False)
    lines = output.splitlines()
    assert lines[0] == "trace T1 — running"
    assert "[" not in lines[1]


def test_humanize_tokens_999_boundary():
    trace = Trace(trace_id="T1")
    s = Span(
        trace_id="T1",
        span_id="S1",
        kind="llm",
        name="m",
        started_at=0.0,
        ended_at=0.1,
        usage=Usage(input_tokens=999, output_tokens=0, cost_usd=0.0, cost_complete=True),
    )
    trace.add(s)
    out = render_trace(trace, color=False)
    assert "999 tok" in out


def test_humanize_tokens_1000_boundary():
    trace = Trace(trace_id="T1")
    s = Span(
        trace_id="T1",
        span_id="S1",
        kind="llm",
        name="m",
        started_at=0.0,
        ended_at=0.1,
        usage=Usage(input_tokens=1000, output_tokens=0, cost_usd=0.0, cost_complete=True),
    )
    trace.add(s)
    out = render_trace(trace, color=False)
    assert "1.0k tok" in out


def test_humanize_duration_boundary_below_1s():
    trace = Trace(trace_id="T1")
    s = Span(trace_id="T1", span_id="S1", kind="task", name="t", started_at=0.0, ended_at=0.999)
    trace.add(s)
    out = render_trace(trace, color=False)
    assert "999ms" in out


def test_humanize_duration_boundary_at_1s():
    trace = Trace(trace_id="T1")
    s = Span(trace_id="T1", span_id="S1", kind="task", name="t", started_at=0.0, ended_at=1.0)
    trace.add(s)
    out = render_trace(trace, color=False)
    assert "1.0s" in out


def test_format_cost_precision_and_no_ge_prefix_when_complete():
    trace = Trace(trace_id="T1")
    s = Span(
        trace_id="T1",
        span_id="S1",
        kind="llm",
        name="m",
        started_at=0.0,
        ended_at=0.1,
        usage=Usage(input_tokens=1, output_tokens=1, cost_usd=1.0, cost_complete=True),
    )
    trace.add(s)
    out = render_trace(trace, color=False)
    assert "$1.0000" in out
    assert "≥$1.0000" not in out


def test_format_cost_incomplete_prefix():
    trace = Trace(trace_id="T1")
    s = Span(
        trace_id="T1",
        span_id="S1",
        kind="llm",
        name="m",
        started_at=0.0,
        ended_at=0.1,
        usage=Usage(input_tokens=1, output_tokens=1, cost_usd=2.5, cost_complete=False),
    )
    trace.add(s)
    out = render_trace(trace, color=False)
    assert "≥$2.5000" in out


def test_cost_segment_omitted_when_cost_usd_none():
    trace = Trace(trace_id="T1")
    s = Span(
        trace_id="T1",
        span_id="S1",
        kind="llm",
        name="m",
        started_at=0.0,
        ended_at=0.1,
        usage=Usage(input_tokens=1, output_tokens=1, cost_usd=None, cost_complete=False),
    )
    trace.add(s)
    out = render_trace(trace, color=False)
    assert "$" not in out
