"""Tests for ``composeai.flow`` -- ``@task``, ``@flow``, ``compose.resume`` (Phase 7).

Everything here runs offline against ``FakeModel``; ``COMPOSE_DIR`` is
redirected at a per-session temp dir by ``tests/conftest.py`` so nothing
ever touches the repo's ``./.compose``.

Deliberately *not* using ``from __future__ import annotations`` (same
reason as ``test_agent.py``/``test_combinators.py``): a couple of tests
define a small pydantic model local to the test function and reference it
in a decorated function's annotations.
"""

import threading
import time

import pytest
from pydantic import BaseModel

from composeai import runs, tracing
from composeai.agentfn import agent
from composeai.combinators import aggregate, pipe
from composeai.combinators import map as compose_map
from composeai.errors import ConfigError, ResumeMismatchError, SerializationError
from composeai.flow import _FLOW_REGISTRY, _TASK_REGISTRY, flow, resume, task
from composeai.hitl import approve
from composeai.runs import Budget
from composeai.testing import FakeModel


@pytest.fixture(autouse=True)
def _clear_flow_registry():
    # Flow names are unique per-process (ConfigError on duplicate); tests
    # define plenty of small flows, so drop registrations between tests to
    # avoid bleed between them re-registering the same function name.
    yield
    _FLOW_REGISTRY.clear()


# --- @task outside a flow ----------------------------------------------------


def test_task_outside_flow_executes_directly():
    calls = []

    @task
    def add(a: int, b: int) -> int:
        calls.append((a, b))
        return a + b

    assert add(2, 3) == 5
    assert calls == [(2, 3)]


def test_task_outside_flow_opens_task_span_with_input_output():
    @task
    def double(x: int) -> int:
        return x * 2

    with tracing.span("flow", "root"):
        result = double(21)
        trace = tracing.current_trace()

    assert result == 42
    assert trace is not None
    task_spans = [s for s in trace.spans if s.kind == "task"]
    assert len(task_spans) == 1
    assert task_spans[0].name == "double"
    assert task_spans[0].output == 42
    assert task_spans[0].replayed is False


def test_task_bare_and_configured_forms_both_work():
    @task(retries=1, timeout=None, name="custom_name")
    def f(x: int) -> int:
        return x

    assert f.name == "custom_name"
    assert f(5) == 5


# --- @task retries + timeout --------------------------------------------------


def test_task_retries_then_succeeds():
    attempts = {"n": 0}

    @task(retries=2)
    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("not yet")
        return "ok"

    with tracing.span("flow", "root"):
        result = flaky()
        trace = tracing.current_trace()

    assert result == "ok"
    assert attempts["n"] == 3
    assert trace is not None
    task_span = next(s for s in trace.spans if s.kind == "task")
    assert len(task_span.attributes["retries"]) == 2


def test_task_retries_exhausted_raises():
    @task(retries=1)
    def always_fails() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        always_fails()


def test_task_timeout_raises_task_timeout_error_promptly():
    from composeai.errors import TaskTimeoutError

    @task(timeout=0.05)
    def slow() -> None:
        time.sleep(2)

    started = time.monotonic()
    with pytest.raises(TaskTimeoutError):
        slow()
    elapsed = time.monotonic() - started
    assert elapsed < 1.0  # must not block for the full 2s sleep


# --- @task called multiple times: keys #1..#3 --------------------------------


def test_task_called_three_times_gets_sequential_keys():
    seen_keys = []

    @task
    def record(x: int) -> int:
        return x

    @flow
    def three_calls() -> list[int]:
        ctx = runs.current_run_context()
        results = []
        for i in range(3):
            results.append(record(i))
            seen_keys.append(sorted(ctx.preloaded.keys()) if ctx else None)
        return results

    run = three_calls.run()
    assert run.output == [0, 1, 2]

    store = runs.open_default()
    keys = sorted(store.journal_all(run.id).keys())
    assert keys == ["record#1", "record#2", "record#3"]


# --- journal replay end-to-end: crash mid-flow, resume ------------------------


def test_flow_replay_skips_completed_tasks_and_reruns_failed_one():
    counters = {"t1": 0, "t2": 0, "t3": 0}
    should_fail = {"value": True}

    @task
    def t1() -> int:
        counters["t1"] += 1
        return 1

    @task
    def t2() -> int:
        counters["t2"] += 1
        return 2

    @task
    def t3() -> int:
        counters["t3"] += 1
        if should_fail["value"]:
            raise RuntimeError("t3 boom")
        return 3

    @flow
    def three_steps() -> int:
        a = t1()
        b = t2()
        c = t3()
        return a + b + c

    with pytest.raises(RuntimeError, match="t3 boom"):
        three_steps.run()

    assert counters == {"t1": 1, "t2": 1, "t3": 1}

    store = runs.open_default()
    rows = store.list_runs(kind="flow", limit=10)
    run_row = next(r for r in rows if r["name"] == "three_steps")
    assert run_row["status"] == "failed"

    should_fail["value"] = False
    run = resume(run_row["run_id"])

    assert run.output == 6
    # t1/t2 must NOT re-execute; t3 (never journaled -- it raised) does.
    assert counters == {"t1": 1, "t2": 1, "t3": 2}


# --- compose.map in-flow: keys match input order, not completion order -------


def test_map_in_flow_journal_keys_match_input_order_not_completion_order():
    execution_log = []
    lock = threading.Lock()

    @task
    def process(duration_and_item: tuple[float, int]) -> int:
        duration, item = duration_and_item
        time.sleep(duration)
        with lock:
            execution_log.append(item)
        return item * 10

    @flow
    def run_map() -> list[int]:
        # Reversed durations: item 0 sleeps longest, so item 3 *finishes*
        # first if completion order (wrongly) drove key assignment.
        items = [(0.15, 0), (0.10, 1), (0.05, 2), (0.0, 3)]
        return compose_map(process, items)

    run = run_map.run()
    assert run.output == [0, 10, 20, 30]
    # Completion order proves parallelism actually happened out-of-order.
    assert execution_log == [3, 2, 1, 0]

    store = runs.open_default()
    journal = store.journal_all(run.id)
    # Keys assigned in *input* order regardless of completion order: each
    # item gets its own deterministic scope ("map#1[i]", reserved serially
    # in input order before dispatch), so every item's nested `process`
    # call is "process#1" *within its own scope* rather than racing a
    # single shared "process" counter across items.
    assert set(journal.keys()) == {
        "map#1[0]/process#1",
        "map#1[1]/process#1",
        "map#1[2]/process#1",
        "map#1[3]/process#1",
    }

    # Run again (resume): single execution per item (no re-execution).
    execution_log.clear()
    run2 = resume(run.id)
    assert run2.output == [0, 10, 20, 30]
    assert execution_log == []  # nothing re-executed -- all replayed from journal


# --- agent-in-flow auto-journal -----------------------------------------------


def test_agent_in_flow_journals_whole_run_and_replays_on_resume():
    # The flow itself must NOT complete on the first run (else `resume()`
    # would short-circuit to the "already completed" branch and never
    # re-execute anything) -- a task *after* the agent call fails once so
    # the agent's step is journaled but the flow run row ends up "failed".
    model = FakeModel(["hello from agent"])
    should_fail = {"value": True}

    @agent(model=model)
    def greeter(name: str) -> str:
        return f"Greet {name}"

    @task
    def after_agent() -> str:
        if should_fail["value"]:
            raise RuntimeError("boom")
        return "after-done"

    @flow
    def greet_flow(name: str) -> str:
        greeting = greeter(name)
        after_agent()
        return greeting

    with pytest.raises(RuntimeError, match="boom"):
        greet_flow.run("World")
    assert len(model.requests) == 1

    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="flow", limit=50) if r["name"] == "greet_flow"]
    run_row = rows[0]
    assert run_row["status"] == "failed"

    should_fail["value"] = False
    # Resume: FakeModel must NOT be consumed again (script stays exhausted
    # at 1 request); replayed agent span has zero usage.
    run2 = resume(run_row["run_id"])
    assert run2.output == "hello from agent"
    assert len(model.requests) == 1

    agent_spans = [s for s in run2.trace.spans if s.kind == "agent"]
    assert len(agent_spans) == 1
    assert agent_spans[0].replayed is True
    assert agent_spans[0].usage is None


def test_agent_in_flow_miss_path_span_carries_step_key():
    # Minor-findings cleanup (Phase 10): the replay-path agent span already
    # carried a `step_key` attribute (see the resume assertion above via
    # `.replayed`); the first-execution ("miss") path must carry it too, for
    # parity -- previously it only showed up in the replay path.
    model = FakeModel(["hello from agent"])

    @agent(model=model)
    def greeter(name: str) -> str:
        return f"Greet {name}"

    @flow
    def greet_flow(name: str) -> str:
        return greeter(name)

    run = greet_flow.run("World")
    assert run.output == "hello from agent"

    agent_spans = [s for s in run.trace.spans if s.kind == "agent"]
    assert len(agent_spans) == 1
    assert agent_spans[0].replayed is False
    assert agent_spans[0].attributes.get("step_key") == "greeter#1"


# --- non-serializable step value -----------------------------------------------


def test_task_journaling_nonserializable_result_names_step_key():
    class Unserializable:
        pass

    @task
    def bad() -> Unserializable:
        return Unserializable()

    @flow
    def bad_flow() -> None:
        bad()

    with pytest.raises(SerializationError, match="bad#1"):
        bad_flow.run()


# --- unserializable flow args rejected up front --------------------------------


def test_flow_rejects_unserializable_args_up_front():
    class Unserializable:
        pass

    @flow
    def takes_arg(x) -> None:
        return None

    with pytest.raises(SerializationError):
        takes_arg.run(Unserializable())

    # No run row should have been created for this rejected call.
    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="flow", limit=50) if r["name"] == "takes_arg"]
    assert rows == []


# --- fingerprint mismatch ------------------------------------------------------


def test_resume_fingerprint_mismatch_raises_unless_allowed():
    @task
    def step() -> int:
        return 1

    @flow
    def fp_flow() -> int:
        step()
        raise RuntimeError("stop before completing")

    with pytest.raises(RuntimeError):
        fp_flow.run()

    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="flow", limit=50) if r["name"] == "fp_flow"]
    run_id = rows[0]["run_id"]

    flow_obj = _FLOW_REGISTRY["fp_flow"]
    original_fingerprint = flow_obj.fingerprint
    flow_obj.fingerprint = "deliberately-different"
    try:
        with pytest.raises(ResumeMismatchError):
            resume(run_id)
        # Escape hatch still lets it through (will raise again, since the
        # step body still raises -- proving it actually re-executed).
        with pytest.raises(RuntimeError):
            resume(run_id, allow_code_change=True)
    finally:
        flow_obj.fingerprint = original_fingerprint


# --- resume of a completed run -------------------------------------------------


def test_resume_of_completed_run_returns_stored_output_without_executing():
    counters = {"n": 0}

    @task
    def counted() -> int:
        counters["n"] += 1
        return 7

    @flow
    def done_flow() -> int:
        return counted()

    run = done_flow.run()
    assert run.output == 7
    assert counters["n"] == 1

    run2 = resume(run.id)
    assert run2.output == 7
    assert run2.status == "completed"
    assert counters["n"] == 1  # not re-executed


# --- trace_id continuity -------------------------------------------------------


def test_resume_spans_carry_original_trace_id():
    @task
    def step_a() -> int:
        return 1

    @task
    def step_b() -> int:
        raise RuntimeError("boom once")

    should_fail = {"value": True}

    @task
    def step_c() -> int:
        if should_fail["value"]:
            raise RuntimeError("boom")
        return 3

    @flow
    def tf() -> int:
        a = step_a()
        c = step_c()
        return a + c

    with pytest.raises(RuntimeError):
        tf.run()

    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="flow", limit=50) if r["name"] == "tf"]
    run_row = rows[0]
    original_trace_id = run_row["trace_id"]
    assert original_trace_id

    should_fail["value"] = False
    run2 = resume(run_row["run_id"])
    assert run2.trace.trace_id == original_trace_id
    for span in run2.trace.spans:
        assert span.trace_id == original_trace_id


# --- resume: missing run / unregistered flow -----------------------------------


def test_resume_missing_run_id_raises_config_error():
    from composeai.errors import ConfigError

    with pytest.raises(ConfigError):
        resume("no-such-run-id")


def test_resume_unregistered_flow_name_raises_config_error():
    from composeai.errors import ConfigError

    @flow
    def registered_then_forgotten() -> int:
        return 1

    run = registered_then_forgotten.run()
    del _FLOW_REGISTRY["registered_then_forgotten"]

    with pytest.raises(ConfigError, match="import"):
        resume(run.id)


# --- duplicate flow name --------------------------------------------------------


def test_duplicate_flow_name_raises_config_error():
    from composeai.errors import ConfigError

    @flow(name="dupe")
    def one() -> int:
        return 1

    with pytest.raises(ConfigError):

        @flow(name="dupe")
        def two() -> int:
            return 2


# --- standalone agent.run() creates a runs row ---------------------------------


def test_standalone_agent_run_creates_run_row():
    model = FakeModel(["hi"])

    @agent(model=model)
    def solo(prompt: str) -> str:
        return prompt

    run = solo.run("hello")
    assert run.output == "hi"

    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="agent", limit=50) if r["name"] == "solo"]
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["trace_id"] == run.trace.trace_id


def test_agent_nested_in_pipe_does_not_get_its_own_run_row():
    from composeai.combinators import pipe

    model = FakeModel(["out"])

    @agent(model=model)
    def inner(x: str) -> str:
        return x

    def upstream(x: str) -> str:
        return x

    p = pipe(upstream, inner)
    store = runs.open_default()
    before = len(store.list_runs(kind="agent", limit=1000))
    p.run("in")
    after = len(store.list_runs(kind="agent", limit=1000))
    assert after == before  # nested agent call must not add an agent-kind run row


# --- flow(x) sugar + flow.stream -------------------------------------------------


def test_flow_call_sugar_returns_output_directly():
    @flow
    def sugar_flow(x: int) -> int:
        return x + 1

    assert sugar_flow(41) == 42


def test_flow_stream_reuses_run_stream_machinery():
    @task
    def step() -> int:
        return 9

    @flow
    def streamed() -> int:
        return step()

    run_stream = streamed.stream()
    events_seen = list(run_stream)
    assert any(e.kind == "run_finished" for e in events_seen)
    assert run_stream.run.output == 9


# --- @flow with pydantic model args/returns (annotation registration) -----------


class Payload(BaseModel):
    n: int


def test_flow_with_pydantic_model_args_registers_types_for_decode():
    @flow(name="pyd_flow")
    def pyd_flow(p: Payload) -> Payload:
        return Payload(n=p.n * 2)

    run = pyd_flow.run(Payload(n=5))
    assert run.output == Payload(n=10)


def test_flow_fingerprint_degrades_without_source():
    """Regression: @flow on a REPL/exec-defined function must not crash."""
    namespace: dict = {}
    exec(  # noqa: S102 -- deliberately sourceless
        "def repl_flow(x):\n    return x * 2\n", namespace
    )
    from composeai.flow import flow as flow_decorator

    f = flow_decorator(name="repl_flow_test")(namespace["repl_flow"])
    assert f.fingerprint.startswith("nosource:")
    run = f.run(21)
    assert run.output == 42


# --- capstone fix wave A regressions -------------------------------------------


# --- journal-key determinism: every stage kind, not just @task -----------------


def test_map_over_agent_stage_in_flow_journal_keys_are_scoped_by_input_index():
    """compose.map(some_agent, items) inside a @flow must key each item's
    journal entries deterministically -- previously only @task got a
    pre-reserved key; an @agent stage fell through to `_dispatch_plain` and
    raced `ctx.next_key()` across worker threads, so completion order (not
    input order) decided the keys.

    The model's response echoes the request's own prompt text (rather than
    a fixed positional script slot) -- three concurrently-dispatched model
    calls can legitimately arrive at ``FakeModel`` in any order (that's real
    concurrency, not a bug), so a fixed-order script would make this test's
    *output* assertion flaky for reasons unrelated to what it's checking:
    journal *key* determinism, not model-call arrival order.
    """

    def echo_prompt(request):
        return request.messages[-1].parts[0].text.upper()

    model = FakeModel([echo_prompt, echo_prompt, echo_prompt])

    @agent(model=model)
    def summarize(x: str) -> str:
        return x

    @flow
    def run_map() -> list[str]:
        return compose_map(summarize, ["x", "y", "z"])

    run = run_map.run()
    assert run.output == ["X", "Y", "Z"]

    store = runs.open_default()
    keys = set(store.journal_all(run.id).keys())
    assert keys == {"map#1[0]/summarize#1", "map#1[1]/summarize#1", "map#1[2]/summarize#1"}

    # Resume (nothing left to do) must replay every item, not re-call the model.
    run2 = resume(run.id)
    assert run2.output == ["X", "Y", "Z"]
    assert len(model.requests) == 3


def test_map_over_plain_wrapper_calling_a_task_is_scoped_deterministically():
    """A plain function that merely calls a @task internally (not a bare
    @task itself) must also get deterministic per-item journal keys."""
    calls = []

    @task
    def do_work(x: int) -> int:
        calls.append(x)
        return x * 2

    def wrapper(x: int) -> int:
        return do_work(x)

    @flow
    def run_map() -> list[int]:
        return compose_map(wrapper, [10, 20, 30])

    run = run_map.run()
    assert run.output == [20, 40, 60]

    store = runs.open_default()
    keys = set(store.journal_all(run.id).keys())
    assert keys == {"map#1[0]/do_work#1", "map#1[1]/do_work#1", "map#1[2]/do_work#1"}


def test_aggregate_branches_in_flow_get_deterministic_per_branch_scope():
    # An Aggregate used directly as a flow's own top-level call (`agg(x)`/
    # `agg.run(x)`) goes through `_run_top` (a fresh run/trace of its own --
    # not the pattern this fix targets); the intended "nested stage" path is
    # exercised via `_invoke_stage`, e.g. an Aggregate mapped/piped as one
    # stage of an outer combinator -- here, `compose.map` over a single item
    # whose stage is the aggregate itself.
    @task
    def task_a() -> str:
        return "a-result"

    @task
    def task_b() -> str:
        return "b-result"

    agg = aggregate(alpha=lambda x: task_a(), beta=lambda x: task_b())

    @flow
    def run_agg() -> list[dict]:
        return compose_map(agg, [None])

    run = run_agg.run()
    assert run.output == [{"alpha": "a-result", "beta": "b-result"}]

    store = runs.open_default()
    keys = set(store.journal_all(run.id).keys())
    assert keys == {
        "map#1[0]/aggregate#1/alpha/task_a#1",
        "map#1[0]/aggregate#1/beta/task_b#1",
    }


# --- nested @flow journaling -----------------------------------------------


def test_nested_flow_call_journals_as_one_step_and_replays_on_resume():
    inner_calls = {"n": 0}

    @task
    def inner_task() -> int:
        inner_calls["n"] += 1
        return 5

    @flow
    def inner_flow() -> int:
        return inner_task()

    should_fail = {"value": True}

    @task
    def risky() -> int:
        if should_fail["value"]:
            raise RuntimeError("boom")
        return 100

    @flow
    def outer_flow() -> int:
        a = inner_flow()
        b = risky()
        return a + b

    with pytest.raises(RuntimeError):
        outer_flow.run()
    assert inner_calls["n"] == 1

    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="flow", limit=50) if r["name"] == "outer_flow"]
    run_row = rows[0]
    assert run_row["status"] == "failed"

    should_fail["value"] = False
    run2 = resume(run_row["run_id"])
    assert run2.output == 105
    # inner_flow (and its own @task) must NOT re-execute on resume -- it
    # replays from a single journaled step, exactly like a nested @agent.
    assert inner_calls["n"] == 1

    journal = store.journal_all(run_row["run_id"])
    assert "inner_flow#1" in journal


# --- budget persisted across resume -----------------------------------------


def test_flow_budget_persists_across_resume_and_still_enforces():
    from composeai.errors import BudgetExceededError
    from composeai.messages import Usage

    model = FakeModel(
        ["first", "second", "third"], usage=Usage(input_tokens=5, output_tokens=5)
    )

    @agent(model=model)
    def spender(prompt: str) -> str:
        return prompt

    @flow
    def budget_flow() -> str:
        spender("go")
        if not approve("continue"):
            return "stopped"
        spender("go again")
        spender("go once more")
        return "done"

    run = budget_flow.run(budget=Budget(tokens=15))
    assert run.status == "paused"

    # Without the fix, resume() always passed budget=None -- the resumed
    # attempt would spend without limit and complete normally instead of
    # raising once the (still-active) 15-token cap is exceeded.
    with pytest.raises(BudgetExceededError):
        resume(run.id, {"continue": True})


# --- resume() validates before journaling answers ---------------------------


def test_resume_does_not_journal_answers_when_fingerprint_mismatches():
    @flow
    def gate_flow() -> str:
        if approve("go"):
            return "yes"
        return "no"

    run = gate_flow.run()
    assert run.status == "paused"

    flow_obj = _FLOW_REGISTRY["gate_flow"]
    original_fingerprint = flow_obj.fingerprint
    flow_obj.fingerprint = "deliberately-different"
    try:
        with pytest.raises(ResumeMismatchError):
            resume(run.id, {"go": True})
    finally:
        flow_obj.fingerprint = original_fingerprint

    # The answer must NOT have been durably journaled by the aborted resume
    # attempt above -- resuming again with the OPPOSITE answer must win.
    run2 = resume(run.id, {"go": False})
    assert run2.output == "no"


# --- @task name uniqueness --------------------------------------------------


def test_task_name_collision_raises_config_error():
    @task(name="dupe_task_name")
    def one() -> int:
        return 1

    with pytest.raises(ConfigError):

        @task(name="dupe_task_name")
        def two() -> int:
            return 2

    assert "dupe_task_name" in _TASK_REGISTRY


# --- timeout zombie thread loses journal access -----------------------------


def test_timed_out_task_zombie_thread_loses_journal_access_after_abandonment():
    from composeai.errors import TaskTimeoutError

    started = threading.Event()
    proceed = threading.Event()

    @task
    def inner_task() -> int:
        return 1

    @task(timeout=0.05)
    def slow() -> str:
        started.set()
        proceed.wait(timeout=5)
        # By the time this line runs, the caller has already timed out and
        # marked this worker abandoned -- this nested @task call must fail
        # internally (swallowed inside the zombie's own worker; there is
        # nowhere left to report it) rather than journal a new step under an
        # already-"completed" run.
        inner_task()
        return "should not get here"  # pragma: no cover

    @flow
    def outer() -> str:
        try:
            return slow()
        except TaskTimeoutError:
            return "caught-timeout"

    run = outer.run()
    assert run.output == "caught-timeout"
    assert run.status == "completed"

    assert started.wait(timeout=2)
    proceed.set()
    # Give the abandoned daemon thread a moment to reach (and be blocked by)
    # the guard.
    time.sleep(0.3)

    store = runs.open_default()
    journal = store.journal_all(run.id)
    assert not any("inner_task" in k for k in journal)
    # The run itself must still show "completed" -- the zombie's failed
    # journal touch must not have reopened or corrupted it.
    row = store.get_run(run.id)
    assert row is not None
    assert row["status"] == "completed"


# --- paused top-level pipe()/aggregate() refuses cleanly ---------------------


def test_paused_top_level_pipe_refuses_with_config_error_and_leaves_no_row():
    def step1(x: str) -> str:
        return x

    def step2(x: str) -> str:
        if not approve("go"):
            return "denied"
        return "sent:" + x

    p = pipe(step1, step2)
    with pytest.raises(ConfigError, match="@flow"):
        p.run("hello")

    # No orphaned "paused" row left behind for `compose runs` to show.
    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="pipe", limit=50) if r["name"] == p._name]
    assert rows == []


def test_paused_top_level_aggregate_refuses_with_config_error_and_leaves_no_row():
    def branch(x: str) -> str:
        if not approve("go2"):
            return "denied"
        return "ok:" + x

    agg = aggregate(only=branch)
    with pytest.raises(ConfigError, match="@flow"):
        agg.run("hello")

    store = runs.open_default()
    rows = [r for r in store.list_runs(kind="aggregate", limit=50) if r["name"] == agg._name]
    assert rows == []


def test_pipe_nested_as_a_flow_stage_can_pause_and_resume_normally():
    """The v1 contract: durable pauses DO work when a pipe is used as a
    nested STAGE of something running inside a @flow (e.g. via
    compose.map) rather than invoked top-level -- confirming the refusal
    above is scoped to the true top-level ``pipe()``/``aggregate()`` call
    (which always mints its own durable row via ``_run_top``, regardless of
    ambient context -- unlike ``@agent``/nested-``@flow`` calls, a bare
    ``Pipeline``/``Aggregate`` object has no "am I nested in a flow?" check
    of its own), not every pipe/aggregate object regardless of context.
    """

    def step1(x: str) -> str:
        return x

    def step2(x: str) -> str:
        if not approve("go3"):
            return "denied"
        return "sent:" + x

    p = pipe(step1, step2)

    @flow
    def run_it(x: str) -> list[str]:
        return compose_map(p, [x])

    run = run_it.run("hello")
    assert run.status == "paused"

    run2 = resume(run.id, {"go3": True})
    assert run2.output == ["sent:hello"]
