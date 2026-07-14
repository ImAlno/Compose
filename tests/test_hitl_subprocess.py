"""The mandatory real two-process HITL pause/resume test (Phase 8).

Process A runs ``approval_flow`` (defined in
``tests/fixtures/subprocess_hitl_flow_defs.py``), journals one ``@task``
step, then calls ``approve("go")`` -- unanswered, so the flow *pauses*:
``.run()`` returns a normal ``Run(status="paused", ...)`` (not an
exception) and process A exits cleanly (code 0). Process B is a brand-new
``python`` invocation (no shared memory, no shared import state with A)
that resumes the same run by id with the approval answer: the
already-journaled step must not re-execute (proven by an external,
file-based side-effect counter that survives across both processes) and
the flow completes, carrying the *same* trace_id throughout.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from composeai import runs as runs_module

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _run(script_name: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FIXTURES_DIR / script_name)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_hitl_pause_in_process_a_resume_in_process_b(tmp_path):
    compose_dir = tmp_path / "compose-store"
    run_id_file = tmp_path / "run_id.txt"
    counters_file = tmp_path / "counters.txt"
    pause_result_file = tmp_path / "pause_result.json"
    result_file = tmp_path / "result.json"

    base_env = dict(os.environ)
    base_env.update(
        COMPOSE_DIR=str(compose_dir),
        RUN_ID_FILE=str(run_id_file),
        COUNTERS_FILE=str(counters_file),
        PAUSE_RESULT_FILE=str(pause_result_file),
        RESULT_FILE=str(result_file),
    )

    # --- Process A: run until approve() pauses it, then exit cleanly -------
    proc_a = _run("subprocess_hitl_run_a.py", base_env)
    assert proc_a.returncode == 0, (
        f"process A must exit cleanly on pause (pausing is not an error)\n"
        f"stdout: {proc_a.stdout}\nstderr: {proc_a.stderr}"
    )
    assert run_id_file.exists(), "flow.run() must write its run_id before exiting"
    run_id = run_id_file.read_text().strip()

    pause_result = json.loads(pause_result_file.read_text())
    assert pause_result["status"] == "paused"
    assert pause_result["pending_id"] == "go"
    original_trace_id = pause_result["trace_id"]
    assert original_trace_id

    counters_after_a = counters_file.read_text().splitlines()
    assert counters_after_a == ["prepare"]

    store = runs_module.RunStore(compose_dir / "runs.db")
    row_after_a = store.get_run(run_id)
    assert row_after_a is not None
    assert row_after_a["status"] == "paused"
    assert row_after_a["trace_id"] == original_trace_id
    pending_after_a = store.pending_interrupts_all(run_id)
    assert len(pending_after_a) == 1
    assert pending_after_a[0]["interrupt_id"] == "go"

    # --- Process B: brand-new process, resumes with the approval answer ----
    proc_b = _run("subprocess_hitl_run_b.py", base_env)
    assert proc_b.returncode == 0, (
        f"resume process failed\nstdout: {proc_b.stdout}\nstderr: {proc_b.stderr}"
    )
    assert result_file.exists()
    result = json.loads(result_file.read_text())

    assert result["status"] == "completed"
    assert result["output"] == 42  # prepare() == 1, then 1 + 41
    assert result["run_id"] == run_id
    assert result["trace_id"] == original_trace_id  # one continuous trace

    # The pre-pause step must NOT have re-executed.
    counters_after_b = counters_file.read_text().splitlines()
    assert counters_after_b == ["prepare"]

    # And the pending interrupt row is gone, the run row completed.
    store2 = runs_module.RunStore(compose_dir / "runs.db")
    assert store2.pending_interrupts_all(run_id) == []
    row_after_b = store2.get_run(run_id)
    assert row_after_b is not None
    assert row_after_b["status"] == "completed"


def test_hitl_pause_in_process_a_resume_in_process_b_async(tmp_path):
    """Async twin of ``test_hitl_pause_in_process_a_resume_in_process_b`` --
    same two-process pause/resume gate, but process A drives
    ``asyncio.run(approval_flow.arun())`` (``subprocess_hitl_run_a_async.py``)
    and process B drives ``asyncio.run(aresume(run_id, answers))``
    (``subprocess_hitl_run_b_async.py``) instead of the sync ``.run()``/
    ``resume()`` facades. Reuses the exact same flow definition module
    (``subprocess_hitl_flow_defs.approval_flow``, a plain sync-bodied
    ``@flow`` -- ``Flow.arun`` dispatches a sync body through
    ``_dispatch.run_stage`` exactly like ``.run()`` does, see
    ``_aexecute_flow``'s docstring) -- only the driver scripts differ, so
    this proves the cross-process pause/resume gate holds for the async
    surface too, not just the sync one. Same assertion shape as the sync
    test above: completed output, replayed (not re-executed) pre-pause
    step, one continuous trace_id."""
    compose_dir = tmp_path / "compose-store"
    run_id_file = tmp_path / "run_id.txt"
    counters_file = tmp_path / "counters.txt"
    pause_result_file = tmp_path / "pause_result.json"
    result_file = tmp_path / "result.json"

    base_env = dict(os.environ)
    base_env.update(
        COMPOSE_DIR=str(compose_dir),
        RUN_ID_FILE=str(run_id_file),
        COUNTERS_FILE=str(counters_file),
        PAUSE_RESULT_FILE=str(pause_result_file),
        RESULT_FILE=str(result_file),
    )

    # --- Process A: arun() until approve() pauses it, then exit cleanly ----
    proc_a = _run("subprocess_hitl_run_a_async.py", base_env)
    assert proc_a.returncode == 0, (
        f"process A must exit cleanly on pause (pausing is not an error)\n"
        f"stdout: {proc_a.stdout}\nstderr: {proc_a.stderr}"
    )
    assert run_id_file.exists(), "flow.arun() must write its run_id before exiting"
    run_id = run_id_file.read_text().strip()

    pause_result = json.loads(pause_result_file.read_text())
    assert pause_result["status"] == "paused"
    assert pause_result["pending_id"] == "go"
    original_trace_id = pause_result["trace_id"]
    assert original_trace_id

    counters_after_a = counters_file.read_text().splitlines()
    assert counters_after_a == ["prepare"]

    store = runs_module.RunStore(compose_dir / "runs.db")
    row_after_a = store.get_run(run_id)
    assert row_after_a is not None
    assert row_after_a["status"] == "paused"
    assert row_after_a["trace_id"] == original_trace_id
    pending_after_a = store.pending_interrupts_all(run_id)
    assert len(pending_after_a) == 1
    assert pending_after_a[0]["interrupt_id"] == "go"

    # --- Process B: brand-new process, aresume()s with the approval answer -
    proc_b = _run("subprocess_hitl_run_b_async.py", base_env)
    assert proc_b.returncode == 0, (
        f"resume process failed\nstdout: {proc_b.stdout}\nstderr: {proc_b.stderr}"
    )
    assert result_file.exists()
    result = json.loads(result_file.read_text())

    assert result["status"] == "completed"
    assert result["output"] == 42  # prepare() == 1, then 1 + 41
    assert result["run_id"] == run_id
    assert result["trace_id"] == original_trace_id  # one continuous trace

    # The pre-pause step must NOT have re-executed.
    counters_after_b = counters_file.read_text().splitlines()
    assert counters_after_b == ["prepare"]

    # And the pending interrupt row is gone, the run row completed.
    store2 = runs_module.RunStore(compose_dir / "runs.db")
    assert store2.pending_interrupts_all(run_id) == []
    row_after_b = store2.get_run(run_id)
    assert row_after_b is not None
    assert row_after_b["status"] == "completed"


def test_hitl_agent_approval_pause_in_process_a_resume_in_process_b(tmp_path):
    """Same shape as above, but for a standalone @agent's approval-gated tool.

    This specifically exercises `agent_state` snapshot decoding in a
    genuinely fresh process (process B never encodes a `Message`/
    `ToolResultPart` itself before needing to decode the ones process A
    persisted) -- a real gap this phase's self-review caught and fixed in
    `composeai.messages` (see its module-level `register_serializable` calls).
    """
    compose_dir = tmp_path / "compose-store"
    run_id_file = tmp_path / "run_id.txt"
    pause_result_file = tmp_path / "pause_result.json"
    result_file = tmp_path / "result.json"

    base_env = dict(os.environ)
    base_env.update(
        COMPOSE_DIR=str(compose_dir),
        RUN_ID_FILE=str(run_id_file),
        PAUSE_RESULT_FILE=str(pause_result_file),
        RESULT_FILE=str(result_file),
    )

    proc_a = _run("subprocess_hitl_agent_run_a.py", base_env)
    assert proc_a.returncode == 0, (
        f"process A must exit cleanly on pause\nstdout: {proc_a.stdout}\nstderr: {proc_a.stderr}"
    )
    run_id = run_id_file.read_text().strip()
    pause_result = json.loads(pause_result_file.read_text())
    assert pause_result["status"] == "paused"
    assert pause_result["pending_id"] == "tool:dangerous_tool:call_1"
    original_trace_id = pause_result["trace_id"]
    assert original_trace_id

    store = runs_module.RunStore(compose_dir / "runs.db")
    row_after_a = store.get_run(run_id)
    assert row_after_a is not None
    assert row_after_a["kind"] == "agent"
    assert row_after_a["status"] == "paused"
    # `run.id` of a paused standalone agent must equal its durable row id.
    assert run_id == row_after_a["run_id"]

    proc_b = _run("subprocess_hitl_agent_run_b.py", base_env)
    assert proc_b.returncode == 0, (
        f"resume process failed\nstdout: {proc_b.stdout}\nstderr: {proc_b.stderr}"
    )
    result = json.loads(result_file.read_text())

    assert result["status"] == "completed"
    assert result["output"] == "final"
    assert result["run_id"] == run_id
    assert result["trace_id"] == original_trace_id
