"""The mandatory real two-process crash/resume test (Phase 7).

Process A starts ``crash_flow`` (defined in ``tests/fixtures/subprocess_flow_defs.py``),
journals two ``@task`` steps, then hard-crashes via ``os._exit`` -- no
Python cleanup, no exception path, nothing but whatever was already
durably committed to the SQLite store. Process B is a brand-new
``python`` invocation (no shared memory, no shared import state with A)
that resumes the same run by id: the two already-journaled steps must
replay (never re-execute -- proven by an external, file-based side-effect
counter that survives across both processes) and the flow completes,
carrying the *same* trace_id throughout.
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


def test_crash_then_resume_across_two_real_processes(tmp_path):
    compose_dir = tmp_path / "compose-store"
    run_id_file = tmp_path / "run_id.txt"
    counters_file = tmp_path / "counters.txt"
    result_file = tmp_path / "result.json"

    base_env = dict(os.environ)
    base_env.update(
        COMPOSE_DIR=str(compose_dir),
        RUN_ID_FILE=str(run_id_file),
        COUNTERS_FILE=str(counters_file),
        RESULT_FILE=str(result_file),
    )

    # --- Process A: run until it journals 2 steps, then hard-crash ---------
    env_a = dict(base_env, COMPOSE_TEST_CRASH="1")
    proc_a = _run("subprocess_run_a.py", env_a)

    assert proc_a.returncode == 17, (
        f"expected the hard-crash exit code 17, got {proc_a.returncode}\n"
        f"stdout: {proc_a.stdout}\nstderr: {proc_a.stderr}"
    )
    assert run_id_file.exists(), "flow body must write its run_id before crashing"
    run_id = run_id_file.read_text().strip()

    counters_after_a = counters_file.read_text().splitlines()
    assert counters_after_a == ["step1", "step2"]

    # Sanity: the run row exists and was never marked completed (A crashed
    # via os._exit, which skips composeai's own failure-handling too).
    store = runs_module.RunStore(compose_dir / "runs.db")
    row_after_a = store.get_run(run_id)
    assert row_after_a is not None
    assert row_after_a["status"] != "completed"
    original_trace_id = row_after_a["trace_id"]
    assert original_trace_id

    # --- Process B: brand-new process, resumes by run_id --------------------
    env_b = dict(base_env, COMPOSE_TEST_CRASH="0")
    proc_b = _run("subprocess_run_b.py", env_b)

    assert proc_b.returncode == 0, (
        f"resume process failed\nstdout: {proc_b.stdout}\nstderr: {proc_b.stderr}"
    )
    assert result_file.exists()
    result = json.loads(result_file.read_text())

    assert result["status"] == "completed"
    assert result["output"] == 3  # step1() + step2() == 1 + 2
    assert result["run_id"] == run_id
    assert result["trace_id"] == original_trace_id  # same trace across the crash

    # The two steps must NOT have re-executed during resume.
    counters_after_b = counters_file.read_text().splitlines()
    assert counters_after_b == ["step1", "step2"]

    # And the run row itself is now durably marked completed.
    store2 = runs_module.RunStore(compose_dir / "runs.db")
    row_after_b = store2.get_run(run_id)
    assert row_after_b is not None
    assert row_after_b["status"] == "completed"
    assert row_after_b["trace_id"] == original_trace_id
