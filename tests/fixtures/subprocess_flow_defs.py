"""Shared flow definitions for the crash/resume subprocess test.

Imported (unchanged) by both driver scripts (``subprocess_run_a.py``,
``subprocess_run_b.py``) so both processes register the *same* flow name
with the *same* fingerprint (sha256 of this file's ``crash_flow`` source --
identical either way, since it's the same file). Configuration travels via
environment variables set by the test/driver, never hardcoded paths:

- ``COMPOSE_DIR``: where the RunStore lives (shared by both processes).
- ``RUN_ID_FILE``: written with the flow's run_id as soon as it's known,
  so the test (and driver B) can pick it up even though the flow crashes
  before ``.run()`` would otherwise return it.
- ``COUNTERS_FILE``: appended to by each task on actual execution -- a
  side-effect counter external to any single process's memory, since
  proving "didn't re-execute" must survive the crash between processes.
- ``COMPOSE_TEST_CRASH``: ``"1"`` means hard-crash (``os._exit``) right
  after the two tasks run, simulating a mid-flow process crash.
"""

from __future__ import annotations

import os

from composeai import flow, task
from composeai import runs as runs_module


def _counters_file() -> str:
    return os.environ["COUNTERS_FILE"]


def _record(name: str) -> None:
    with open(_counters_file(), "a") as f:
        f.write(name + "\n")


@task
def step1() -> int:
    _record("step1")
    return 1


@task
def step2() -> int:
    _record("step2")
    return 2


@flow
def crash_flow() -> int:
    ctx = runs_module.current_run_context()
    assert ctx is not None
    with open(os.environ["RUN_ID_FILE"], "w") as f:
        f.write(ctx.run_id)

    a = step1()
    b = step2()

    if os.environ.get("COMPOSE_TEST_CRASH") == "1":
        os._exit(17)  # hard crash: no Python cleanup, no exception handling at all

    return a + b
