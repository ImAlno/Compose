"""Shared flow definition for the HITL pause/resume-across-processes test.

Imported (unchanged) by both driver scripts (``subprocess_hitl_run_a.py``,
``subprocess_hitl_run_b.py``) so both processes register the *same* flow
name with the *same* fingerprint. Configuration travels via environment
variables set by the test/driver, never hardcoded paths:

- ``COMPOSE_DIR``: where the RunStore lives (shared by both processes).
- ``COUNTERS_FILE``: appended to by the task on actual execution -- a
  side-effect counter external to any single process's memory, proving the
  pre-pause step didn't re-execute across the pause/resume boundary.
"""

from __future__ import annotations

import os

from composeai import flow, task
from composeai.hitl import approve


def _counters_file() -> str:
    return os.environ["COUNTERS_FILE"]


def _record(name: str) -> None:
    with open(_counters_file(), "a") as f:
        f.write(name + "\n")


@task
def prepare() -> int:
    _record("prepare")
    return 1


@flow
def approval_flow() -> int:
    n = prepare()
    if approve("go"):
        return n + 41
    return -1
