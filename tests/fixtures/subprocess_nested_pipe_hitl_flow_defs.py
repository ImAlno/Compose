"""Shared flow definition for the nested-pipe HITL pause/resume-across-
processes test (v0.5.0 Plan A, Task 3 -- the cross-process twin of
``test_flow_pauses_inside_nested_pipe_and_resumes`` in
``tests/test_nested_combinators.py``).

Same shape as ``subprocess_hitl_flow_defs.py`` (imported unchanged by both
driver scripts so both processes register the same flow name/fingerprint),
except the ``@flow`` body calls a ``pipe()`` result DIRECTLY
(``the_pipe(x)``, i.e. ``Pipeline.__call__``'s nested-adoption path added by
Tasks 1-2) rather than a bare ``@task`` -- pinning that a pause raised from
inside a stage of a pipe nested this way still pauses the ENCLOSING flow run
durably, and that the pre-pause stage (an ``@agent`` backed by a
``FakeModel``) does not re-execute on resume in a brand-new process. Since a
``FakeModel`` instance can't survive across a process boundary, re-execution
is instead pinned by an external, file-based side-effect counter
(``COUNTERS_FILE``), the same technique ``subprocess_hitl_flow_defs.py``
uses for its own ``prepare`` task.
"""

from __future__ import annotations

import os

from composeai import agent, flow, pipe
from composeai.hitl import approve
from composeai.testing import FakeModel


def _counters_file() -> str:
    return os.environ["COUNTERS_FILE"]


def _record(name: str) -> None:
    with open(_counters_file(), "a") as f:
        f.write(name + "\n")


model_a = FakeModel(["stage-a-out"])


@agent(model=model_a, name="subproc_nested_pipe_stage_a")
def nested_pipe_stage_a(x: str) -> str:
    """A."""
    _record("stage_a")
    return x


def nested_pipe_stage_b(x: str) -> str:
    _record("stage_b_attempt")
    if not approve("nested_pipe_go"):
        return "denied:" + x
    _record("stage_b_sent")
    return "sent:" + x


the_pipe = pipe(nested_pipe_stage_a, nested_pipe_stage_b)


@flow
def nested_pipe_approval_flow(x: str) -> str:
    return the_pipe(x)
