"""Driver A: run ``nested_pipe_approval_flow`` until the pipe's ``approve()``
stage pauses it, then exit cleanly.

Run as ``python subprocess_nested_pipe_hitl_run_a.py`` with ``COMPOSE_DIR``/
``RUN_ID_FILE``/``COUNTERS_FILE``/``PAUSE_RESULT_FILE`` set in the
environment. Pausing is not an error -- ``nested_pipe_approval_flow.run()``
returns a normal ``Run(status="paused", ...)`` and this process exits with
code 0, same as any other Python script (mirrors
``subprocess_hitl_run_a.py``).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from subprocess_nested_pipe_hitl_flow_defs import nested_pipe_approval_flow  # noqa: E402

run = nested_pipe_approval_flow.run("hello")

Path(os.environ["RUN_ID_FILE"]).write_text(run.id)
Path(os.environ["PAUSE_RESULT_FILE"]).write_text(
    json.dumps(
        {
            "status": run.status,
            "trace_id": run.trace.trace_id,
            "pending_id": run.pending.id if run.pending is not None else None,
        }
    )
)
