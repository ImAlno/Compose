"""Driver B: resume ``nested_pipe_approval_flow`` with the approval answer,
in a brand-new process.

Run as ``python subprocess_nested_pipe_hitl_run_b.py`` with the same
``COMPOSE_DIR``/``COUNTERS_FILE`` as driver A, plus ``RUN_ID_FILE`` (to read
the run id from) and ``RESULT_FILE`` (to write the outcome to, as JSON).
Mirrors ``subprocess_hitl_run_b.py``.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from subprocess_nested_pipe_hitl_flow_defs import nested_pipe_approval_flow  # noqa: E402, F401

from composeai import resume

run_id = Path(os.environ["RUN_ID_FILE"]).read_text().strip()
run = resume(run_id, {"nested_pipe_go": True})

Path(os.environ["RESULT_FILE"]).write_text(
    json.dumps(
        {
            "output": run.output,
            "status": run.status,
            "trace_id": run.trace.trace_id,
            "run_id": run.id,
        }
    )
)
