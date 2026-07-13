"""Driver B: resume the run started (and crashed) by ``subprocess_run_a.py``.

Run as ``python subprocess_run_b.py`` with the same ``COMPOSE_DIR``/
``COUNTERS_FILE`` as driver A, plus ``RUN_ID_FILE`` (to read the run id
from) and ``RESULT_FILE`` (to write the outcome to, as JSON, for the test
process to assert on -- this process's stdout isn't otherwise captured
structurally).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from subprocess_flow_defs import crash_flow  # noqa: E402, F401 -- registers "crash_flow" by name

from composeai import resume

run_id = Path(os.environ["RUN_ID_FILE"]).read_text().strip()
run = resume(run_id)

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
