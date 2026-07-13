"""Driver A: run a standalone agent until its approval-gated tool pauses it.

Run as ``python subprocess_hitl_agent_run_a.py`` with ``COMPOSE_DIR``/
``RUN_ID_FILE``/``PAUSE_RESULT_FILE`` set in the environment. Pausing is not
an error -- ``runner.run()`` returns a normal ``Run(status="paused", ...)``
and this process exits with code 0.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from subprocess_hitl_agent_defs import make_agent  # noqa: E402

from composeai.testing import FakeModel  # noqa: E402

model = FakeModel(
    [{"tool_calls": [{"name": "dangerous_tool", "arguments": {}, "id": "call_1"}]}]
)
runner = make_agent(model)

run = runner.run()

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
