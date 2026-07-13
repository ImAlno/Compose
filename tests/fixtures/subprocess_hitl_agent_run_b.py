"""Driver B: resume the paused standalone agent, in a brand-new process.

Run as ``python subprocess_hitl_agent_run_b.py`` with the same
``COMPOSE_DIR`` as driver A, plus ``RUN_ID_FILE`` (to read the run id from)
and ``RESULT_FILE`` (to write the outcome to, as JSON). Registers the *same*
agent (by name) in this fresh process's own registry so ``resume()`` can
route to it, and restores its conversation from the durable ``agent_state``
snapshot -- never re-asking the model the tool-call turn that already
happened in process A.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from subprocess_hitl_agent_defs import make_agent  # noqa: E402

from composeai import resume  # noqa: E402
from composeai.testing import FakeModel  # noqa: E402

model = FakeModel(["final"])
make_agent(model)  # registers "runner" in this process's own _AGENT_REGISTRY

run_id = Path(os.environ["RUN_ID_FILE"]).read_text().strip()
run = resume(run_id, {"tool:dangerous_tool:call_1": True})

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
