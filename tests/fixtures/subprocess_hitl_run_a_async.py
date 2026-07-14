"""Async twin of ``subprocess_hitl_run_a.py`` -- driver A, but through the
async surface: ``asyncio.run(approval_flow.arun())`` instead of the sync
``.run()``. Same ``COMPOSE_DIR``/``RUN_ID_FILE``/``COUNTERS_FILE``/
``PAUSE_RESULT_FILE`` contract as the sync driver, and reuses the identical
flow definition (``subprocess_hitl_flow_defs.approval_flow``, a plain
sync-bodied ``@flow`` -- ``Flow.arun`` dispatches a sync body via
``_dispatch.run_stage``, same as ``.run()``; see ``_aexecute_flow``'s
docstring) -- only the driver call differs. Pausing is not an error here
either -- ``approval_flow.arun()`` returns a normal ``Run(status="paused",
...)`` and this process exits with code 0.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from subprocess_hitl_flow_defs import approval_flow  # noqa: E402

run = asyncio.run(approval_flow.arun())

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
