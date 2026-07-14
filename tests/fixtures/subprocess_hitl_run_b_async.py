"""Async twin of ``subprocess_hitl_run_b.py`` -- driver B, but through the
async surface: ``asyncio.run(aresume(run_id, {"go": True}))`` instead of
the sync ``resume(...)``. Same ``COMPOSE_DIR``/``COUNTERS_FILE``/
``RUN_ID_FILE``/``RESULT_FILE`` contract as the sync driver, and reuses the
same flow definition module so this brand-new process registers the
identical ``approval_flow`` (same name, same fingerprint) that process A
paused.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from subprocess_hitl_flow_defs import approval_flow  # noqa: E402, F401 -- registers "approval_flow"

from composeai import aresume

run_id = Path(os.environ["RUN_ID_FILE"]).read_text().strip()
run = asyncio.run(aresume(run_id, {"go": True}))

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
