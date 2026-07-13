"""Driver A: start ``crash_flow``, journal 2 steps, then hard-crash.

Run as ``python subprocess_run_a.py`` with ``COMPOSE_DIR``/``RUN_ID_FILE``/
``COUNTERS_FILE`` set in the environment and ``COMPOSE_TEST_CRASH=1``. Never
returns normally -- ``crash_flow`` calls ``os._exit(17)`` itself partway
through, so this process's exit code is 17, not whatever Python would have
produced from a normal return.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from subprocess_flow_defs import crash_flow  # noqa: E402

crash_flow.run()
# Unreachable: crash_flow._fn calls os._exit(17) before returning.
