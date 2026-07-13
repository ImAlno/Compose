"""Session-wide test isolation for the durable flow runtime (Phase 7).

Redirects ``COMPOSE_DIR`` at a per-session temp directory (and calls
``runs.reset_default()`` so the change takes effect immediately) before any
test runs -- so no test in the suite ever creates or writes ``./.compose``
in the repo itself, regardless of whether it uses ``composeai.runs.open_default()``
directly, or indirectly via ``@agent``/``@flow``/``@task``.
"""

import os

import pytest

from composeai import runs

# Re-exported so every test file can request `cassette` as a fixture
# parameter (`def test_x(cassette): ...`) without importing it itself --
# pytest only auto-discovers fixtures defined in (or imported into) a
# conftest.py, never ones merely imported into an arbitrary test module.
from composeai.testing import cassette  # noqa: F401


@pytest.fixture(autouse=True, scope="session")
def _redirect_compose_dir(tmp_path_factory):
    compose_dir = tmp_path_factory.mktemp("compose-dir")
    os.environ["COMPOSE_DIR"] = str(compose_dir)
    runs.reset_default()
    yield
    runs.reset_default()
    del os.environ["COMPOSE_DIR"]


@pytest.fixture(autouse=True)
def _clear_agent_registry():
    """Function-scoped, global: ``@agent`` names are unique per-process (Phase 8,
    mirroring ``@flow``'s registry -- see ``composeai.agentfn._AGENT_REGISTRY``),
    and test functions across many files reuse common agent names (``researcher``,
    ``runner``, ``greeter``, ...). Without clearing between every test, the second
    test to define e.g. ``researcher`` would raise ConfigError on decoration.
    """
    yield
    from composeai.agentfn import _AGENT_REGISTRY

    _AGENT_REGISTRY.clear()


@pytest.fixture(autouse=True)
def _clear_task_registry():
    """Function-scoped, global: ``@task`` names are unique per-process (mirroring
    ``@flow``/``@agent``'s own registries -- see ``composeai.flow._TASK_REGISTRY``),
    and test functions across many files reuse common task names (``process``,
    ``record``, ``t1``, ...). Without clearing between every test, the second
    test to define e.g. ``record`` would raise ConfigError on decoration.
    """
    yield
    from composeai.flow import _TASK_REGISTRY

    _TASK_REGISTRY.clear()
