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
def _reset_registries():
    """Function-scoped, global: @agent/@flow/@task names are unique per
    process, and tests across files reuse common names. Uses the public
    helper -- dogfooding composeai.testing.reset_registries()."""
    yield
    from composeai.testing import reset_registries

    reset_registries()
