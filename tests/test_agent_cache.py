"""Tests for ``@agent(cache=True)`` -- the filesystem-backed response cache
(Phase 9's dev/test kit, ``composeai.testing.CachingModel``).

Applies to ``complete()`` only: streaming bypasses the cache entirely (see
``test_cache_streaming_bypasses_cache_and_writes_no_file``), and a cache
hit reports ``Usage()`` (zero) on its llm span -- never the original,
already-paid-for tokens.

Every test gets its own isolated ``COMPOSE_DIR`` (autouse fixture below):
the cache key is a pure hash of the request (model/system/messages/...),
with no per-test salt, so two tests defining same-named/same-prompt agents
would otherwise collide against the session-wide default store (see
``tests/conftest.py``) and read back a *previous test's* cached response.
"""

from __future__ import annotations

import pytest

from composeai import agent, runs
from composeai.testing import FakeModel


@pytest.fixture(autouse=True)
def _isolated_compose_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOSE_DIR", str(tmp_path))
    runs.reset_default()
    yield
    runs.reset_default()


def test_cache_hit_on_second_identical_call_reuses_first_response():
    fake = FakeModel(["only one scripted response"])

    @agent(model=fake, cache=True)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    run1 = greeter.run("world")
    run2 = greeter.run("world")  # same prompt -> same full_hash -> cache hit

    assert run1.output == "only one scripted response"
    assert run2.output == "only one scripted response"
    assert len(fake.requests) == 1  # the script was only ever consumed once


def test_cache_miss_on_different_prompt():
    fake = FakeModel(["first", "second"])

    @agent(model=fake, cache=True)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    run1 = greeter.run("world")
    run2 = greeter.run("someone else")  # different prompt -> different full_hash -> miss

    assert run1.output == "first"
    assert run2.output == "second"
    assert len(fake.requests) == 2


def test_cache_hit_span_marked_cached_with_zero_usage():
    fake = FakeModel(["scripted"])

    @agent(model=fake, cache=True)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    run1 = greeter.run("world")
    run2 = greeter.run("world")

    llm_spans_1 = [s for s in run1.trace.spans if s.kind == "llm"]
    llm_spans_2 = [s for s in run2.trace.spans if s.kind == "llm"]
    assert len(llm_spans_1) == 1
    assert len(llm_spans_2) == 1

    assert llm_spans_1[0].attributes.get("cached") is not True
    assert llm_spans_2[0].attributes.get("cached") is True

    assert llm_spans_2[0].usage is not None
    assert llm_spans_2[0].usage.input_tokens == 0
    assert llm_spans_2[0].usage.output_tokens == 0
    # Never re-report the original tokens on the run's own rollup either.
    assert run2.usage.input_tokens == 0
    assert run2.usage.output_tokens == 0
    # The first (real) call *did* pay for tokens.
    assert run1.usage.input_tokens > 0


def test_cache_file_present_after_miss(tmp_path):
    fake = FakeModel(["cached text"])

    @agent(model=fake, cache=True)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("world")

    cache_dir = tmp_path / "cache"
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1


def test_cache_streaming_bypasses_cache_and_writes_no_file(tmp_path):
    fake = FakeModel(["first", "second"])

    @agent(model=fake, cache=True)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    list(greeter.stream("world"))
    list(greeter.stream("world"))  # same prompt, but streaming -- must NOT hit the cache

    assert len(fake.requests) == 2
    cache_dir = tmp_path / "cache"
    assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))


def test_cache_false_by_default_never_touches_filesystem_cache(tmp_path):
    fake = FakeModel(["a", "b"])

    @agent(model=fake)  # cache=False (default)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("world")
    greeter.run("world")

    assert len(fake.requests) == 2  # no caching at all
    cache_dir = tmp_path / "cache"
    assert not cache_dir.exists()


def test_cache_write_leaves_no_stray_tmp_files(tmp_path):
    """Regression: cache writes are now atomic (tmp file + os.replace)
    rather than a plain `write_text` directly on the target path -- verify
    no leftover `.tmp` file is left behind after a normal write, and only
    the final `<hash>.json` file remains."""
    fake = FakeModel(["cached text"])

    @agent(model=fake, cache=True)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    greeter.run("world")

    cache_dir = tmp_path / "cache"
    files = list(cache_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"
    assert ".tmp" not in files[0].name


def test_cache_does_not_serve_across_different_providers_sharing_a_bare_model_id(
    monkeypatch,
):
    """Regression: `compute_full_hash` used to hash only the *bare* model id
    (the registry already strips the provider prefix before building a
    `ModelRequest`) -- two agents on different providers that happen to
    share a bare model-id string, with the same system/messages, would
    silently be served each other's cached response from the shared
    process-wide cache dir. `ModelRequest.provider` now disambiguates them.
    """
    from composeai.models import registry

    fake_a = FakeModel(["from provider A"])
    fake_b = FakeModel(["from provider B"])

    monkeypatch.setitem(registry._PROVIDER_FACTORIES, "sentinel-cache-a", lambda mid: fake_a)
    monkeypatch.setitem(registry._PROVIDER_FACTORIES, "sentinel-cache-b", lambda mid: fake_b)

    @agent(model="sentinel-cache-a/shared-id", cache=True)
    def agent_a(name: str) -> str:
        """Same system prompt on purpose."""
        return name

    @agent(model="sentinel-cache-b/shared-id", cache=True)
    def agent_b(name: str) -> str:
        """Same system prompt on purpose."""
        return name

    run_a = agent_a.run("same prompt")
    run_b = agent_b.run("same prompt")

    assert run_a.output == "from provider A"
    assert run_b.output == "from provider B"  # not served provider A's cached response
    assert len(fake_a.requests) == 1
    assert len(fake_b.requests) == 1  # provider B's model was actually called, not skipped
