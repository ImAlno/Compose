import pytest

from composeai.agentfn import agent
from composeai.errors import ConfigError
from composeai.flow import flow, task
from composeai.models import registry
from composeai.models.base import ModelRequest, ModelResponse
from composeai.testing import FakeModel


@pytest.fixture(autouse=True)
def _reset_registry_state():
    # The registry caches resolved adapters and registered factories in
    # module-level dicts; keep tests independent of call order.
    saved_modules = dict(registry._PROVIDER_MODULES)
    saved_factories = dict(registry._PROVIDER_FACTORIES)
    saved_cache = dict(registry._CACHE)
    yield
    registry._PROVIDER_MODULES.clear()
    registry._PROVIDER_MODULES.update(saved_modules)
    registry._PROVIDER_FACTORIES.clear()
    registry._PROVIDER_FACTORIES.update(saved_factories)
    registry._CACHE.clear()
    registry._CACHE.update(saved_cache)


class _StubModel:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError


# --- parse_model_string ---


def test_parse_model_string_splits_provider_and_model():
    assert registry.parse_model_string("anthropic/claude-sonnet-5") == (
        "anthropic",
        "claude-sonnet-5",
    )


def test_parse_model_string_keeps_only_first_slash_in_provider_split():
    # Model ids can themselves contain slashes (e.g. openai-compatible paths).
    assert registry.parse_model_string("anthropic/foo/bar") == ("anthropic", "foo/bar")


def test_parse_model_string_no_slash_raises_config_error():
    with pytest.raises(ConfigError):
        registry.parse_model_string("claude-sonnet-5")


def test_parse_model_string_unknown_provider_raises_config_error():
    with pytest.raises(ConfigError, match="unknown-provider"):
        registry.parse_model_string("unknown-provider/some-model")


def test_parse_model_string_error_lists_known_providers():
    with pytest.raises(ConfigError, match="anthropic"):
        registry.parse_model_string("bogus/model")


# --- register_provider ---


def test_register_provider_makes_provider_resolvable():
    registry.register_provider("stub", _StubModel)
    model = registry.resolve("stub/my-model")
    assert isinstance(model, _StubModel)
    assert model.model_id == "my-model"


# --- resolve: instance passthrough ---


def test_resolve_passes_through_existing_model_instance():
    instance = _StubModel("whatever")
    assert registry.resolve(instance) is instance


def test_resolve_rejects_non_string_non_model():
    with pytest.raises(ConfigError):
        registry.resolve(12345)  # type: ignore[arg-type]


# --- resolve: caching ---


def test_resolve_caches_per_provider_and_model_id():
    registry.register_provider("stub", _StubModel)
    first = registry.resolve("stub/cached-model")
    second = registry.resolve("stub/cached-model")
    assert first is second


def test_resolve_does_not_share_cache_across_different_model_ids():
    registry.register_provider("stub", _StubModel)
    a = registry.resolve("stub/model-a")
    b = registry.resolve("stub/model-b")
    assert a is not b


# --- lazy import failure ---


def test_lazy_import_failure_raises_config_error_naming_extra(monkeypatch):
    monkeypatch.setitem(
        registry._PROVIDER_MODULES, "anthropic", "composeai.models._does_not_exist_xyz"
    )
    with pytest.raises(ConfigError, match=r"pip install 'composeai\[anthropic\]'"):
        registry.resolve("anthropic/claude-haiku-4-5")


def test_provider_modules_are_not_imported_until_first_resolve():
    import sys

    sys.modules.pop("composeai.models.anthropic", None)
    assert "composeai.models.anthropic" not in sys.modules
    registry.resolve("anthropic/claude-haiku-4-5")
    assert "composeai.models.anthropic" in sys.modules


# --- openai provider wiring ---


def test_openai_provider_is_registered():
    assert "openai" in registry._known_providers()


def test_resolve_openai_model_string_builds_openai_model():
    from composeai.models.openai import OpenAIModel

    model = registry.resolve("openai/gpt-5.4")
    assert isinstance(model, OpenAIModel)
    assert model.model_id == "gpt-5.4"


def test_openai_lazy_import_failure_raises_config_error_naming_extra(monkeypatch):
    monkeypatch.setitem(
        registry._PROVIDER_MODULES, "openai", "composeai.models._does_not_exist_xyz"
    )
    with pytest.raises(ConfigError, match=r"pip install 'composeai\[openai\]'"):
        registry.resolve("openai/gpt-5.4")


def test_openai_module_is_not_imported_until_first_resolve():
    import sys

    sys.modules.pop("composeai.models.openai", None)
    assert "composeai.models.openai" not in sys.modules
    registry.resolve("openai/gpt-5.4")
    assert "composeai.models.openai" in sys.modules


# --- Task 11: reset_registries() / @agent(name=)/replace= / @flow/@task replace= ---


def test_reset_registries_allows_redefinition():
    from composeai.testing import reset_registries

    @task
    def task11_dup(x: int) -> int:  # pyright: ignore[reportRedeclaration]
        return x

    reset_registries()

    @task
    def task11_dup(x: int) -> int:  # noqa: F811 -- redefinition is the point
        return x + 1

    assert task11_dup(1) == 2


def test_agent_name_override_registers_under_custom_name():
    from composeai.agentfn import _AGENT_REGISTRY

    model = FakeModel(["hi"])

    @agent(model=model, name="task11_custom_name")
    def task11_some_fn(question: str) -> str:
        return question

    assert task11_some_fn.name == "task11_custom_name"
    assert _AGENT_REGISTRY["task11_custom_name"] is task11_some_fn


def test_agent_replace_true_rebinds():
    from composeai.agentfn import _AGENT_REGISTRY

    m1 = FakeModel(["one"])
    m2 = FakeModel(["two"])

    @agent(model=m1)
    def task11_rebind(question: str) -> str:  # pyright: ignore[reportRedeclaration]
        return question

    @agent(model=m2, replace=True)
    def task11_rebind(question: str) -> str:  # noqa: F811
        return question

    assert task11_rebind("x") == "two"
    assert _AGENT_REGISTRY["task11_rebind"] is task11_rebind


def test_duplicate_agent_name_still_raises_without_replace():
    model = FakeModel([])

    @agent(model=model)
    def task11_unique(question: str) -> str:
        return question

    with pytest.raises(ConfigError):

        @agent(model=model)  # noqa: F811
        def task11_unique(question: str) -> str:
            return question


def test_flow_and_task_replace_true():
    from composeai.flow import _FLOW_REGISTRY, _TASK_REGISTRY

    @flow
    def task11_reflow() -> str:  # pyright: ignore[reportRedeclaration]
        return "a"

    @flow(replace=True)
    def task11_reflow() -> str:  # noqa: F811
        return "b"

    assert task11_reflow() == "b"
    assert _FLOW_REGISTRY["task11_reflow"] is task11_reflow

    @task
    def task11_retask() -> str:  # pyright: ignore[reportRedeclaration]
        return "a"

    @task(replace=True)
    def task11_retask() -> str:  # noqa: F811
        return "b"

    assert task11_retask() == "b"
    assert _TASK_REGISTRY["task11_retask"] is task11_retask
