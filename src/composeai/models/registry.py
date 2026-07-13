"""Resolve ``"provider/model"`` strings (or passthrough ``Model`` instances) to adapters.

Provider modules are imported lazily -- ``composeai.models.anthropic`` is
only imported the first time an ``"anthropic/..."`` model string is
resolved, so importing :mod:`composeai` never requires any provider SDK to
be installed.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from ..errors import ConfigError
from .base import Model

ProviderFactory = Callable[[str], Model]

# provider name -> dotted module path to import lazily on first resolve.
_PROVIDER_MODULES: dict[str, str] = {
    "anthropic": "composeai.models.anthropic",
    "openai": "composeai.models.openai",
}

# provider name -> factory, populated either by a successful lazy import
# (which caches the provider module's `create_model`) or by an explicit
# `register_provider` call.
_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {}

# (provider, model_id) -> resolved adapter instance.
_CACHE: dict[tuple[str, str], Model] = {}


def _known_providers() -> list[str]:
    return sorted(set(_PROVIDER_MODULES) | set(_PROVIDER_FACTORIES))


def parse_model_string(model: str) -> tuple[str, str]:
    """Split ``"provider/model-id"`` into ``(provider, model_id)``.

    The model id may itself contain slashes (only the first one is
    significant); the provider must be one of the known providers.
    """
    if "/" not in model:
        raise ConfigError(
            f"Invalid model string {model!r}: expected 'provider/model-id'. "
            f"Known providers: {', '.join(_known_providers())}"
        )
    provider, _, model_id = model.partition("/")
    if not provider or not model_id:
        raise ConfigError(
            f"Invalid model string {model!r}: expected 'provider/model-id'. "
            f"Known providers: {', '.join(_known_providers())}"
        )
    if provider not in _PROVIDER_MODULES and provider not in _PROVIDER_FACTORIES:
        raise ConfigError(
            f"Unknown provider {provider!r} in model string {model!r}. "
            f"Known providers: {', '.join(_known_providers())}"
        )
    return provider, model_id


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register (or override) a provider factory: ``model_id -> Model``.

    Used by Phase 4 for the openai/compatible providers, and by tests.
    """
    _PROVIDER_FACTORIES[name] = factory


def resolve(model: str | Model) -> Model:
    """Resolve ``model`` to a :class:`~composeai.models.base.Model` instance.

    An existing ``Model`` instance passes through unchanged. A
    ``"provider/model-id"`` string resolves the provider (importing its
    module lazily on first use) and is cached per ``(provider, model_id)``.
    """
    if isinstance(model, str):
        provider, model_id = parse_model_string(model)
        cache_key = (provider, model_id)
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached
        factory = _PROVIDER_FACTORIES.get(provider)
        if factory is None:
            factory = _load_provider_factory(provider)
        instance = factory(model_id)
        _CACHE[cache_key] = instance
        return instance
    if isinstance(model, Model):
        return model
    raise ConfigError(
        f"model must be a 'provider/model-id' string or a Model instance, got {type(model)!r}"
    )


def _load_provider_factory(provider: str) -> ProviderFactory:
    module_name = _PROVIDER_MODULES[provider]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"Install the provider SDK: pip install 'composeai[{provider}]'") from exc
    factory: ProviderFactory = module.create_model
    _PROVIDER_FACTORIES[provider] = factory
    return factory
