import composeai
from composeai.models.compatible import OpenAICompatibleModel
from composeai.testing import FakeModel


def test_version():
    assert composeai.__version__ == "0.7.0"


def test_openai_compatible_is_exported():
    model = composeai.openai_compatible("http://localhost:11434/v1", "llama3")
    assert isinstance(model, OpenAICompatibleModel)
    assert model.model_id == "llama3"


def test_pricing_api_is_public():
    assert "register_price" in composeai.__all__
    assert "ModelPrice" in composeai.__all__
    price = composeai.ModelPrice(input=1.0, output=2.0)
    composeai.register_price("openai-compatible", "pkg-test-model", price)


# --- v0.4.0 Plan B, Task 9: exports audit --------------------------------


def test_async_surface_names_are_exported_and_all_is_sorted():
    """The four v0.4.0 Plan B async free functions (amap, anow, arandom,
    aresume) are public and ``__all__`` stays alphabetically sorted --
    the whole point of a flat, sorted export list is that a diff adding a
    name out of place is instantly visible in review."""
    for name in ("amap", "anow", "arandom", "aresume"):
        assert name in composeai.__all__, f"{name!r} missing from composeai.__all__"
    assert composeai.__all__ == sorted(composeai.__all__)


def test_decorated_agent_exposes_arun_and_astream():
    """Every ``@agent``-decorated function is an ``AgentFunction`` -- the
    public async surface (``arun``/``astream``, v0.4.0 Plan B, Task 4) must
    be present on it alongside the pre-existing sync ``run``/``stream``."""
    model = FakeModel(["ok"])

    @composeai.agent(model=model, name="pkg_export_async_surface_agent")
    def greeter(name: str) -> str:
        """You are a friendly greeter."""
        return f"Greet {name}"

    assert hasattr(greeter, "arun")
    assert callable(greeter.arun)
    assert hasattr(greeter, "astream")
    assert callable(greeter.astream)


def test_chat_surface_is_exported():
    import composeai

    assert hasattr(composeai, "Chat")
    assert hasattr(composeai, "chat")
    assert hasattr(composeai, "load_chat")
    assert "Chat" in composeai.__all__
    assert "chat" in composeai.__all__
    assert "load_chat" in composeai.__all__
    assert composeai.__version__ == "0.7.0"
