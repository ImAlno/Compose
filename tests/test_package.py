import composeai
from composeai.models.compatible import OpenAICompatibleModel


def test_version():
    assert composeai.__version__ == "0.2.0"


def test_openai_compatible_is_exported():
    model = composeai.openai_compatible("http://localhost:11434/v1", "llama3")
    assert isinstance(model, OpenAICompatibleModel)
    assert model.model_id == "llama3"


def test_pricing_api_is_public():
    assert "register_price" in composeai.__all__
    assert "ModelPrice" in composeai.__all__
    price = composeai.ModelPrice(input=1.0, output=2.0)
    composeai.register_price("openai-compatible", "pkg-test-model", price)
