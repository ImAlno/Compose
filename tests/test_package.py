import composeai
from composeai.models.compatible import OpenAICompatibleModel


def test_version():
    assert composeai.__version__ == "0.1.0.dev0"


def test_openai_compatible_is_exported():
    model = composeai.openai_compatible("http://localhost:11434/v1", "llama3")
    assert isinstance(model, OpenAICompatibleModel)
    assert model.model_id == "llama3"
