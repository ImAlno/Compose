import pytest

from composeai.messages import Usage
from composeai.models.prices import (
    PRICES_AS_OF,
    ModelPrice,
    compute_cost,
    get_price,
    register_price,
    resolve_price,
)

# --- ModelPrice cache defaults ---


def test_cache_read_defaults_to_tenth_of_input():
    price = resolve_price(ModelPrice(input=10.0, output=50.0))
    assert price.cache_read == 1.0


def test_cache_write_5m_defaults_to_1_25x_input():
    price = resolve_price(ModelPrice(input=10.0, output=50.0))
    assert price.cache_write_5m == 12.5


def test_cache_write_1h_defaults_to_2x_input():
    price = resolve_price(ModelPrice(input=10.0, output=50.0))
    assert price.cache_write_1h == 20.0


def test_explicit_cache_prices_are_not_overridden():
    price = resolve_price(
        ModelPrice(input=10.0, output=50.0, cache_read=2.0, cache_write_5m=3.0, cache_write_1h=4.0)
    )
    assert price.cache_read == 2.0
    assert price.cache_write_5m == 3.0
    assert price.cache_write_1h == 4.0


def test_prices_as_of_is_a_date_string():
    assert PRICES_AS_OF == "2026-07-12"


# --- Price table ---


@pytest.mark.parametrize(
    "model_id,expected_input,expected_output",
    [
        ("claude-fable-5", 10, 50),
        ("claude-opus-4-8", 5, 25),
        ("claude-opus-4-7", 5, 25),
        ("claude-opus-4-6", 5, 25),
        ("claude-sonnet-5", 3, 15),
        ("claude-sonnet-4-6", 3, 15),
        ("claude-haiku-4-5", 1, 5),
    ],
)
def test_known_anthropic_prices(model_id, expected_input, expected_output):
    price = get_price("anthropic", model_id)
    assert price is not None
    assert price.input == expected_input
    assert price.output == expected_output


def test_unknown_model_returns_none():
    assert get_price("anthropic", "claude-nonexistent-9") is None
    assert get_price("some-other-provider", "claude-sonnet-5") is None


@pytest.mark.parametrize(
    "model_id,expected_input,expected_output,expected_cache_read",
    [
        ("gpt-5.6-sol", 5.0, 30.0, 0.50),
        ("gpt-5.6-terra", 2.5, 15.0, 0.25),
        ("gpt-5.6-luna", 1.0, 6.0, 0.10),
        ("gpt-5.5", 5.0, 30.0, 0.50),
        ("gpt-5.4", 2.5, 15.0, 0.25),
        ("gpt-5.4-mini", 0.75, 4.5, 0.075),
        ("gpt-5.4-nano", 0.20, 1.25, 0.02),
        ("gpt-5.3-codex", 1.75, 14.0, 0.175),
    ],
)
def test_known_openai_prices(model_id, expected_input, expected_output, expected_cache_read):
    price = get_price("openai", model_id)
    assert price is not None
    assert price.input == expected_input
    assert price.output == expected_output
    assert price.cache_read == expected_cache_read


@pytest.mark.parametrize("model_id", ["gpt-5.5-pro", "gpt-5.4-pro"])
def test_openai_pro_tier_prices_have_no_explicit_cache_read(model_id):
    # These tiers don't publish a discounted cached-input rate, so
    # cache_read is intentionally left unset (falls back to the 10%
    # default via resolve_price, per the "never guess" rule -- we don't
    # fabricate a cache_read value that wasn't published).
    price = get_price("openai", model_id)
    assert price is not None
    assert price.cache_read is None


def test_openai_compatible_provider_has_no_builtin_prices():
    # Cost for user-hosted servers is always None/incomplete unless the
    # caller explicitly registers a price for their own model name.
    assert get_price("openai-compatible", "llama3") is None


# --- register_price override hook ---


def test_register_price_adds_new_entry(monkeypatch):
    register_price("test-provider", "test-model", ModelPrice(input=1.0, output=2.0))
    price = get_price("test-provider", "test-model")
    assert price is not None
    assert price.input == 1.0
    assert price.output == 2.0


def test_register_price_overrides_existing_entry():
    original = get_price("anthropic", "claude-haiku-4-5")
    assert original is not None
    try:
        register_price("anthropic", "claude-haiku-4-5", ModelPrice(input=999.0, output=999.0))
        overridden = get_price("anthropic", "claude-haiku-4-5")
        assert overridden is not None
        assert overridden.input == 999.0
    finally:
        register_price("anthropic", "claude-haiku-4-5", original)


# --- compute_cost math ---


def test_compute_cost_input_and_output_only():
    price = ModelPrice(input=10.0, output=50.0)
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = compute_cost(usage, price, 0, 0)
    assert cost == pytest.approx(10.0 + 50.0)


def test_compute_cost_includes_cache_read():
    price = ModelPrice(input=10.0, output=50.0)  # cache_read defaults to 1.0/MTok
    usage = Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
    cost = compute_cost(usage, price, 0, 0)
    assert cost == pytest.approx(1.0)


def test_compute_cost_includes_both_cache_write_ttls():
    price = ModelPrice(input=10.0, output=50.0)  # 5m=12.5/MTok, 1h=20.0/MTok
    usage = Usage()
    cost = compute_cost(usage, price, 1_000_000, 1_000_000)
    assert cost == pytest.approx(12.5 + 20.0)


def test_compute_cost_full_mix():
    price = ModelPrice(input=10.0, output=50.0)
    usage = Usage(input_tokens=500_000, output_tokens=200_000, cache_read_tokens=100_000)
    cost = compute_cost(usage, price, 50_000, 25_000)
    expected = (
        500_000 * 10.0 + 200_000 * 50.0 + 100_000 * 1.0 + 50_000 * 12.5 + 25_000 * 20.0
    ) / 1_000_000
    assert cost == pytest.approx(expected)


def test_compute_cost_defaults_cache_write_args_to_zero():
    price = ModelPrice(input=10.0, output=50.0)
    usage = Usage(input_tokens=1_000_000)
    assert compute_cost(usage, price) == pytest.approx(10.0)
