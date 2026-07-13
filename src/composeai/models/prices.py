"""Per-(provider, model) USD-per-million-token pricing and cost math.

Cost is never fabricated: callers that can't find a price for a
``(provider, model)`` pair get ``None`` back from :func:`get_price` and
must set ``Usage(cost_usd=None, cost_complete=False)`` themselves rather
than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..messages import Usage

PRICES_AS_OF = "2026-07-12"

_CACHE_READ_FACTOR = 0.1
_CACHE_WRITE_5M_FACTOR = 1.25
_CACHE_WRITE_1H_FACTOR = 2.0


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD price per *million* tokens for one model.

    ``cache_read``/``cache_write_5m``/``cache_write_1h`` default relative
    to ``input`` when left ``None`` -- see :func:`resolve_price`, which is
    the single place those defaults are computed.
    """

    input: float
    output: float
    cache_read: float | None = None
    cache_write_5m: float | None = None
    cache_write_1h: float | None = None


def resolve_price(price: ModelPrice) -> ModelPrice:
    """Return a copy of ``price`` with all cache fields concretely filled in."""
    return ModelPrice(
        input=price.input,
        output=price.output,
        cache_read=(
            price.cache_read if price.cache_read is not None else price.input * _CACHE_READ_FACTOR
        ),
        cache_write_5m=(
            price.cache_write_5m
            if price.cache_write_5m is not None
            else price.input * _CACHE_WRITE_5M_FACTOR
        ),
        cache_write_1h=(
            price.cache_write_1h
            if price.cache_write_1h is not None
            else price.input * _CACHE_WRITE_1H_FACTOR
        ),
    )


# Anthropic sticker prices (USD per MTok, input/output) as of PRICES_AS_OF.
# claude-sonnet-5 has an introductory promotional price in effect through
# 2026-08-31 (discounted from the figures below); this table intentionally
# carries the post-intro sticker price, not the promo price.
_PRICES: dict[tuple[str, str], ModelPrice] = {
    ("anthropic", "claude-fable-5"): ModelPrice(input=10.0, output=50.0),
    ("anthropic", "claude-opus-4-8"): ModelPrice(input=5.0, output=25.0),
    ("anthropic", "claude-opus-4-7"): ModelPrice(input=5.0, output=25.0),
    ("anthropic", "claude-opus-4-6"): ModelPrice(input=5.0, output=25.0),
    ("anthropic", "claude-sonnet-5"): ModelPrice(input=3.0, output=15.0),
    ("anthropic", "claude-sonnet-4-6"): ModelPrice(input=3.0, output=15.0),
    ("anthropic", "claude-haiku-4-5"): ModelPrice(input=1.0, output=5.0),
    # OpenAI sticker prices (USD per MTok, input/output/cached-input)
    # verified against https://platform.openai.com/docs/pricing (redirects
    # to https://developers.openai.com/api/docs/pricing) on 2026-07-12 --
    # `cache_read` is the *published* cached-input rate for each model, not
    # the 10%-of-input default `resolve_price` would otherwise compute (it
    # happens to equal that ratio for every model below, but is set
    # explicitly per the project's "never guess" rule rather than left to
    # fall back). gpt-5.5-pro / gpt-5.4-pro don't publish a discounted
    # cached-input rate, so `cache_read` is left unset for those two (falls
    # back to the 10% default, which is moot since those tiers report no
    # cached tokens in practice).
    ("openai", "gpt-5.6-sol"): ModelPrice(input=5.0, output=30.0, cache_read=0.50),
    ("openai", "gpt-5.6-terra"): ModelPrice(input=2.5, output=15.0, cache_read=0.25),
    ("openai", "gpt-5.6-luna"): ModelPrice(input=1.0, output=6.0, cache_read=0.10),
    ("openai", "gpt-5.5"): ModelPrice(input=5.0, output=30.0, cache_read=0.50),
    ("openai", "gpt-5.5-pro"): ModelPrice(input=30.0, output=180.0),
    ("openai", "gpt-5.4"): ModelPrice(input=2.5, output=15.0, cache_read=0.25),
    ("openai", "gpt-5.4-mini"): ModelPrice(input=0.75, output=4.5, cache_read=0.075),
    ("openai", "gpt-5.4-nano"): ModelPrice(input=0.20, output=1.25, cache_read=0.02),
    ("openai", "gpt-5.4-pro"): ModelPrice(input=30.0, output=180.0),
    ("openai", "gpt-5.3-codex"): ModelPrice(input=1.75, output=14.0, cache_read=0.175),
}


def get_price(provider: str, model: str) -> ModelPrice | None:
    """Look up the price for ``(provider, model)``, or ``None`` if unknown."""
    return _PRICES.get((provider, model))


def register_price(provider: str, model: str, price: ModelPrice) -> None:
    """Register (or override) the price for ``(provider, model)``."""
    _PRICES[(provider, model)] = price


def compute_cost(
    usage: Usage,
    price: ModelPrice,
    cache_write_5m_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
) -> float:
    """USD cost of ``usage`` at ``price``.

    ``cache_write_5m_tokens``/``cache_write_1h_tokens`` are the per-TTL
    cache-creation token counts (providers that don't split by TTL should
    pass the full amount as 5m tokens and leave 1h at 0). Never fabricates:
    callers are responsible for checking :func:`get_price` returned a price
    before calling this.
    """
    resolved = resolve_price(price)
    total = (
        usage.input_tokens * resolved.input
        + usage.output_tokens * resolved.output
        + usage.cache_read_tokens * (resolved.cache_read or 0.0)
        + cache_write_5m_tokens * (resolved.cache_write_5m or 0.0)
        + cache_write_1h_tokens * (resolved.cache_write_1h or 0.0)
    )
    return total / 1_000_000
