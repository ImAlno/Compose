# Providers

`@agent(model=...)` accepts either a `"provider/model-id"` string (resolved lazily against the matching extra) or a `Model` instance you construct yourself for full control over API keys, HTTP timeouts, and — for OpenAI-compatible servers — the base URL and structured-output strategy.

## Model strings vs. `Model` instances

```python
import composeai as compose


@compose.agent(model="anthropic/claude-sonnet-5")
def researcher(topic: str) -> str:
    """You are a researcher."""
    return compose.prompt(f"Summarize: {topic}")
```

A `"anthropic/..."` or `"openai/..."` string is split on the first `/`, the matching provider module is imported lazily (so `import composeai` never requires a provider SDK to be installed), and the resulting adapter is cached per `(provider, model_id)`. If the SDK isn't installed, resolving the string raises `ConfigError` telling you which extra to install (`pip install "composeai[anthropic]"` / `pip install "composeai[openai]"`).

A `Model` instance — `AnthropicModel(...)`, `OpenAIModel(...)`, or the result of `compose.openai_compatible(...)` — passes straight through wherever a model string would go. Use one when you need anything beyond the string form's defaults: a custom `api_key`, an injected SDK `client`, a per-model `timeout`, or (for compatible servers) a `base_url`.

```python
from composeai.models.anthropic import AnthropicModel
from composeai.models.openai import OpenAIModel

anthropic_model = AnthropicModel("claude-sonnet-5", api_key="sk-...", timeout=60)
openai_model = OpenAIModel("gpt-5.6-luna", client=my_custom_client)

@compose.agent(model=anthropic_model)
def researcher(topic: str) -> str: ...
```

`AnthropicModel`/`OpenAIModel` live in `composeai.models.anthropic`/`composeai.models.openai` — they aren't re-exported from top-level `composeai`, since the string form covers the common case.

## API keys and `client=`

Both `AnthropicModel` and `OpenAIModel` read their key from the environment by default — `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — or accept one explicitly:

```python
AnthropicModel("claude-sonnet-5", api_key="sk-ant-...")
OpenAIModel("gpt-5.6-luna", api_key="sk-...")
```

Missing both raises `ConfigError` naming the environment variable and the `api_key=`/`client=` alternative. Passing your own `client=` (an already-constructed SDK client) skips the key check and construction entirely — but doesn't remove the SDK dependency: `complete()`/`stream()` still `import anthropic` / `import openai` themselves, for exception mapping, so the package must be installed either way.

`openai_compatible(...)`'s factory does **not** read any environment variable and has no `client=` parameter — local servers (Ollama, vLLM, ...) often require a non-empty key even though they don't check it, so a missing `api_key` there defaults to the placeholder `"unused"` rather than raising. If you need to inject your own SDK client for a compatible server, construct `OpenAICompatibleModel` directly (`composeai.models.compatible.OpenAICompatibleModel`) instead of going through the factory.

## HTTP `timeout=`

All three model constructors — `AnthropicModel`, `OpenAIModel`, and `openai_compatible(...)` — accept `timeout: float | None = None` (seconds), bounding **each individual HTTP request** at the SDK-client level:

```python
AnthropicModel("claude-sonnet-5", timeout=60)
OpenAIModel("gpt-5.6-luna", timeout=60)
compose.openai_compatible("http://localhost:11434/v1", "llama3", timeout=120)
```

Left unset, the SDK's own default applies (no explicit `None` is ever passed through — that would mean "no timeout at all" rather than "use the SDK's default").

This is a different mechanism from `@agent(timeout=...)`: the agent's `timeout` is a **turn-boundary watchdog**, checked only *between* LLM turns, so it can never interrupt a call already in flight. The model constructor's `timeout` is the only real guard against a genuinely hung HTTP request — set it if you need one.

## `openai_compatible`: the full tour

```python
def openai_compatible(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    timeout: float | None = None,
    input_price: float | None = None,
    output_price: float | None = None,
    schema_mode: str = "native",
) -> Model: ...
```

For any OpenAI-compatible Chat Completions server — Ollama, vLLM, LM Studio, ollama.com, a self-hosted proxy. There is no `"provider/model-id"` string form for this adapter (`base_url` is required and varies per deployment); the returned `Model` instance is what you pass to `@agent(model=...)`.

```python
model = compose.openai_compatible(
    "https://ollama.com/v1", "kimi-k2.6",
    api_key="...", timeout=120,
    input_price=0.60, output_price=2.50,
    schema_mode="auto",
)
```

- **`base_url`**, **`model`** — positional and required.
- **`api_key`** — see above; defaults to a placeholder if omitted.
- **`timeout`** — see above.
- **`schema_mode`** — one of three strategies for getting structured output back, all keyed off the agent's `output_type`:
  - **`"native"`** (the default) — sends `response_format: {"type": "json_schema", ...}` and trusts the server to honor it, same as talking to real OpenAI.
  - **`"prompt"`** — embeds the JSON Schema as an instruction in the last user message instead, and parses the reply leniently (strips markdown fences, scans for the first balanced `{...}` object). For servers that *accept* `response_format` but silently ignore it and return free-form text (ollama.com does this, for every model tested). When the lenient parse still fails, this adapter does **not** raise a provider error — it returns `parsed=None`, and the agent loop's own output extraction turns that into a *repairable* `ComposeError`, eligible for `@agent(max_repairs=...)` corrective turns rather than `retries=` (which stays provider-error territory — a dropped connection, not a malformed reply).
  - **`"auto"`** — starts native and permanently demotes to prompt mode the first time a call comes back with generic non-JSON content (the server accepted `response_format` but ignored it): that one call is retried in prompt mode, and every later call on the same model instance goes straight to prompt mode for the rest of the process — at most one wasted native call ever. `stream()` follows whichever mode is currently effective but never demotes itself (a mid-stream retry would re-emit already-yielded text deltas), so route at least one `complete()`-style (non-streaming) call through the instance first if a streaming caller should benefit from the auto-demotion. Transport errors and the reasoning-tokens-only diagnostic (below) never demote, in any mode.

  An invalid value (anything other than `"native"`/`"prompt"`/`"auto"`) raises `ConfigError` at construction time.

- **`input_price`**/**`output_price`** — USD per **million** tokens; pass both together or neither, else `ConfigError`. Registering a price is what makes `compose costs` and `Budget(usd=...)` able to see spend on a paid compatible provider (ollama.com, hosted vLLM, ...) — without one, every call reports `Usage(cost_usd=None, cost_complete=False)`, and that spend is simply invisible to a USD budget (pass a `tokens` budget too if you need a hard cap regardless of pricing).

### `register_price` / `ModelPrice`

`input_price=`/`output_price=` on `openai_compatible(...)` is sugar for calling `composeai.register_price` yourself:

```python
from composeai import register_price, ModelPrice

register_price("openai-compatible", "kimi-k2.6", ModelPrice(input=0.60, output=2.50))
```

`register_price(provider: str, model: str, price: ModelPrice) -> None` is process-global per `(provider, model)` — last writer wins. `ModelPrice(input, output, cache_read=None, cache_write_5m=None, cache_write_1h=None)` is USD per million tokens; the three cache fields default relative to `input` (10%, 1.25×, 2× respectively) when left unset. Use `register_price` directly for the string `"provider/model-id"` form, where there's no factory call to thread `input_price=`/`output_price=` through.

## Reasoning-model gotchas

Reasoning models can burn thousands of hidden tokens *before* producing any visible content. Two consequences:

- **Keep `max_tokens` generous.** A budget too tight for the model's thinking phase comes back with no visible output at all — not a slow answer, an empty one.
- **Two distinct diagnostics fire when structured output can't be decoded**, and they mean different things:

  1. *"Model returned only reasoning tokens with empty content for a structured-output request..."* — the reply's text was empty (or whitespace) **and** `usage.reasoning_tokens` was nonzero. This means the model spent its entire token budget thinking and produced nothing after — the fix is to raise `max_tokens`, or switch to `schema_mode="prompt"` if the server doesn't enforce constrained decoding for this model at all. This diagnostic never demotes `schema_mode="auto"` and raises the same way in every mode.
  2. A generic *"Server returned non-JSON content for a structured-output request..."* — there *is* text, but it doesn't parse as JSON. In `"native"` mode this is a hard `ProviderError` (not repairable — `retries=` territory only). In `"auto"` mode, this exact condition is what triggers the one-time, permanent demotion to prompt mode described above. In `"prompt"` mode, a parse failure never reaches this error at all — it falls through to the repairable-`ComposeError`/`max_repairs=` path instead.

## Request config: prompt caching, thinking, effort

Three `@agent` knobs (also fields on `ModelRequest` for hand-built
requests) map per provider:

| Knob | Anthropic | OpenAI (Responses) | openai_compatible |
|---|---|---|---|
| `prompt_cache=True` (default) | `cache_control` breakpoints: one on the system block (caches tools+system across runs), one on the conversation tail once multi-turn (caches the growing tool-loop conversation) | no-op — OpenAI caches automatically server-side; cached tokens already show up in `Usage.cache_read_tokens` | no-op |
| `thinking=True` / `False` | `{"type": "adaptive", "display": "summarized"}` / `{"type": "disabled"}` | no-op (reasoning models have no toggle) | no-op |
| `effort="..."` | `output_config.effort` (`"low"`…`"max"`) | `reasoning.effort` (`"minimal"`…`"high"`) | no-op |

Notes:

- Defaults send **nothing** for `thinking`/`effort` — each model's own
  default applies, and composeai keeps no per-model capability table.
  A shape the model rejects (e.g. `thinking=False` on a model with
  always-on thinking) surfaces as a `ProviderError`.
- Thinking spend is reported in `Usage.reasoning_tokens` (a subset of
  `output_tokens`, split out for visibility).
- Cache billing: reads ~0.1× input price, 5-minute-TTL writes ~1.25×
  (already reflected in `compose costs` via the `ModelPrice` cache
  fields documented above). Prompts below a model's minimum cacheable
  length are silently not cached — no error, no write premium.
- `prompt_cache` never enters request hashes, so cassettes and
  `@agent(cache=True)` entries recorded on 0.5.x keep replaying.

## See also

[agents](agents.md) covers `@agent(model=..., timeout=..., fallback=...)` and the rest of the resilience knobs; [budgets](budgets.md) covers `Budget(usd=...)` and why unpriced spend can't be capped by it; [observability](observability.md) covers `compose costs --by model` for seeing what a run actually spent.
