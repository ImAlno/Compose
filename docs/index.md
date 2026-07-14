# composeai

composeai is a radically simple, functional framework for multi-agent AI workflows: agents are typed Python functions, composed with plain combinators, and everything that runs is traced to disk automatically.

## What composeai is

Agents are typed functions — a docstring is the system prompt, the function body returns the user prompt, and the return annotation is the structured output type, validated back into that type (or raised) when the call completes. Composition is checked before it runs — `pipe(researcher, copywriter)` verifies every stage boundary at build time, so a wiring bug raises `CompositionTypeError` before a single API call is made — and `@flow` makes a whole pipeline durable, journaling every step so a crash or a paused human-in-the-loop interrupt can `resume()` later without re-paying for finished work. Every run is traced locally and automatically: spans, token usage, and cost persist to a SQLite store on disk with no setup and no opt-in.

The only runtime dependency is **pydantic**; provider SDKs (Anthropic, OpenAI) are optional extras.

## Install

```bash
pip install composeai                  # core: pydantic + stdlib only
pip install "composeai[anthropic]"     # + the Anthropic SDK
pip install "composeai[openai]"        # + the OpenAI SDK
pip install "composeai[all]"           # both
```

## The 90-second tour

Define an agent as a typed function. The docstring is its system prompt, the return annotation is its structured output type:

```python
import composeai as compose
from pydantic import BaseModel, Field


class FactSheet(BaseModel):
    topic: str
    key_facts: list[str] = Field(description="Three crisp, verifiable facts")


@compose.tool
def count_words(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text whose words should be counted.
    """
    return len(text.split())


@compose.agent(model="anthropic/claude-sonnet-5", tools=[count_words])
def researcher(topic: str) -> FactSheet:
    """You are a meticulous researcher. Extract crisp, verifiable facts."""
    return compose.prompt(f"Build a fact sheet about: {topic}")


facts = researcher("quantum computing")   # -> FactSheet (or raises)
```

Calling `researcher(...)` runs the whole loop and returns the validated `FactSheet`. Every call like this is automatically persisted, so you can inspect it from the command line right afterward:

```console
$ compose trace --last
trace 01KXC6RDB8E29NZHEAC2F54M11 — ok — [$0.0150 · 2.9k tok · 3ms]
└─ ◆ researcher [$0.0150 · 2.9k tok · 3ms]
   ├─ ▸ anthropic/claude-sonnet-5 [$0.0075 · 1.5k tok · 0ms]
   ├─ ⚙ count_words [0ms]
   └─ ▸ anthropic/claude-sonnet-5 [$0.0075 · 1.5k tok · 0ms]
```

No accounts, no exporters, no instrumentation to wire up — the trace (and its cost) is just there, in `./.compose`, the moment the agent finishes.

## Where to go next

| Page | What's there |
| :--- | :--- |
| [agents](agents.md) | The `@agent` idiom, structured output and repairs, tools, resilience knobs, naming/replacing agents, `.run()`/`.stream()` |
| [composition](composition.md) | `pipe`, `aggregate`, `map`, build-time type checking, nesting combinators |
| [flows](flows.md) | `@task`/`@flow`, the journal, determinism, `resume()`, human-in-the-loop |
| [providers](providers.md) | Model strings vs `Model` instances, API keys, `openai_compatible`, pricing, reasoning-model gotchas |
| [observability](observability.md) | The local tracing model, every `compose` CLI command, `--import`, `COMPOSE_TRACE_CONTENT` |
| [budgets](budgets.md) | `Budget(usd=, tokens=)`, what counts, cumulative spend across `resume()`, `BudgetExceededError` |
| [testing](testing.md) | `FakeModel`, cassettes, `@agent(cache=True)`, `reset_registries()` |
| [mcp](mcp.md) | Connect MCP servers' tools to your agents |

## See also

Start with [agents](agents.md) for the `@agent` idiom in depth, or [composition](composition.md) to see how agents wire together into pipelines.
