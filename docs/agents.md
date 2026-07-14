# Agents

An agent is a plain Python function decorated with `@compose.agent`: its docstring becomes the system prompt, its body returns the user prompt, and its return-type annotation becomes the structured output type that the model's reply is validated into.

## The `@agent` idiom

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

The decorated function *is* the agent: docstring → system prompt, body → user prompt (or a full `list[Message]` conversation), return annotation → structured output schema, validated back into that type. `compose.prompt(...)` marks the body's returned value as the prompt — a typed no-op (declared `-> Any`) that keeps static type checkers happy about a body returning a `str` where the annotation promises `FactSheet`. Returning a bare `str` (or `list[Message]`) works identically at runtime; `prompt()` only exists for type-checker ergonomics. Tools run in a loop until the model produces a final answer.

`model=` is the only required argument, and it's keyword-only: either a `"provider/model-id"` string (resolved lazily) or an existing `Model` instance — including `FakeModel` from `composeai.testing`, for tests that need no network or provider SDK.

## Structured output and `max_repairs=`

Failed structured output doesn't have to be fatal. With `@agent(max_repairs=2)`, if the model's final reply fails JSON parsing or schema validation, composeai appends the validation error as a corrective user message and re-asks — up to `max_repairs` times — instead of raising immediately:

```python
@compose.agent(model="anthropic/claude-sonnet-5", max_repairs=2)
def researcher(topic: str) -> FactSheet:
    """You are a meticulous researcher. Extract crisp, verifiable facts."""
    return compose.prompt(f"Build a fact sheet about: {topic}")
```

Each repair is a full-price LLM turn — the whole conversation is re-sent — and counts against `max_turns`, so it isn't free, but it's far cheaper (and more effective against small or local models) than a cold re-run. The default is `max_repairs=0` (fail fast). Every validation failure that reaches you, repaired or not, comes back as a `ComposeError` — a raw pydantic `ValidationError` never leaks out of `@agent`.

## Tools

`@compose.tool` turns a plain, typed function into a model-callable tool. It builds a strict JSON Schema from the function's signature and parses a Google-style docstring for the tool's own description and its per-argument descriptions:

```python
@compose.tool
def search_docs(query: str, limit: int = 5) -> str:
    """Search internal documentation for matching pages.

    Args:
        query: The search query.
        limit: Maximum number of results to return.
    """
    return f"{limit} results for {query!r}"
```

Everything before a line reading exactly `Args:` becomes the tool description; each `name: description` line under `Args:` becomes that parameter's schema description. A tool with no docstring (and no explicit `description=`) raises `ConfigError` at decoration time — the model relies on the description to know when to call it.

`@tool(requires_approval=True)` gates the tool behind a human: the agent pauses mid-loop until `resume(run_id, answers={"tool_name": True})` (or `False` to deny — the model sees `"denied by user"` and carries on). See [flows](flows.md) for the full human-in-the-loop story.

`@tool(timeout=...)` (seconds) bounds one execution of the tool body. A timed-out call surfaces to the model as an `is_error` tool result — the agent keeps running and the model can react — it never aborts the run:

```python
@compose.tool(timeout=5.0)
def fetch_url(url: str) -> str:
    """Fetch a URL's contents.

    Args:
        url: The URL to fetch.
    """
    ...
```

When the model requests several tool calls in one turn, they run in parallel with no blanket bound on the batch as a whole — `@tool(timeout=...)` is the only per-call guard, so an individual tool with no timeout set can run indefinitely alongside its siblings.

Tools can also come from MCP servers — see [mcp](mcp.md).

## Resilience knobs

All of these are keyword-only arguments to `@compose.agent(...)`:

| Argument | Default | What it does |
| :--- | :--- | :--- |
| `retries` | `0` | Retry a failed provider call this many times before giving up (or falling back). |
| `fallback` | `None` | A second `"provider/model-id"` string or `Model`, resolved lazily and used only if every `retries` attempt against the primary model fails. |
| `timeout` | `None` | Seconds. Checked at turn boundaries only; an in-flight model call is never interrupted. Raises `AgentTimeoutError`. |
| `max_turns` | `10` | Maximum LLM turns (including repair turns) before raising `MaxTurnsExceededError`. |
| `max_tokens` | `16000` | Passed to the model on every call; hitting it before the response finishes raises `ComposeError`. |
| `temperature` | `None` | Passthrough-only — composeai never sets one for you. Modern Claude models reject sampling parameters outright, so leave it unset for Claude. |

Note that `@agent(timeout=...)` is unrelated to a model constructor's own `timeout=`: the agent's `timeout` is a turn-boundary watchdog, while the model's `timeout` bounds each individual HTTP request at the SDK-client level. See [providers](providers.md). Pressing Ctrl-C yourself is a different situation again — see [flows](flows.md#ctrl-c-mid-run) for what actually happens to the in-flight attempt.

## `name=` and `replace=`

`@agent(name=...)` overrides the registered/routing name (default: the function's `__name__`) — useful when two agents would otherwise share a function name. Agent names must be unique per process, since `resume()` uses the name to route a paused or crashed standalone agent run back to its definition; a duplicate name raises `ConfigError` at decoration time.

`@agent(replace=True)` re-binds an existing name instead of raising — handy for runtime-bound factories and test fixtures. **Warning:** standalone-agent resume has no fingerprint/staleness check (unlike `@flow`), so a paused run resumed after a `replace=True` rebind continues silently against the *new* definition.

## `.run()` vs calling vs `.stream()`

Calling the agent directly (`researcher("quantum computing")`) is sugar for `researcher.run("quantum computing").output`. Need more than the output? `.run()` returns the whole `Run`:

```python
run = researcher.run("quantum computing")
run.output          # the FactSheet
run.usage           # tokens + USD cost, rolled up across every LLM call
run.trace.print()   # the trace tree
```

Both the call form and `.run()` accept an optional keyword-only `budget`, enforced across every LLM call in the run — see [budgets](budgets.md):

```python
researcher.run("quantum computing", budget=compose.Budget(usd=0.50, tokens=200_000))
```

`.stream(...)` runs the same loop on a background thread and returns a `RunStream` for live consumption — token deltas interleaved with the same span events tracing already produces:

```python
stream = researcher.stream("quantum computing")

for event in stream:
    if event.kind == "text_delta" and event.text:
        print(event.text, end="", flush=True)

stream.run.trace.print()   # blocks until settled; the full trace, same events
```

The full vocabulary an event's `kind` can take (`composeai.events.Event.kind`) is `span_started`, `text_delta`, `thinking_delta`, `tool_call_started`, `tool_args_delta`, `tool_call_finished`, `span_finished`, `paused`, and `run_finished` — the same events tracing is built from, on agents, pipelines, and flows alike, so a live UI and `compose trace` can never disagree about what happened.

## See also

[composition](composition.md) wires agents together into pipelines with build-time type checking; [flows](flows.md) makes a sequence of agent calls durable and resumable; [testing](testing.md) covers `FakeModel` for testing agents with no network.
