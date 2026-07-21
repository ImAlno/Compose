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
| `max_turns` | `10` | Maximum LLM turns (including repair turns) before raising `MaxTurnsExceededError`. `None` = unbounded — pair with `timeout=` or a run `Budget`. |
| `prompt_cache` | `True` | Mark cacheable prefix spans on providers with explicit cache control (Anthropic: system-prompt breakpoint + conversation-tail breakpoint once multi-turn). Cached reads bill ~0.1×, writes ~1.25× — a large net win for tool loops and fan-outs. `False` sends byte-identical requests to 0.5.x. No-op on OpenAI (automatic server-side). Never affects request hashes (cassettes/`cache=` keys are unchanged). |
| `thinking` | `None` | `None` sends nothing (each model's own default applies). `True` requests adaptive thinking with summarized display (so `thinking_delta` events carry text); `False` explicitly disables thinking. Anthropic only; no-op on OpenAI. |
| `effort` | `None` | Provider-defined reasoning-effort string passed through verbatim (Anthropic: `"low"`/`"medium"`/`"high"`/`"xhigh"`/`"max"` → `output_config.effort`; OpenAI: `"minimal"`/`"low"`/`"medium"`/`"high"` → `reasoning.effort`). Invalid values surface as `ProviderError`. |
| `thinking_budget` | `None` | Request a specific extended-thinking token budget on Anthropic (`int`, `1024 ≤ budget_tokens < max_tokens`); `None` lets the provider decide. Ignored on OpenAI and other providers. |
| `max_tokens` | `16000` | Passed to the model on every call; hitting it before the response finishes raises `ComposeError`. |
| `temperature` | `None` | Passthrough-only — composeai never sets one for you. Modern Claude models reject sampling parameters outright, so leave it unset for Claude. |

Note that `@agent(timeout=...)` is unrelated to a model constructor's own `timeout=`: the agent's `timeout` is a turn-boundary watchdog, while the model's `timeout` bounds each individual HTTP request at the SDK-client level. See [providers](providers.md). Pressing Ctrl-C yourself is a different situation again — see [flows](flows.md#ctrl-c-mid-run) for what actually happens to the in-flight attempt.

## `name=` and `replace=`

`@agent(name=...)` overrides the registered/routing name (default: the function's `__name__`) — useful when two agents would otherwise share a function name. Agent names must be unique per process, since `resume()` uses the name to route a paused or crashed standalone agent run back to its definition; a duplicate name raises `ConfigError` at decoration time.

`@agent(replace=True)` re-binds an existing name instead of raising — handy for runtime-bound factories and test fixtures. **Warning:** standalone-agent resume has no fingerprint/staleness check (unlike `@flow`), so a paused run resumed after a `replace=True` rebind continues silently against the *new* definition.

## `.run()` vs calling vs `.stream()`

`@compose.agent` infers an `AgentFunction[P, R]` that preserves the decorated function's whole signature — `P` its parameter list (names included), `R` its return type — so `researcher` type-checks as `(topic: str) -> FactSheet` and `researcher.run(...)` returns a `Run[FactSheet]` whose `.output` is a `FactSheet` a checker can see. See [typing](typing.md) for the full static-typing contract.

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

### `RunStream.cancel()`

`stream.cancel()` cooperatively cancels a streaming run — safe to call from any thread (typically the consumer thread driving the `for event in stream:` loop), and safe to call more than once. No new turn or tool work starts, the in-flight LLM stream is aborted, and the run ends as `Run(status="cancelled")`: iteration stops cleanly after the terminal `run_finished(status="cancelled")` event, and `stream.run` returns that cancelled `Run` **without raising** — unlike a failed run, which re-raises.

```python
rs = my_agent.stream("...")
for event in rs:
    if some_condition(event):
        rs.cancel()          # from the consumer thread

assert rs.run.status == "cancelled"   # no exception
```

Cancellation is cooperative, not preemptive: a tool call already executing runs to completion (Python threads can't be force-stopped) — `cancel()` only prevents *new* tool calls from starting, stops the loop between turns, and aborts the in-flight stream between provider events. In 0.9.0 this is sync `RunStream` only; `AsyncRunStream.cancel()` is future work.

## Per-call overrides

`.run()`, `.arun()`, `.stream()`, and `.astream()` all accept five keyword-only overrides, applied to that one call without touching the decoration:

- `system=` replaces the agent's docstring/system prompt for this run.
- `model=` swaps the model (a `"provider/model-id"` string or a `Model` instance).
- `approver=` is called synchronously for `requires_approval` tools instead of pausing — return `True`/`False`, or an `ApprovalReply(allow: bool, message: str | None = None)` for finer control. A denial with a `message` (`ApprovalReply(allow=False, message="…")`) is shown to the agent as the denied tool's result **in place of** the default `"denied by user"`, so the agent can adapt; a bare `False` (or `ApprovalReply(allow=False)` with no message) still yields `"denied by user"`. `ApprovalReply(allow=True, …)` runs the tool — the message is ignored for an allow. **Live-approver only:** the message is available only when an inline `approver=` is consulted synchronously; a pause→`resume(..., {id: False})` denial reads the journaled `bool` and falls back to `"denied by user"`, since the durably stored answer is a plain `bool` by design.

  ```python
  def approver(interrupt):
      if is_risky(interrupt):
          return ApprovalReply(allow=False, message="Use the read-only variant instead.")
      return True

  run = my_agent.run(approver=approver)
  ```
- `tool_interceptor=` is a `composeai.ToolInterceptor` — an object with `before(call) -> BeforeTool | None` and `after(call, result) -> ToolResultPart | None` — that fires around **every** tool call, gated or not, before the approver even sees it. `before` returning `None` (or a default `BeforeTool()`) proceeds unchanged; `BeforeTool(arguments={...})` proceeds with modified arguments; `BeforeTool(action="deny", message="...")` skips both execution and the approver, feeding `message` back as the denied tool's error result. `after` returning `None` leaves the tool's result unchanged; returning a `ToolResultPart` replaces it. `None` (the default) is a complete no-op — byte-identical behavior to not passing `tool_interceptor=` at all. It is **not journaled**: on a resumed run, `after` does not re-fire for tools that already executed before the pause, and the interceptor itself is re-supplied to the free `resume(run_id, answers, tool_interceptor=…)` to keep firing on the newly-answered calls (a `Chat` carries its own construction-time `tool_interceptor` through `c.resume()` automatically — that method takes no `tool_interceptor` argument).

  ```python
  class AuditingInterceptor:
      def before(self, call):
          return None  # inspect, but don't change anything

      def after(self, call, result):
          log(call.name, result.content)
          return None

  run = my_agent.run(tool_interceptor=AuditingInterceptor())
  ```
- `context_manager=` receives `(messages, last_input_tokens)` before every provider call and returns the messages actually sent.

These are the same five knobs `compose.chat` takes; see [chats](chats.md) for the full semantics of `approver=`, `tool_interceptor=`, and `context_manager=`.

## `.arun()` and `.astream()`

Both have asyncio-native twins, awaited directly on your own already-running event loop instead of composeai's background runtime thread:

```python
import asyncio


async def main() -> None:
    run = await researcher.arun("quantum computing")
    print(run.output)


asyncio.run(main())
```

```python
async def main() -> None:
    stream = researcher.astream("quantum computing")
    async for event in stream:
        if event.kind == "text_delta" and event.text:
            print(event.text, end="", flush=True)
    run = await stream.run()   # a method here, unlike sync RunStream.run


asyncio.run(main())
```

An `@agent` function's body may itself be `async def` — composeai awaits it natively either way, so a coroutine body works through both the sync facade and `.arun()`/`.astream()`. See [async](async.md) for the full async surface, including async tool bodies and nested async flows.

## See also

[composition](composition.md) wires agents together into pipelines with build-time type checking; [flows](flows.md) makes a sequence of agent calls durable and resumable; [testing](testing.md) covers `FakeModel` for testing agents with no network; [typing](typing.md) covers the `AgentFunction[P, R]`/`Run[R]` static-typing contract.
