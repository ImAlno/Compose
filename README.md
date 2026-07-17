# composeai — 𝑓(𝑔(𝑥))

Multi-agent workflows as the typed Python functions you already write — wiring checked before a single API call is made.

- **Agents are typed functions.** The docstring is the system prompt, the body returns the user prompt, the return annotation is the structured output type. Calling one returns that type — or raises.
- **There's nothing new to learn.** No graph objects, no `Runnable`/`StateGraph` class ecosystem — composeai is functions, a handful of decorators (`@agent`, `@tool`, `@task`, `@flow`), and the pydantic models you already use.
- **Fan-out is parallel by default.** `aggregate(...)` runs every branch concurrently and `compose.map(fn, items)` processes items in parallel — no futures, no executors, no asyncio unless you want it.
- **Composition is checked before it runs.** `pipe(researcher, copywriter)` (or `researcher >> copywriter`) verifies every stage boundary at build time; a wiring bug raises `CompositionTypeError` before a single API call is made — and your type checker sees the same contract (`pipe(researcher, copywriter)` infers `Pipeline[Topic, Article]`), so the mismatch is flagged in your editor too.
- **Type boundaries are enforced at runtime, too.** Every value crossing a `pipe`/`aggregate`/`map` stage is validated against the stage's declared input type (pydantic strict mode): a `dict` is coerced into your model, an `int` widens to a `float`, but a lossy or shape-wrong value raises `StageTypeError` — the runtime twin of the build-time `CompositionTypeError`. Annotate a boundary `Any` to opt out.
- **Tracing is always on, and local.** Every run persists spans, token usage, and cost to a SQLite store on your disk (`./.compose`). No SaaS, no instrumentation, no opt-in — the trace is just there.
- **Flows are durable.** A `@flow` journals every step; if it crashes — or pauses on a **named interrupt** (`approve("publish")`, `ask_human("pick a title")`) waiting for a human — `resume(run_id)` continues it in the same process or a brand-new one, days later, replaying finished steps without re-paying for them.
- **Every run can carry a spend cap.** `Budget(usd=..., tokens=...)` is enforced after every LLM call in a run's subtree, and stays cumulative across `resume()` — a run can't dodge its budget by crashing and getting resumed.
- **Streaming and tracing are the same event bus.** `.stream()` yields `text_delta`/`thinking_delta`/`tool_call_started`/`tool_args_delta` interleaved with the very `span_started`/`span_finished`/`run_finished` events the trace is built from — on agents, pipelines, and flows alike, so a live UI and the trace can never disagree.
- **Sync and async surfaces over one engine.** `.arun()`/`.astream()`, `aresume`, `amap`, `anow`/`arandom`, and async `@tool`/`@task`/`@agent`/`@flow` bodies mirror the sync API exactly — the same engine, driven either from composeai's own background thread or directly on your already-running event loop.
- **MCP servers plug straight into `tools=`.** `compose.mcp_tools(command=..., ...)` connects to a Model Context Protocol server (stdio or streamable HTTP) and turns its tools into ordinary composeai `Tool` objects — indistinguishable from `@compose.tool` ones, including the same `requires_approval=` pause/resume.

Runtime dependencies: **pydantic + the standard library**. Provider SDKs are optional extras. Python ≥ 3.10.

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

No accounts, no exporters, no instrumentation to wire up — the trace (and its cost) is just there, in `./.compose`, the moment the agent finishes. See [agents](docs/agents.md) for `.run()`/`.stream()`, repairs, and resilience knobs; [composition](docs/composition.md) and [flows](docs/flows.md) for wiring agents into pipelines and durable, resumable workflows; [budgets](docs/budgets.md) for the `Budget(usd=..., tokens=...)` cap shown above.

## Documentation

| Page | What's there |
| :--- | :--- |
| [docs/index.md](docs/index.md) | Project overview, the 90-second tour, and where to go next |
| [docs/agents.md](docs/agents.md) | The `@agent` idiom, structured output and repairs, tools, resilience knobs, naming/replacing agents, `.run()`/`.stream()` |
| [docs/composition.md](docs/composition.md) | `pipe`, `aggregate`, `map`, build-time type checking, nesting combinators |
| [docs/typing.md](docs/typing.md) | The static-typing contract (`AgentFunction[P, R]`, `Pipeline[In, Out]`, `Run[R]`), the pipe ladder, `StageTypeError` runtime validation |
| [docs/flows.md](docs/flows.md) | `@task`/`@flow`, the journal, determinism, `resume()`, human-in-the-loop |
| [docs/async.md](docs/async.md) | `.arun()`/`.astream()`, `aresume`, `amap`, `anow`/`arandom`, async `@tool`/`@task`/`@agent`/`@flow` bodies |
| [docs/providers.md](docs/providers.md) | Model strings vs `Model` instances, API keys, `openai_compatible`, pricing, reasoning-model gotchas |
| [docs/observability.md](docs/observability.md) | The local tracing model, every `compose` CLI command, `--import`, `COMPOSE_TRACE_CONTENT` |
| [docs/budgets.md](docs/budgets.md) | `Budget(usd=, tokens=)`, what counts, cumulative spend across `resume()`, `BudgetExceededError` |
| [docs/testing.md](docs/testing.md) | `FakeModel`, cassettes, `@agent(cache=True)`, `reset_registries()` |
| [docs/mcp.md](docs/mcp.md) | Connect MCP servers' tools to your agents |

## Rules of the road

The contracts composeai holds you to — and the ones it holds itself to:

- **Flow bodies must be deterministic.** Replay works by re-running the body and substituting journaled step results in call order. Side effects, randomness, and clock reads belong inside `@task`/`@agent` steps, never in the flow body itself. Nothing detects a violation; it just won't replay correctly.
- **A step journals only after it returns.** If the process dies between a task's external side effect and its journal write, resume re-runs that task — make external side effects idempotent.
- **Journals are for pause/approval and crash recovery, not cross-release storage.** If a `@flow`'s source changes between pause and resume, `resume()` fails loud with `ResumeMismatchError` (the journal may no longer match the new call sequence); `allow_code_change=True` overrides it deliberately.
- **State lives at `COMPOSE_DIR`** (default `./.compose`). `COMPOSE_TRACE_CONTENT=0` stops *spans* from capturing input/output payloads — usage, status, and timing are always recorded regardless. This does **not** extend to anything composeai needs as functional state to actually work, all of which are written in full unconditionally, regardless of `COMPOSE_TRACE_CONTENT`: a paused agent's `agent_state` snapshot (durable pause/resume — `approve()`/`ask_human()`/`@tool(requires_approval=True)` — requires the full in-progress conversation, including tool call arguments and results, so `resume()` can continue exactly where it left off); the `@flow` journal (step results must be real to replay correctly); and the test kit's own on-disk artifacts — `record_cassette`/the `cassette` fixture and `@compose.agent(cache=True)`'s filesystem response cache both need a call's real request/response to be replayable or servable as a cache hit later. If your tools handle secrets/PII in their arguments or results, treat `{COMPOSE_DIR}` (`runs.db`, `cache/`, and any cassette files you commit) as sensitive (filesystem permissions, encryption at rest, etc.) rather than relying on `COMPOSE_TRACE_CONTENT` to keep it out of the store.
- **`temperature` is passthrough-only.** composeai never sets one for you, and modern Claude models reject sampling parameters with a 400 — leave it unset for Claude.
- **Cost is never fabricated.** Priced models get exact USD from a dated, in-repo price table; calls with no known price report `cost_usd=None`, and any total that includes them renders as a `≥$X` partial (or `-` when nothing was priceable) instead of a made-up number. USD budgets consequently can't see unpriced spend — set a token budget too if you need a hard cap.
- **Retries can re-stream.** With `retries > 0`, a provider error striking mid-stream re-runs the call from the start — consumers of `.stream()` may see the same deltas twice on one llm span. Render final outputs (or treat a fresh delta burst as a reset) if double-rendering matters.

## vs. the alternatives

| | LangChain / LangGraph | composeai |
| :--- | :--- | :--- |
| **Core architecture** | Configuration & state graphs (`Runnable`, `StateGraph`) | Plain typed functions, composed with `pipe`/`aggregate`/`map` |
| **Learning curve** | A proprietary class ecosystem | Decorators on regular functions and pydantic types |
| **Wiring bugs surface** | At runtime, mid-graph | At composition time, before any API call |
| **Debugging** | Deeply nested framework traces | A breakpoint between two functions; local trace trees with exact costs |
| **Observability** | External/SaaS platforms, opt-in callbacks | Always-on local SQLite tracing + a CLI, zero setup |
| **Durability & HITL** | Separate checkpointer/orchestrator machinery | Journaled `@flow` + named interrupts, one `resume()` |
| **Dependencies** | Heavy transitive footprint | pydantic + stdlib; provider SDKs as optional extras |

## Roadmap

- OpenTelemetry exporter (the span model already tracks `gen_ai.*` attribute conventions)
- OpenAI request-side reasoning niceties (`reasoning.summary`, `encrypted_content` round-trip) -- 0.6.0 shipped `effort` passthrough; these remain unshipped
- TypeScript sibling package
- Consolidate the MCP bridge onto the runtime loop (each MCP server currently owns its own dedicated event-loop thread rather than sharing composeai's)
- Span-persistence queue tuning under streaming storms (the store's single writer thread is a FIFO queue; a very high-rate `.stream()`/`.astream()` workload hasn't been load-tested against it)
- Typed `aggregate()` branch values -- infer a per-branch `TypedDict` output (`{"a": <A's return>, "b": <B's return>}`) instead of today's uniform `dict[str, Any]`, so a downstream stage sees each branch's real result type

## License

MIT
