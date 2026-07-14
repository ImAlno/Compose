# composeai — 𝑓(𝑔(𝑥))

A radically simple, functional framework for multi-agent AI workflows.

- **Agents are typed functions.** The docstring is the system prompt, the body returns the user prompt, the return annotation is the structured output type. Calling one returns that type — or raises.
- **Composition is checked before it runs.** `pipe(researcher, copywriter)` verifies every stage boundary at build time; a wiring bug raises `CompositionTypeError` before a single API call is made.
- **Tracing is always on, and local.** Every run persists spans, token usage, and cost to a SQLite store on your disk (`./.compose`). No SaaS, no instrumentation, no opt-in — the trace is just there:

```
trace 01KXC6RDB8E29NZHEAC2F54M11 — ok — [$0.0150 · 2.9k tok · 3ms]
└─ ◆ researcher [$0.0150 · 2.9k tok · 3ms]
   ├─ ▸ anthropic/claude-sonnet-5 [$0.0075 · 1.5k tok · 0ms]
   ├─ ⚙ count_words [0ms]
   └─ ▸ anthropic/claude-sonnet-5 [$0.0075 · 1.5k tok · 0ms]
```

- **Flows are durable.** A `@flow` journals every step; if it crashes — or pauses on a **named interrupt** (`approve("publish")`) waiting for a human — `resume(run_id)` continues it in the same process or a brand-new one, days later, replaying finished steps without re-paying for them.
- **Streaming and tracing are the same event bus.** `.stream()` yields token deltas interleaved with the very span events the trace is built from, so a live UI and the trace can never disagree.

Runtime dependencies: **pydantic + the standard library**. Provider SDKs are optional extras. Python ≥ 3.10.

## Install

```bash
pip install composeai                  # core: pydantic + stdlib only
pip install "composeai[anthropic]"     # + the Anthropic SDK
pip install "composeai[openai]"        # + the OpenAI SDK
pip install "composeai[all]"           # both
```

## Providers

A `"provider/model-id"` string (`"anthropic/claude-sonnet-5"`, `"openai/gpt-5.6-luna"`) resolves lazily against the matching extra. Any other OpenAI-compatible Chat Completions server — Ollama, vLLM, LM Studio, ollama.com, a self-hosted proxy — goes through an explicit factory instead, since it needs a `base_url`:

```python
import composeai as compose

model = compose.openai_compatible(
    "https://ollama.com/v1", "kimi-k2.6",
    api_key="...", timeout=120,
    input_price=0.60, output_price=2.50,   # USD per million tokens -- ollama.com bills real usage
)

@compose.agent(model=model)
def researcher(topic: str) -> FactSheet: ...
```

- **`timeout=`** (seconds) bounds each HTTP request at the SDK-client level — the only real guard against a hung call (`@agent(timeout=...)` only checks turn boundaries, so it can't interrupt one already in flight; see Quickstart). Also accepted directly by the `AnthropicModel`/`OpenAIModel` constructors.
- **`input_price=`/`output_price=`** (USD per **million** tokens, both together or neither — else `ConfigError`) register this model's price by calling the public `composeai.register_price(provider, model, composeai.ModelPrice(input=..., output=...))` on your behalf. Without a registered price, `compose costs` and `Budget(usd=...)` can't see the spend at all (unpriced calls always report `cost_usd=None` — see "Rules of the road"); `register_price`/`ModelPrice` are also available directly, for the string `"provider/model-id"` form.
- **`schema_mode="prompt"`** — some compat servers accept `response_format: json_schema` but silently ignore it, returning free-form text or markdown instead of the constrained shape (verified against ollama.com for every model tested). Prompt mode embeds the schema in the last user message instead and parses the reply leniently (strips code fences, extracts the first balanced JSON object). Reasoning models can burn thousands of hidden tokens before any visible content — keep `max_tokens` generous, or a structured call comes back empty (composeai raises a targeted error naming `reasoning_tokens` when that happens, instead of a bare JSON-decode failure). The same reasoning-tokens hint is folded into the ordinary "hit max_tokens" error too, for any provider — not just prompt mode.

## Quickstart

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

The decorated function *is* the agent: docstring → system prompt, body → user prompt (or a full `list[Message]` conversation), return annotation → structured output schema, validated back into the type. `compose.prompt(...)` marks the body's return value as the prompt — it's a typed no-op that keeps static type checkers happy about the body returning a prompt where the annotation declares the output type (a bare `str` works identically at runtime). Tools run in a loop until the model produces the final answer.

Need more than the output? `.run()` returns the whole `Run`:

```python
run = researcher.run("quantum computing")
run.output          # the FactSheet
run.usage           # tokens + USD cost, rolled up across every LLM call
run.trace.print()   # the tree shown above
```

Every agent also takes an optional spend cap: `researcher.run(topic, budget=compose.Budget(usd=0.50, tokens=200_000))` raises `BudgetExceededError` at the first LLM-call boundary past the cap.

Failed structured output doesn't have to be fatal: `@agent(max_repairs=2)` appends the validation error as a corrective user message and re-asks within the same conversation before giving up — each repair is a full-price turn and counts against `max_turns`, but it's far cheaper (and more effective against small/local models) than a cold re-run. Every validation failure that reaches you, repaired or not, is a `ComposeError` — a raw pydantic `ValidationError` never leaks out of `@agent`. `name=`/`replace=` on `@agent` (mirrored on `@flow`/`@task`) override the registered name and allow rebinding it instead of raising — useful for runtime-bound factories and test fixtures; `replace=True` on an agent carries one caveat: a paused run resumed after the replace continues silently against the *new* definition (no fingerprint/staleness check, unlike `@flow`).

Note that `@agent(timeout=...)` and a model constructor's `timeout=` are unrelated: the agent's `timeout` is a turn-boundary watchdog (checked between turns, so it can't interrupt a single in-flight call), while the model's `timeout` bounds each individual HTTP request at the SDK-client level — see [Providers](#providers) above.

## Composition

```python
@compose.agent(model="openai/gpt-5.6-luna")
def copywriter(sheet: FactSheet) -> str:
    """Turn a fact sheet into one punchy line."""
    return f"Write one punchy line from: {sheet.model_dump_json()}"


write_post = compose.pipe(researcher, copywriter)   # types checked HERE
post = write_post("quantum computing")
```

A mismatch never gets to run (and never spends a token):

```python
compose.pipe(researcher, researcher)
# CompositionTypeError: pipe(): stage 1 (researcher) returns FactSheet
#                       but stage 2 (researcher) expects str
```

Fan-out is a function too — branches run in parallel threads:

```python
audits = compose.aggregate(words=lambda s: len(s.split()), chars=len)
audits("count these words")          # {'words': 3, 'chars': 17}

compose.map(summarize, sources)      # one stage over many items, order preserved
```

Stages are agents, pipelines, aggregates, or plain Python callables — routing is an `if` statement, not a graph edge class.

`map(fn, items, *, max_workers=None, timeout_per_item=None, on_error="raise")` takes two knobs beyond the basic fan-out: `timeout_per_item` (seconds) races each item on its own thread and raises `TaskTimeoutError` for just that one instead of blocking every other item forever, and `on_error="collect"` replaces the default "raise the first failure by index" behavior with a `list[MapResult]` (`ok`, `value`, `error`, `error_type`) — one entry per item, in input order, with nothing raised — so a caller can keep whatever succeeded instead of writing its own try/except-and-tag wrapper:

```python
results = compose.map(summarize, sources, on_error="collect", timeout_per_item=30)
ok = [r.value for r in results if r.ok]
```

Inside a `@flow`, `map()` already journals each item individually as it completes — a failed item (whether it raises, under the default `on_error="raise"`, or is just recorded as a failed `MapResult` under `"collect"`) never discards its siblings' completed work; only the unfinished tail re-runs on `resume()`. That's true when each item is itself a journaling stage (`@task`/`@agent`/nested `@flow`/`pipe`/`aggregate`); a plain Python callable has no journal entry of its own to replay, so its item just re-runs on `resume()` like any other unwrapped code.

`aggregate()`/an agent's parallel tool calls have no *per-branch* timeout of their own — a single hung branch (a tool body blocked on a network call with no read timeout, a provider HTTP call that never returns) blocks the whole combinator, and therefore the enclosing run, until it finishes or the process is killed. Give an individual stage its own bound with `@task(timeout=...)` (or build one into a tool/agent body) if it needs one; `map()`'s `timeout_per_item` (above) is the one combinator-level exception.

## Durable flows & human-in-the-loop

```python
@compose.flow
def research(topic: str) -> str:
    sources = fetch_sources(topic)                     # journaled step
    summaries = compose.map(summarize, sources)        # parallel, journaled steps
    draft = editor(summaries)                          # a whole agent run = one step
    if compose.approve("publish", payload={"draft": draft}):   # named interrupt
        return publish(draft)
    return f"kept as draft: {draft}"


run = research.run("quantum computing")
run.status    # "paused"
run.pending   # id='publish' kind='approval' question=None payload={'draft': ...}
```

Pausing is not an error — the process may simply exit. Later, from **any** process:

```python
run = compose.resume(run.id, answers={"publish": True})
run.status    # "completed"
```

On resume, finished steps replay from the journal (the `editor` agent's LLM calls are not re-made and not re-billed); only the unfinished tail runs:

```
└─ ▶ research [3ms]
   ├─ • fetch_sources [0ms] (replayed)
   ├─ ⇉ map(summarize) [2ms]
   │  ├─ • summarize [0ms] (replayed)
   │  └─ • summarize [0ms] (replayed)
   ├─ ◆ editor [0ms] (replayed)
   └─ • publish [0ms]
```

Interrupts are *named* (`approve("publish")`, `ask_human("style", "Formal or casual?")`), never positional — answers are journaled under that name, so resuming is idempotent and order-independent. The same mechanism gates tools: `@compose.tool(requires_approval=True)` pauses the agent mid-loop until `resume(run_id, answers={"tool_name": True})` (or `False` to deny — the model sees "denied by user" and carries on). The same crash-recovery works with no interrupt at all: if a flow dies halfway, `resume(run_id)` re-runs it and the journal skips what already happened.

A `budget=` passed to the original `.run()` stays enforced across every `resume()` of that run — spend already persisted by earlier attempts counts against the cap (replayed steps themselves cost nothing), so a run can't dodge its budget by dying and getting resumed. If a run legitimately needs more room, override it explicitly: `resume(run_id, budget=Budget(usd=5.0))` replaces the stored budget for this attempt and every later one (a plain last-write-wins update, not journaled — the journal is first-write-wins and would otherwise freeze the first override forever). Prior attempts' real spend still counts against the new cap; omitting `budget=` keeps whatever was already stored.

## Streaming

```python
stream = poet.stream("event buses")

for event in stream:
    if event.kind == "text_delta" and event.text:
        print(event.text, end="", flush=True)

stream.run.trace.print()   # blocks until settled; the full trace, same events
```

One bus carries `text_delta` / `thinking_delta` / `tool_call_started` / `tool_args_delta` / `span_started` / `span_finished` / `run_finished` — pipelines and flows stream the same way (`write_post.stream(...)`, `research.stream(...)`).

## The CLI

Everything above persisted to `./.compose` (override with `COMPOSE_DIR`). The `compose` CLI reads it:

```console
$ compose runs
01KXC6RD  flow      research             completed  9s ago       $0.0012      760 tok
01KXC6RD  agent     researcher           completed  9s ago       $0.0150     2920 tok
01KXC6RD  agent     researcher           completed  9s ago       $0.0150     2920 tok

$ compose trace 01KXC6RDF5CNYVXCRVCQWJDBRR    # or: compose trace --last
trace 01KXC6RDF6TQE8PPKA0XSECBKS — paused — [$0.0012 · 760 tok · 127ms]
├─ ▶ research [$0.0012 · 760 tok · 3ms] ⏸ paused
│  ├─ • fetch_sources [0ms]
│  ├─ ⇉ map(summarize) [2ms]
│  │  ├─ • summarize [0ms]
│  │  └─ • summarize [2ms]
│  ├─ ◆ editor [$0.0012 · 760 tok · 0ms]
│  │  └─ ▸ anthropic/claude-haiku-4-5 [$0.0012 · 760 tok · 0ms]
│  └─ ⏸ publish [0ms] ⏸ paused
└─ ▶ research [3ms]
   ├─ • fetch_sources [0ms] (replayed)
   ...

$ compose costs --by model        # or --by name / --by day, --since 7d
costs by model
  claude-haiku-4-5         calls=1      tokens=760        cost=$0.0012
  claude-sonnet-5          calls=4      tokens=5840       cost=$0.0300
  TOTAL                    calls=5      tokens=6600       cost=$0.0312

$ compose diff 01KXC6RDB8TS2SPXGKKXV7Z64G 01KXC6RDBC6Z1H2DSBMFAKF01D

diff 01KXC6RD -> 01KXC6RD
  Δcost: +$0.0000
  Δtokens: +0
  Δduration: -2ms

  ◆ researcher  Δcost=+$0.0000 Δtokens=+0 Δdur=-2ms
    ▸ anthropic/claude-sonnet-5  Δcost=+$0.0000 Δtokens=+0 Δdur=-0ms  (output changed)
    ...
```

`compose trace` on a paused run also prints its pending interrupts and the exact `resume(...)` call to answer them. `compose runs -q "search terms"` full-text-searches span payloads; `compose export RUN_ID --cassette x.json` turns a run's recorded LLM calls into a replayable test fixture (below). `compose path` prints the state directory.

The CLI runs standalone — it never imports your app's code, so a run whose payloads reference your own pydantic/dataclass/enum types can't decode them out of the box:

```console
$ compose trace 01KXC6RD...
compose: Cannot decode unregistered type 'research_agent.schemas:SubQuestion': import the
module that defines it ... Types are never imported automatically, by design (security).
```

Pass `--import your_module` (repeatable) on `runs`, `trace`, `diff`, and `export` to import it first and register every pydantic model/dataclass/enum found in its namespace:

```console
$ compose trace --import research_agent.schemas 01KXC6RD...
```

The same registration is available programmatically — `composeai.register_module_types(module)` scans a module's namespace (handy for a barrel/schemas module that re-exports types from elsewhere), and `composeai.register_serializable` registers one class directly (the decorator form: `@composeai.register_serializable`). Both are what `--import` calls under the hood.

## Test kit

`FakeModel` scripts an agent without a provider or network — each item is one model turn:

```python
from composeai.testing import FakeModel

model = FakeModel([
    {"tool_calls": [{"name": "count_words", "arguments": {"text": "a b c"}}]},  # turn 1
    {"json": {"topic": "quantum computing", "key_facts": ["fact one"]}},        # turn 2
])

@compose.agent(model=model, tools=[count_words])
def researcher(topic: str) -> FactSheet:
    """You are a meticulous researcher."""
    return compose.prompt(f"Build a fact sheet about: {topic}")
```

Plain strings script text turns; `.stream()` synthesizes word-level deltas from the same script. Every request the agent made is recorded in `model.requests` for assertions.

**Cassettes** record real model traffic once and replay it offline forever. Re-export the bundled pytest fixture from your `conftest.py`:

```python
from composeai.testing import cassette  # noqa: F401
```

```python
def test_researcher(cassette):
    with cassette("tests/cassettes/researcher.json"):
        facts = researcher("quantum computing")
```

Run once with `COMPOSE_RECORD=1` against the real provider and commit the JSON; afterwards the test replays it with no key, no network, and no SDK constructed. (`compose export` builds the same file from an already-persisted run.)

**`@compose.agent(model=..., cache=True)`** is for iterating locally against a real provider: identical requests are answered from a filesystem cache under `{COMPOSE_DIR}/cache/`, report zero usage (a hit is never re-billed), and tag their span `cached=true`. Applies to non-streaming calls only.

`@agent`/`@flow`/`@task` names are unique per process (so `resume()` can route a run back to its definition), which collides with test suites that redefine the same function across test modules/parametrized cases. `composeai.testing.reset_registries()` clears all three registries between tests instead of reaching into the private `_AGENT_REGISTRY`/`_FLOW_REGISTRY`/`_TASK_REGISTRY` dicts by hand:

```python
import pytest
from composeai.testing import reset_registries

@pytest.fixture(autouse=True)
def _reset():
    reset_registries()
    yield
```

(`replace=True` on any of the three decorators is the other option, for a single deliberate rebind rather than a blanket per-test reset.)

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

## Releasing (maintainers)

Two equivalent paths:

- **Local**: put a project-scoped PyPI token in `.env` (git-ignored; see the placeholder), commit your work, then `scripts/release.sh X.Y.Z` — it bumps the version, runs the full gate, builds, and uploads. Commit the bump and tag `vX.Y.Z` afterwards.
- **GitHub**: publish a GitHub Release and `.github/workflows/release.yml` tests, builds, and publishes via PyPI trusted publishing (no stored token). One-time setup: add this repo as a trusted publisher on pypi.org (workflow `release.yml`, environment `pypi`).

## Roadmap

- OpenTelemetry exporter (the span model already tracks `gen_ai.*` attribute conventions)
- Async API (`await agent(...)`, async tools)
- TypeScript sibling package
- Extended-thinking / reasoning request configuration (Anthropic `thinking` budget, OpenAI `reasoning.summary`/`encrypted_content`) -- today `ThinkingPart` only round-trips whatever a provider returns unprompted by default; there's no `ModelRequest` field to actually ask for it

## License

MIT
