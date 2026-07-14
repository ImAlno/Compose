# Observability

Every run — `@agent`, `@flow`, `pipe`, `aggregate` — persists its full trace to a local SQLite store as it happens: spans, token usage, USD cost, replay status, and (unless you opt out) the actual input/output payloads. There's no account to create, no exporter to configure, and no opt-in step — the trace is just there the moment you run anything.

## The tracing model

State lives at `COMPOSE_DIR` (default `./.compose`, read lazily so tests and multi-project setups can redirect it), specifically at `{COMPOSE_DIR}/runs.db` — a single SQLite file, WAL-mode, holding the `runs`, `journal`, `spans`, `span_payloads`, `pending_interrupts`, and `agent_state` tables. Nothing is sent anywhere: no SaaS backend, no OpenTelemetry collector, no network call of any kind. Data never leaves the machine unless you move the file yourself.

Every `@agent`/`@task`/model-adapter call opens a span that nests under whatever span is active, and every finished span is persisted as it completes — so `compose trace`/`compose runs`/`compose costs` can inspect a run while it's still going, not just after it finishes. This is the same store `resume()` reads the journal from and `Budget` reads prior spend from; the CLI is a read-mostly reporting layer on top of it, not a separate system.

## The CLI

The `compose` command reads `{COMPOSE_DIR}/runs.db` with its own plain `sqlite3` connections — it never imports `composeai.models.anthropic`/`composeai.models.openai` (or the provider SDKs) at import time, so every subcommand works with no provider SDK installed at all.

### `compose runs` — list recent runs

```console
$ compose runs
01K3F8G7QZR3XJ8N4V0T5W2Y1B  flow      research             completed  2m ago       $0.0187     3200 tok
01K3F8G6XJ0Z5T2Q9H1M3N7V6C  agent     researcher           completed  3m ago       $0.0150     2920 tok
01K3F8G6QF1M9X3B7K2P4R8T5Y  agent     researcher           paused     5m ago       $0.0012      760 tok
```

Flags: `-n/--limit` (default 10, must be positive), `--json` (dumps the raw rows as JSON instead), `--status {completed,failed,paused,running}`, `--kind {agent,flow,pipe,aggregate}`, `--since` (`"<N>d"`, `"<N>h"`, or `"YYYY-MM-DD"`), and `-q/--query` for full-text search over every span's captured input/output payloads (falls back to a warning and no filtering if the local `sqlite3` build lacks FTS5):

```console
$ compose runs -q "quantum computing" --status completed
01K3F8G7QZR3XJ8N4V0T5W2Y1B  flow      research             completed  2m ago       $0.0187     3200 tok
01K3F8G6XJ0Z5T2Q9H1M3N7V6C  agent     researcher           completed  3m ago       $0.0150     2920 tok
```

### `compose trace` — render one run's trace

```console
$ compose trace 01K3F8G7QZR3XJ8N4V0T5W2Y1B     # or: compose trace --last
trace 01K3F8G7QZR3XJ8N4V0T5W2Y1B — ok — [$0.0187 · 3.2k tok · 1.8s]
└─ ▶ research [$0.0187 · 3.2k tok · 1.8s]
   ├─ • fetch_sources [8ms]
   ├─ ⇉ map(summarize) [132ms]
   │  ├─ • summarize [122ms]
   │  └─ • summarize [132ms]
   ├─ ◆ editor [$0.0187 · 3.2k tok · 1.6s]
   │  └─ ▸ anthropic/claude-sonnet-5 [$0.0187 · 3.2k tok · 1.6s]
   └─ • publish [50ms]
```

`--last` skips straight to the most recently created run; otherwise `compose trace` accepts a full run id or any unique prefix of one. A run whose id you copy-pasted only part of resolves the same way. On a **paused** run, the trace is followed by a banner naming every pending interrupt and the exact call to resolve it:

```console
$ compose trace 01K3F8G6QF1M9X3B7K2P4R8T5Y
trace 01K3F8G6QF1M9X3B7K2P4R8T5Y — paused — [$0.0012 · 760 tok · 210ms]
└─ ▶ research [$0.0012 · 760 tok · 210ms] ⏸ paused
   ...

⏸  run 01K3F8G6QF1M9X3B7K2P4R8T5Y is paused with 1 pending interrupt(s):
  - id='publish' kind='approval' question=None
    payload={'draft': 'quantum computing enables...'}

To resume:
    from composeai import resume
    resume('01K3F8G6QF1M9X3B7K2P4R8T5Y', answers={'publish': True})
```

### `compose diff` — structurally diff two traces

```console
$ compose diff 01K3F8G6QF1M9X3B7K2P4R8T5Y 01K3F8G7QZR3XJ8N4V0T5W2Y1B
diff 01K3F8G6QF1M -> 01K3F8G7QZR3
  Δcost: +$0.0037
  Δtokens: +285
  Δduration: -90ms

  ◆ researcher  Δcost=+$0.0037 Δtokens=+285 Δdur=-90ms
    ▸ anthropic/claude-sonnet-5  Δcost=+$0.0037 Δtokens=+285 Δdur=-90ms
```

Spans are matched by structural path (kind, name, and position among same-kind-and-name siblings), not by span id — so two independent runs of the same agent/flow line up node-for-node. A `(output changed)` suffix appears on any matched span whose output payload hashes differently between the two runs; a line prefixed `+`/`-` marks a span present in only one of the two traces.

### `compose costs` — group-by spend report

```console
$ compose costs --by model        # or --by name / --by day, --since 7d
costs by model
  claude-sonnet-5          calls=4      tokens=8420       cost=$0.0524
  claude-haiku-4-5         calls=1      tokens=760        cost=$0.0012
  gpt-5.6-luna             calls=2      tokens=1900       cost=$0.0034
  TOTAL                    calls=7      tokens=11080      cost=$0.0570
```

`--by` is one of `model` (grouped by the model string the call was made against), `name` (grouped by the owning run's name — the `@flow`/`@agent` it belongs to), or `day` (grouped by calendar date). Only `llm`-kind spans count. A bucket priced in full shows a plain `$X.XXXX`; a bucket where at least one call had no known price shows a `≥$X.XXXX` partial (the true total is at least that much, never fabricated); a bucket with no priced calls at all shows `-`.

### `compose export` — turn a run into a replayable cassette

```console
$ compose export 01K3F8G7QZR3XJ8N4V0T5W2Y1B --cassette tests/cassettes/research.json
wrote 3 entries to tests/cassettes/research.json
```

Pulls every persisted `llm` span for the run and writes them as a cassette — the same file format `composeai.testing`'s `cassette` fixture replays offline. See [testing](testing.md) for how cassettes are consumed in tests. If the run was captured with `COMPOSE_TRACE_CONTENT=0` (below), the exported entries for those spans have no `system`/`messages` to work with and print a warning to that effect.

### `compose path` — print the state directory

```console
$ compose path
.compose
```

## `--import`: decoding your own types

The CLI never imports your application code — a run whose payloads reference your own pydantic model, dataclass, or enum can't decode them out of the box:

```console
$ compose trace 01K3F8G7...
compose: Cannot decode unregistered type 'research_agent.schemas:SubQuestion': import the
module that defines it (e.g. `import research_agent.schemas`) or call register_serializable(...)
before decoding this data. Types are never imported automatically, by design (security).
```

This is a deliberate security stance: nothing composeai does ever imports arbitrary code just because it saw a matching type tag in stored data — running someone else's `runs.db` through `compose trace` must never execute code that data implies, only code you explicitly asked to load. `--import MODULE` (repeatable) is that explicit ask — it's accepted on `runs`, `trace`, `diff`, and `export` (the four subcommands that decode stored payloads), imports the named module, and registers every pydantic model/dataclass/enum found in its namespace before decoding anything:

```console
$ compose trace --import research_agent.schemas 01K3F8G7...
```

The same registration is available programmatically, for library code that wants to make its own types decodable without going through the CLI at all:

```python
import composeai as compose


@compose.register_serializable
class SubQuestion:
    ...


import research_agent.schemas

compose.register_module_types(research_agent.schemas)
```

`register_module_types(module)` scans `vars(module)` and recurses into pydantic field types, registering every model/dataclass/enum it finds — handy for a barrel/schemas module that re-exports types defined elsewhere. `register_serializable(cls)` registers one class directly (and doubles as a decorator, as above). Both are exactly what `--import` calls under the hood.

## `COMPOSE_TRACE_CONTENT=0`

Setting `COMPOSE_TRACE_CONTENT=0` in the environment stops **span** input/output payload capture — usage, status, and timing are still recorded regardless. Nothing else about the run changes. Quoting the project's own rules of the road on exactly what this does and doesn't gate:

> `COMPOSE_TRACE_CONTENT=0` stops *spans* from capturing input/output payloads — usage, status, and timing are always recorded regardless. This does **not** extend to anything composeai needs as functional state to actually work, all of which are written in full unconditionally, regardless of `COMPOSE_TRACE_CONTENT`: a paused agent's `agent_state` snapshot (durable pause/resume — `approve()`/`ask_human()`/`@tool(requires_approval=True)` — requires the full in-progress conversation, including tool call arguments and results, so `resume()` can continue exactly where it left off); the `@flow` journal (step results must be real to replay correctly); and the test kit's own on-disk artifacts — `record_cassette`/the `cassette` fixture and `@compose.agent(cache=True)`'s filesystem response cache both need a call's real request/response to be replayable or servable as a cache hit later. If your tools handle secrets/PII in their arguments or results, treat `{COMPOSE_DIR}` (`runs.db`, `cache/`, and any cassette files you commit) as sensitive (filesystem permissions, encryption at rest, etc.) rather than relying on `COMPOSE_TRACE_CONTENT` to keep it out of the store.

In short: this flag is for keeping raw prompts/completions out of `compose trace`'s tree view and `-q` search results specifically — not a substitute for treating `{COMPOSE_DIR}` itself as sensitive if your agents ever handle secrets or PII.

## See also

[flows](flows.md) covers the paused-run banner and `resume()` in context; [budgets](budgets.md) covers `Budget` and how it relates to `compose costs`; [testing](testing.md) covers cassettes, `compose export`'s output format, and `FakeModel`.
