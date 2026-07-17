# Flows

`@flow` turns a plain Python function into a **durable, journaled run**: every `@task`/`@agent` call it makes is recorded as it happens, so a crash — or a deliberate pause waiting on a human — can be picked up again with `resume(run_id)`, in the same process or a brand-new one, without re-running (or re-paying for) anything that already finished.

## `@task`

```python
import composeai as compose


@compose.task(retries=2, timeout=30, name="fetch_page")
def fetch_page(url: str) -> str:
    return f"contents of {url}"
```

`@task` is usable bare (`@compose.task`) or with keyword arguments:

- **`retries`** (default `0`) — on any `Exception`, retry the task body up to this many times before letting the exception propagate. Each attempt's error type and message are recorded on the task's trace span.
- **`timeout`** (seconds, default `None`) — bounds one execution on a dedicated daemon thread. A timeout raises `TaskTimeoutError` immediately and is **never retried** (retrying an already-abandoned, still-running thread would just pile up more abandoned threads); there is no safe way to stop the abandoned thread, so treat a timed-out task as failed and move on. The abandoned thread also immediately loses journal write access, so it can't corrupt a flow that has already moved on without it.
- **`name`** — the journal key prefix for this task (defaults to the function's `__name__`). Task names must be unique per process; a duplicate raises `ConfigError` unless `replace=True`.
- **`replace`** (default `False`) — re-bind an existing task name instead of raising. Only affects steps not yet journaled — an already-journaled step's stored value still replays unchanged regardless of what the task's body now does.

A `Task` object is directly callable whether or not a flow is active: outside a flow it just runs inside a plain trace span; inside one, each call auto-journals as a step.

## `@flow` and the journal

```python
@compose.flow
def research(topic: str) -> str:
    sources = fetch_sources(topic)                     # journaled step
    summaries = compose.map(summarize, sources)        # parallel, journaled steps
    draft = editor(summaries)                          # a whole agent run = one step
    return draft
```

Every `@task` call (and `@agent` call, and `compose.map` item, and nested `@flow` call) made from an active flow body is journaled to the durable store keyed by call order — `f"{name}#{n}"`, with `n` assigned in **flow-body order**, never completion order, so concurrent dispatch (`compose.map`, parallel tool calls) still gets deterministic keys. Each name gets its own counter, so two calls to `fetch_sources` inside one flow journal under `fetch_sources#1` and `fetch_sources#2`.

`flow_obj(topic)` is sugar for `flow_obj.run(topic).output`; `.run()` always starts a **new** durable run and returns the full `Run` (`.output`, `.usage`, `.trace`, `.status`, `.pending`). A flow also has `.stream(...)`, same shape as an agent's.

`@compose.flow` infers a `Flow[P, R]` that preserves the function's signature, so `.run(...)` returns a `Run[R]` (here `Run[str]`) whose `.output` is typed. That `.output` type describes a **completed** run — a paused or failed one has no output of that type, so discriminate on `.status`/`.pending` (never guard on `.output`; a checker narrows it to `Never`). See [typing](typing.md#paused-runs-and-runr) for the paused-run typing contract in full.

### Crash-resume across processes

Journaled steps replay instantly on `resume()` — only the unfinished tail actually executes. Here's the flow from above, saved to a module so it can be imported fresh in a second process:

```python
# research_flow.py
import composeai as compose


@compose.task
def fetch_sources(topic: str) -> list[str]:
    return [f"https://example.com/{topic}"]


@compose.task
def summarize(source: str) -> str:
    return f"summary of {source}"


@compose.agent(model="anthropic/claude-sonnet-5")
def editor(summaries: list[str]) -> str:
    """Turn source summaries into a short draft."""
    return compose.prompt(f"Draft a short paragraph from: {summaries}")


@compose.task
def publish(draft: str) -> str:
    return f"published: {draft}"


@compose.flow
def research(topic: str) -> str:
    sources = fetch_sources(topic)
    summaries = compose.map(summarize, sources)
    draft = editor(summaries)
    if compose.approve("publish", payload={"draft": draft}):
        return publish(draft)
    return f"kept as draft: {draft}"
```

Process A starts the run. It pauses on the unanswered `approve("publish")` gate (more on that below) and the process can simply exit:

```python
# process_a.py
from research_flow import research

run = research.run("quantum computing")
run.status    # "paused"
run.id        # save this run_id somewhere durable
run.pending   # Interrupt(id='publish', kind='approval', question=None, payload={'draft': ...})
```

Process B — hours later, a completely fresh interpreter — imports the same module (so the `@flow` re-registers under its name) and resumes by `run_id`:

```python
# process_b.py
from research_flow import research  # noqa: F401 -- import registers the @flow
from composeai import resume

run = resume("01K3F8G7QZR3XJ8N4V0T5W2Y1B", answers={"publish": True})
run.status    # "completed"
run.output    # "published: ..."
```

On this second run, `fetch_sources`, `map(summarize)`, and `editor` all replay from the journal — `editor`'s LLM call is not re-made and not re-billed — and only `publish` (never reached before the pause) actually executes:

```
└─ ▶ research [3ms]
   ├─ • fetch_sources [0ms] (replayed)
   ├─ ⇉ map(summarize) [2ms]
   │  ├─ • summarize [0ms] (replayed)
   │  └─ • summarize [0ms] (replayed)
   ├─ ◆ editor [0ms] (replayed)
   └─ • publish [0ms]
```

### Ctrl-C mid-run

A sync `@flow`/`@agent` call is a facade over composeai's own always-running background engine. Pressing Ctrl-C raises `KeyboardInterrupt` in **your** code immediately, and cancels the current attempt too: composeai schedules the engine's task for cancellation on its runtime thread before re-raising, throwing a real `CancelledError` into its await chain — the same mechanism a `timeout=` uses, not a signal merely observed after the fact (best-effort: a Ctrl-C landing in the narrow window before the attempt has even started yet just re-raises, with nothing scheduled to cancel). The durable row still lands cleanly either way — `"completed"`, `"failed"` (with error message `"cancelled"` when the attempt really was interrupted), or still `"running"` if the process exits before cancellation finishes unwinding — never a corrupt or half-written one, and every outcome is resumable.

One thing cancellation *can't* stop: a sync `@task`/`@flow` body already running on its own dedicated worker thread keeps running in the background — Python threads can't be force-stopped — but it loses journal write access the instant the attempt above it is cancelled, so nothing it does afterward can land in the journal. Don't `resume()` the same run in the *same* process immediately after a Ctrl-C expecting that thread to already be gone: it may still be unwinding — it's the abandon-guard, not timing, that keeps its now-pointless work harmless. See [async](async.md) for the full cancellation contract, including the async surface's own equivalent.

## The determinism rule, and `compose.now()`/`compose.random()`

Replay works by re-running the flow body and substituting journaled step results, in call order, wherever they occur. That only produces the right answer if **the flow body is a deterministic function of its journaled step results** — side effects, randomness, and clock reads belong inside `@task`/`@agent` calls, never directly in the flow body. Nothing detects a violation of this contract; a body that breaks it just won't replay correctly.

Wall-clock reads and randomness are common enough to need a dedicated escape hatch: `compose.now()` and `compose.random()` may be called **directly in a flow body**. Each call journals one value in flow-body order (keys `now#1`, `now#2`, ... / `random#1`, `random#2`, ...) and replays that exact value verbatim on every resume, so a flow can branch on "what time is it" or "flip a coin" without violating determinism:

```python
import composeai as compose


@compose.flow
def maybe_offer_discount(customer_id: str) -> str:
    ts = compose.now()                 # datetime, timezone-aware UTC -- journaled once
    if compose.random() < 0.1:         # float in [0, 1) -- journaled once
        return f"{customer_id} gets a discount as of {ts.isoformat()}"
    return f"{customer_id}: no discount"
```

Outside an active flow, `compose.now()` is exactly `datetime.now(timezone.utc)` and `compose.random()` is exactly `random.random()` — no journaling, no special behavior.

**When to reach for `now()`/`random()` vs. a `@task`:** use `now()`/`random()` for a single, cheap value the flow body itself needs to branch on (a timestamp to stamp onto output, a coin flip for an A/B split). Reach for `@task` instead when there's real work involved — network calls, file I/O, anything that should get `retries`/`timeout`, or that legitimately takes long enough to want its own span. `now()`/`random()` have no retry/timeout knobs of their own; they're a single journaled draw, not a unit of work.

## `resume()`

```python
def resume(
    run_id: str,
    answers: dict[str, Any] | None = None,
    *,
    budget: Budget | None = None,
    allow_code_change: bool = False,
) -> Run: ...
```

The one entry point for continuing any durable run — a `@flow` or a standalone `@agent` — in this process or a fresh one.

- **`answers`** — journaled under the interrupt's id so the paused `approve()`/`ask_human()`/approval-gated tool call finds its answer already there when the body re-executes: `resume(run_id, answers={"publish": True, "style": "formal"})`. A bare tool name (e.g. `"my_tool"`) resolves automatically to the full `tool:{name}:{call_id}` interrupt id when exactly one pending interrupt matches; an ambiguous or unmatched shorthand raises `ConfigError` naming the pending ids. Answers are journaled first-write-wins, so a second, different `resume()` call for the same interrupt can never silently overwrite the first.
- **`budget`** — overrides the run's stored `Budget` for this attempt **and every later one**. This is a plain last-write-wins column update, not journaled (the journal is first-write-wins, which would otherwise freeze the very first override forever). Prior attempts' real spend still counts against the new cap — you can't dodge a budget by crashing and resuming with a bigger one applied only going forward. `None` (the default) keeps whatever budget was already stored; there's no way to *clear* a budget via resume.
- **`allow_code_change`** (default `False`) — see the fingerprint check below.

Behavior by run state:

- **Missing run** — `ConfigError`.
- **Standalone `@agent` run** (`kind == "agent"`) — routed to the agent loop: restores the saved conversation and continues it, or re-pauses on the next unanswered interrupt.
- **Flow not registered in this process** — its defining module was never imported, so its `@flow` decoration never ran: `ConfigError` naming the fix (import the module before calling `resume()`, as in Process B above).
- **Already `"completed"`** — returns the stored output as a completed `Run` without re-executing anything.
- **Source changed since the run started** (fingerprint mismatch — `@flow` fingerprints its source via a hash of `inspect.getsource(fn)` at decoration time) — `ResumeMismatchError`, unless `allow_code_change=True`: the journal may no longer match the new code's call sequence, so this is an explicit opt-in to that risk.
- **Otherwise** (`"running"`, `"paused"`, or `"failed"`) — re-executes the flow body with the *same* `run_id` and *same* `trace_id`, so resumed spans join the original trace; journaled steps replay, the rest actually runs.

```python
from composeai import resume, Budget

# answer a pending approval
run = resume(run_id, answers={"publish": True})

# raise the cap for this attempt and every later one, even though nothing paused
run = resume(run_id, budget=Budget(usd=5.0))

# explicitly accept the risk of resuming against changed flow source
run = resume(run_id, allow_code_change=True)
```

## Human-in-the-loop

`approve(id, payload=None) -> bool` and `ask_human(id, question, payload=None) -> Any` are the two primitives. Both look the answer up in the active run's journal first — a hit returns it (coerced to `bool` for `approve`); a miss pauses the run. Pausing is **not an error**: the process may exit right after it, and a paused span is marked `"paused"` in the trace, never `"error"`.

```python
import composeai as compose


@compose.flow
def publish_flow(draft: str) -> str:
    if compose.approve("publish", payload={"draft": draft}):
        return f"published: {draft}"
    return "kept as draft"
```

Interrupts are **named**, never positional — `id` is a string you pick, and it's what the answer is journaled under, so resuming is idempotent and order-independent regardless of which interrupt a given resume happens to answer.

The same mechanism gates tool calls: `@compose.tool(requires_approval=True)` pauses the agent loop mid-turn on an unanswered call, under the reserved interrupt id `tool:{tool_name}:{call_id}`:

```python
@compose.tool(requires_approval=True)
def send_email(to: str, body: str) -> str:
    """Send an email.

    Args:
        to: Recipient address.
        body: Email body.
    """
    return f"sent to {to}"
```

Resuming with `answers={"send_email": True}` (the bare tool name resolves to the full interrupt id, same shorthand as above) lets the call run; `answers={"send_email": False}` denies it — the model sees a tool result of `"denied by user"` (an error result) and carries on rather than the run failing outright.

Pause/resume works identically whether the pause originated directly in a flow body, inside a nested `@task`, or by propagating up from an `@agent`'s tool call — and across process boundaries exactly like the crash-resume example above: `run.id` is all a second process needs to pick a paused run back up.

## Nested flows

Calling a `@flow` from inside another active `@flow`'s body is a **nested** flow call: it's journaled as one step of the *enclosing* flow, the same way a nested `@agent` call is, rather than starting a brand-new durable run row of its own.

```python
import composeai as compose


@compose.flow
def sub_flow(x: int) -> int:
    return x * 2


@compose.flow
def outer_flow(x: int) -> int:
    doubled = sub_flow(x)   # one journaled step of outer_flow, not a separate run
    return doubled + 1


result = outer_flow(5)   # 11
```

Without this, resuming `outer_flow` would re-execute `sub_flow`'s entire body — and every `@task`/`@agent` call inside it, including paid LLM calls — from scratch on every attempt, even after it had already completed once. On a journal hit, `sub_flow`'s body never runs again at all; on a miss, it runs for real inside its own nested span (same trace, no new `run_id`) and its output is journaled as the outer flow's step value. A pause raised from inside `sub_flow`'s body isn't caught at the call site — it propagates up to whichever flow (outermost or otherwise) is actually running `resume()`/`.run()`, exactly like a pause inside a nested `@task` would.

The same adoption applies to `pipe()`/`aggregate()` results: calling one directly (not `.run()`/`.stream()`) from inside a flow body joins the enclosing run the same way — one trace, usage rolled up, budgets enforced cumulatively, pauses resuming right through it. See [composition](composition.md#inside-a-flow) for the full contract, including the async-body caveat.

## Async: `arun()`, `aresume()`, `anow()`/`arandom()`

`@flow` bodies may be `async def`: call other `@task`/`@agent`s through their own `.arun(...)` twin, read the journal-safe clock/random helpers as `anow()`/`arandom()` instead of `now()`/`random()`, and drive the whole flow via `await flow_obj.arun(...)` / `await aresume(run_id, ...)` — both run natively on your own event loop, never composeai's background runtime thread.

```python
import asyncio
import composeai as compose


@compose.task
async def fetch_sources(topic: str) -> list[str]:
    return [f"https://example.com/{topic}"]


@compose.flow
async def research(topic: str) -> str:
    sources = await fetch_sources.arun(topic)
    ts = await compose.anow()
    return f"{len(sources)} sources as of {ts.isoformat()}"


async def main() -> None:
    run = await research.arun("quantum computing")
    print(run.output)


asyncio.run(main())
```

`anow()`/`arandom()` draw from the exact same per-run key counter as `now()`/`random()` (`now#1`, `now#2`, ... / `random#1`, ...) — a flow body converted from sync to async (or back) still replays whatever the other version already journaled on an earlier attempt. `aresume()` is `resume()`'s async twin: same contract (`answers=`, `budget=` override, `allow_code_change=`), same cross-process pickup by `run_id`, just awaited on your own loop instead of blocking a thread. See [async](async.md) for the full async surface, including async `@task`/`@tool` bodies and nested async flows.

## See also

[agents](agents.md) covers the `@agent` idiom used for `editor`-style steps inside a flow; [composition](composition.md) covers `pipe`/`aggregate`/`map`, all usable (and journaled) inside a flow body; [budgets](budgets.md) covers `Budget` and cumulative spend across `resume()` attempts; [observability](observability.md) covers `compose trace` on a paused run and the exact `resume(...)` call it prints for you; [async](async.md) covers `arun()`/`aresume()` and every other asyncio-native twin in composeai; [typing](typing.md) covers the `Flow[P, R]`/`Run[R]` static-typing contract and how to discriminate paused runs.
