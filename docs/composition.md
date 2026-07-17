# Composition

`pipe`, `aggregate`, and `map` wire agents (or plain callables, or other pipes/aggregates) together into larger stages, with every consecutive connection type-checked before anything runs.

## `pipe`

`pipe(*stages)` chains stages in sequence — the first stage's output feeds the second stage's input, and so on. A stage is any `@agent` function, a plain callable, or another `pipe`/`aggregate` result:

```python
@compose.agent(model="openai/gpt-5.6-luna")
def copywriter(sheet: FactSheet) -> str:
    """Turn a fact sheet into one punchy line."""
    return f"Write one punchy line from: {sheet.model_dump_json()}"


write_post = compose.pipe(researcher, copywriter)   # types checked HERE
post = write_post("quantum computing")
```

The killer feature is that `pipe()` checks every consecutive stage pair for type compatibility *at build time*, using each stage's `.input_type`/`.output_type`. A wiring bug never gets to run — and never spends a token:

```python
from composeai.errors import CompositionTypeError

try:
    compose.pipe(researcher, researcher)
except CompositionTypeError as exc:
    print(exc)
    # pipe(): stage 1 (researcher) returns FactSheet but stage 2 (researcher) expects str
```

`pipe()` requires at least 2 stages, or it raises `CompositionTypeError` immediately.

That build-time check has a runtime twin. As each stage runs, the concrete value handed to it is validated against the stage's declared input type (pydantic strict mode) at the single dispatch chokepoint, and the coerced value flows onward — a `dict` is instantiated into a model, an `int` widens to a `float`, an already-correct instance passes through untouched, but a lossy or shape-wrong value raises `StageTypeError`:

```python
from composeai.errors import StageTypeError

try:
    write_post({"not": "a topic"})   # wrong shape for researcher's input
except StageTypeError as exc:
    print(exc)
    # pipeline input: stage 'researcher' expected ... but got an incompatible value -- ...
```

`StageTypeError` is `CompositionTypeError`'s runtime counterpart (both subclass `ComposeError` and `TypeError`, kept distinct so a wiring bug and a bad-data bug can be caught separately). A boundary annotated `Any` (or unannotated) opts out — it's never validated. See [typing](typing.md) for the full coercion table and the static-typing contract these checks mirror.

### Operator sugar

`a >> b` is exactly `pipe(a, b)` — same build-time type check, same `CompositionTypeError` on a mismatch, no separate code path. It works for any mix of `@agent` functions, plain callables, and `pipe`/`aggregate` results, as long as at least one operand is a composeai stage type; chaining stays flat, so `researcher >> copywriter >> editor` builds one 3-stage `Pipeline`, not a `Pipeline` nested inside a `Pipeline`:

```python
write_post = researcher >> copywriter   # exactly pipe(researcher, copywriter), types checked HERE
post = write_post("quantum computing")
```

Both forms are also statically typed: `pipe()`/`>>` infer a `Pipeline[In, Out]` a type checker can see, so a mismatch is flagged in your editor, not just at build time. `pipe()`'s overload ladder covers 2–9 stages (a longer `pipe()` call is a static no-match by design — chain with `>>`, which has no arity cap). See [typing](typing.md).

## `aggregate`

`aggregate(**branches)` runs every named branch in parallel threads and gathers a `{name: output}` dict, in declaration order:

```python
audits = compose.aggregate(words=lambda s: len(s.split()), chars=len)
audits("count these words")   # {'words': 3, 'chars': 17}
```

Every branch settles (success or exception) before `aggregate()` returns. On any failure, the exception from the first branch *in declaration order* is raised — regardless of which branch actually finished first or failed first in real time.

`aggregate(timeout_per_branch=..., **branches)` bounds each branch with its own timeout (seconds); a branch that runs longer raises `TaskTimeoutError` under the same first-branch-in-declaration-order rule:

```python
audits = compose.aggregate(
    timeout_per_branch=5.0,
    words=lambda s: len(s.split()),
    chars=len,
)
```

Without `timeout_per_branch`, a single hung branch blocks the whole `aggregate()` call (and the enclosing run) until it finishes or the process is killed — give an individual stage its own bound with `@task(timeout=...)` if it needs one and you aren't using `timeout_per_branch`. One consequence of `timeout_per_branch` being a keyword parameter alongside `**branches`: a branch cannot itself be named `timeout_per_branch`. `aggregate()` requires at least 1 branch, or it raises `CompositionTypeError`.

## `map`

`map(fn, items, ...)` applies one stage to many items in parallel, preserving input order:

```python
compose.map(summarize, sources)      # one stage over many items, order preserved
```

Two keywords beyond the basic fan-out:

- `max_workers` (default: `None`, meaning one worker per item) caps how many items run concurrently.
- `timeout_per_item` (seconds, default `None`) races each item on its own thread and raises `TaskTimeoutError` for just that one instead of blocking every other item forever.

`on_error="collect"` replaces the default `on_error="raise"` behavior (the first failure by index is re-raised once every item has settled) with a `list[MapResult]` — one entry per item, in input order, with nothing raised:

```python
results = compose.map(summarize, sources, on_error="collect", timeout_per_item=30)
ok = [r.value for r in results if r.ok]
```

`MapResult` fields: `ok: bool`, `value: Any = None`, `error: str | None = None`, `error_type: str | None = None`. Errors are carried as strings, never exception objects, so a collected result is safe to journal and replay.

Inside a `@flow`, `map()` already journals each item individually as it completes — a failed item (whether it raises under `on_error="raise"`, or is just recorded as a failed `MapResult` under `on_error="collect"`) never discards its siblings' completed work; only the unfinished tail re-runs on `resume()`. That's true when each item is itself a journaling stage (`@task`/`@agent`/nested `@flow`/`pipe`/`aggregate`) — a plain Python callable has no journal entry of its own to replay, so its item just re-runs on `resume()` like any other unwrapped code. See [flows](flows.md) for the journal in depth.

## Nesting combinators

Stages compose recursively: a `pipe()` can contain an `aggregate()`, an `aggregate()`'s branches can each be a `pipe()`, and `map()`'s `fn` can be any of the above — routing is an `if` statement, not a graph edge class:

```python
research_and_summarize = compose.pipe(
    compose.aggregate(words=lambda s: len(s.split()), chars=len),
    copywriter,
)
```

Every nested `Pipeline`/`Aggregate` exposes the same `.input_type`/`.output_type` a plain `@agent` does, so build-time type checking applies uniformly no matter how deep the nesting goes.

## Async: `.arun()`/`.astream()`, `amap()`

Every `pipe()`/`aggregate()` result also exposes `.arun()`/`.astream()` alongside `.run()`/`.stream()`, running natively on your own event loop instead of composeai's background runtime thread. `compose.map()` has its own async twin, `amap()` — identical contract (`max_workers`, `timeout_per_item`, `on_error`), just awaited directly instead of bridged through that thread. See [async](async.md) for the full async surface.

## Inside a flow

Calling a `pipe()`/`aggregate()` result directly — the bare `write_post(topic)` sugar, not `.run()`/`.stream()` — while a `@flow` body (or any other active span: a bare `@task`, another `@agent` call, ...) is already running joins that enclosing run instead of starting a new one: one trace, usage rolled up into the enclosing run's `.usage`, `Budget` enforced cumulatively against the whole run's spend, and a pause inside a nested stage (an `approve()`/`ask_human()` call) resumes right through it, exactly like a pause inside a nested `@task` does:

```python
@compose.flow
def publish_post(topic: str) -> str:
    post = write_post(topic)   # joins publish_post's own run/trace -- no separate row
    return post
```

`write_post`'s stages land as spans nested under `publish_post`'s own root span in the same trace — `compose trace` shows them as children, not a separate run.

An explicit `.run()`/`.stream()` call is different: it always starts its own independent run and trace, even from inside an active flow — exactly like `Flow.run()` does when called from inside another flow (see [flows](flows.md#nested-flows)). Reach for `.run()`/`.stream()` there only when a genuinely separate run is what you want; use the bare call for a nested step.

One caveat: inside an `async def` `@flow`/`@task` body being driven through the sync `.run()` facade, the bare combinator sugar (`write_post(topic)`) raises `RuntimeError: composeai runtime cannot be re-entered from its own loop thread` — the async body already runs directly on composeai's own runtime loop, so the sugar's bridge back onto that same loop is refused — use `await write_post.arun(topic)` there instead; see [async](async.md).

## See also

[agents](agents.md) covers the `@agent` idiom these stages are usually built from; [flows](flows.md) makes a sequence of combinator calls durable and resumable; [budgets](budgets.md) covers `budget=` on a top-level `pipe`/`aggregate` `.run()` call; [typing](typing.md) covers the static-typing contract (`Pipeline[In, Out]`, the pipe ladder) and the `StageTypeError` runtime validation these boundaries perform.
