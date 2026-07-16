# Async

Every composeai entry point that runs an LLM loop or a durable flow has an asyncio-native twin: `.arun()`/`.astream()` on `@agent`, `pipe()`, and `aggregate()`; `.arun()` on `@flow`; `aresume()`, `amap()`, `anow()`/`arandom()`. Both surfaces — the familiar sync one and this async one — drive the exact same `async def` engine underneath; the difference is only *whose* event loop runs it.

## Why two surfaces

- **The sync facade** (`.run()`, `.stream()`, the bare call, `resume()`, `compose.map()`) submits the engine coroutine to composeai's own lazily-started daemon thread ("composeai-runtime"), which runs a persistent event loop, and blocks the calling thread on the result. Because the engine never touches the *caller's* loop, the sync facade works identically in a plain script (no loop at all) and from a thread whose own loop is already running — a Jupyter cell, an ASGI request handler — where a naive `asyncio.run()` would raise `RuntimeError: ... already running`.
- **The async surface** (`.arun()`, `.astream()`, `aresume()`, `amap()`, `anow()`/`arandom()`) runs the SAME engine coroutine directly on **your** already-running loop — no bridge, no extra thread — and never starts composeai's runtime thread at all. Two `agent_fn.arun(...)` calls `asyncio.gather`ed together run concurrently on your loop like any other coroutines.

Pick whichever matches the code you're already writing: sync for a script or a notebook cell, async for code that's already inside `async def` (a web handler, an async batch job).

## The async API tour

### `.arun()` and `.astream()`

```python
import asyncio
import composeai as compose


@compose.agent(model="anthropic/claude-sonnet-5")
def researcher(topic: str) -> str:
    """You are a researcher."""
    return compose.prompt(f"Research: {topic}")


async def main() -> None:
    run = await researcher.arun("quantum computing")
    print(run.output)

    stream = researcher.astream("quantum computing")
    async for event in stream:
        if event.kind == "text_delta" and event.text:
            print(event.text, end="", flush=True)
    final = await stream.run()   # note: a method here, not a property --
    print(final.usage)           # unlike sync RunStream.run


asyncio.run(main())
```

`.arun()` returns a `Run`, exactly like `.run()`. `.astream()` returns an `AsyncRunStream` — `async for` it for live events, the same vocabulary `.stream()` produces (see [agents](agents.md)); `await stream.run()` gets the final `Run`, re-raising the loop's exception if it failed. `Pipeline`/`Aggregate` (from `pipe()`/`aggregate()`) expose the identical `.arun()`/`.astream()` pair — see [composition](composition.md).

### `Flow.arun()` and `aresume()`

```python
import asyncio
import composeai as compose


@compose.task
async def fetch_sources(topic: str) -> list[str]:
    return [f"https://example.com/{topic}"]


@compose.agent(model="anthropic/claude-sonnet-5")
def editor(sources: list[str]) -> str:
    """Turn source summaries into a short draft."""
    return compose.prompt(f"Draft a short paragraph from: {sources}")


@compose.flow
async def research(topic: str) -> str:
    sources = await fetch_sources.arun(topic)
    draft = await editor.arun(sources)
    return draft.output


async def main() -> None:
    run = await research.arun("quantum computing")
    print(run.status, run.output)


asyncio.run(main())
```

`Flow` has no `.astream()` — only `.arun()` alongside the sync `.run()`/`.stream()`. `aresume(run_id, ...)` is `resume()`'s async twin, same contract (`answers=`, `budget=` override, `allow_code_change=`) and the same cross-process pickup by `run_id`, just awaited on your own loop instead of blocking a thread:

```python
from composeai import aresume


async def continue_it(run_id: str) -> None:
    run = await aresume(run_id, answers={"publish": True})
    print(run.status)


asyncio.run(continue_it("01K3F8G7QZR3XJ8N4V0T5W2Y1B"))
```

### `amap()`

```python
import asyncio
import composeai as compose


async def fetch(url: str) -> str:
    return f"contents of {url}"


async def main() -> None:
    results = await compose.amap(fetch, ["https://a", "https://b"])
    print(results)


asyncio.run(main())
```

`amap()` mirrors `map()`'s exact contract — `max_workers`, `timeout_per_item`, `on_error="raise"|"collect"` — just awaited directly instead of bridged through composeai's runtime thread. See [composition](composition.md).

### `anow()` and `arandom()`

```python
import composeai as compose


@compose.flow
async def maybe_offer_discount(customer_id: str) -> str:
    ts = await compose.anow()
    if await compose.arandom() < 0.1:
        return f"{customer_id} gets a discount as of {ts.isoformat()}"
    return f"{customer_id}: no discount"
```

`anow()`/`arandom()` are the journal-safe async twins of `now()`/`random()` — and they draw from the *same* per-run key counter (`now#1`, `now#2`, ... / `random#1`, `random#2`, ...). A flow body converted from sync to async (or back) still replays whatever the other version already journaled on an earlier attempt: `now()` and `anow()` are interchangeable across a resume, not just within one attempt. Outside a flow, `anow()` is exactly `datetime.now(timezone.utc)` and `arandom()` is exactly `random.random()`, same as the sync versions.

## Async bodies

`@tool`, `@task`, `@agent`, and `@flow` bodies may all be `async def` — composeai detects it and awaits the body natively, on the sync facade or the async surface alike, because the engine underneath is always async either way:

```python
import asyncio
import composeai as compose


@compose.tool(timeout=5.0)
async def fetch_url(url: str) -> str:
    """Fetch a URL's contents.

    Args:
        url: The URL to fetch.
    """
    return f"contents of {url}"


@compose.task
async def summarize(text: str) -> str:
    return f"summary of {text}"


@compose.agent(model="anthropic/claude-sonnet-5", tools=[fetch_url])
async def researcher(topic: str) -> str:
    """You are a researcher."""
    return compose.prompt(f"Research: {topic}")


@compose.flow
async def research(topic: str) -> str:
    draft = await researcher.arun(topic)
    return await summarize.arun(draft.output)
```

`@tool(timeout=...)` on an async body cancels *cooperatively* (a real `CancelledError` thrown via `asyncio.wait_for`) instead of racing a daemon thread the way a timed-out sync tool does — the async body leaves no zombie thread behind (a timed-out sync body still does; see "Threads that still exist" below), and the model still just sees the same `is_error` tool result.

**Nested flows, both directions:** sync sugar (`inner(...)`) calling an async-bodied inner flow, and `await inner.arun(...)` calling either a sync- or async-bodied inner flow, both journal the nested call as one step of the enclosing run — exactly like a nested sync call always has. A nested `await inner.arun(budget=...)` raises `ConfigError` rather than silently ignoring the budget: a nested flow step can't take its own cap, the enclosing run's budget already governs it.

**The `run()`/`arun()` nested asymmetry:** `Flow.run()` never checks for an ambient flow context at all — called from inside another flow's body, it always mints a brand-new, independently-resumable top-level run, unlike the sugar call or `arun()`. This is specific to `Flow`; `Task`/`AgentFunction` don't have it — their `run()`-family and `arun()`-family methods both check for an ambient flow the same way, symmetrically. Prefer the sugar call or `.arun()` inside a flow body when you want a nested step; reach for `.run()` there only when a genuinely separate run is what you actually want.

## Mixing surfaces

Both directions work:

- **Sync sugar inside a running loop** (the Jupyter/ASGI case) — calling `agent_fn(...)`/`flow_obj(...)`/`.run()` from inside a coroutine that's already running on a loop just works: the sync facade bridges onto composeai's runtime thread's OWN loop, never the caller's, so there's no `RuntimeError: ... already running`.

  ```python
  import asyncio
  import composeai as compose


  @compose.agent(model="anthropic/claude-sonnet-5")
  def researcher(topic: str) -> str:
      """You are a researcher."""
      return compose.prompt(f"Research: {topic}")


  async def notebook_cell() -> None:
      fact = researcher("quantum computing")   # sync sugar -- fine, even here
      print(fact)


  asyncio.run(notebook_cell())
  ```

- **Async bodies inside sync flows** — a plain sync `@flow` may call an `async def` `@task`/`@agent` through ordinary sync sugar (`afetch(x)`, not `await afetch.arun(x)`); it bridges through the runtime loop from the sync flow body's own dedicated worker thread, journaling exactly like a sync task would.

## Cancellation & Ctrl-C

Cancelling an in-flight `arun()`/`astream()` (via `Task.cancel()` on your own loop, or Ctrl-C on a sync facade call) throws a real `asyncio.CancelledError` into the engine coroutine's await chain — the same mechanism a `timeout=` uses. The run's durable row still lands cleanly: it's marked `"failed"`, with the error message synthesized as `"cancelled"` when the underlying exception's own message is empty (a plain `CancelledError()` stringifies to `""`, which would otherwise leave `compose runs`/`compose trace` with nothing to render) — resumable with `resume()`/`aresume()` like any other failed run.

One caveat: a sync `@task`/`@flow` body already dispatched onto its own dedicated worker thread (see "Threads that still exist" below) can't be force-stopped once cancellation reaches the coroutine awaiting it — Python threads don't support that. It keeps running in the background, but the instant the awaiting coroutine is cancelled, that thread's journal write access is revoked (the same abandon-guard `@task(timeout=...)` installs) — any journal write it attempts after that point raises instead of landing under the cancelled attempt's keys, so a zombie stage can never corrupt a resumed attempt's journal.

Ctrl-C on a **sync** facade call now cancels the current attempt the same way — see [flows](flows.md#ctrl-c-mid-run) for the full walkthrough of what a `KeyboardInterrupt` does to the durable row. Ctrl-C reaching your own code while it's inside an **async** `await ...arun(...)`/`await ...astream(...)` call is nothing composeai-specific: it's whatever `asyncio.run()`/your event loop already does with `KeyboardInterrupt`, same as any other `await`.

## Threads that still exist

Even on the async surface — which never starts composeai's runtime thread — a couple of background threads remain, neither of them optional:

- **A sync `@tool`/`@task`/`@flow` body (or a sync-only model adapter) still dispatches onto its own dedicated stage worker thread** whenever the async engine reaches it — deliberately one thread per in-flight sync stage, not a shared pool: a shared, bounded executor would deadlock the moment a sync stage calls back into a composeai facade with the pool already full.
- **Every durable store write goes through one dedicated writer thread per store** ("composeai-store") — journal writes, span persistence, and row updates all funnel through it, in order, off a queue — never a direct write from whichever thread happened to produce the data.

Neither is composeai's runtime thread, and neither is "no background thread at all": the async surface only guarantees that the agent/flow/task *loop itself* runs on your loop, not that everything it touches does.

## See also

[agents](agents.md) covers `.run()`/`.stream()`, the sync twin of `.arun()`/`.astream()`; [flows](flows.md) covers `@flow`, `resume()`, and the journal that `aresume()` shares; [composition](composition.md) covers `pipe`/`aggregate`/`map`, whose async twins appear above; [budgets](budgets.md) covers `Budget`, honored identically by `aresume()`.
