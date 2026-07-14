# Budgets

`Budget(usd=..., tokens=...)` caps spend across every LLM call in a run's subtree — pass it to `.run()`/`.stream()` (or the bare call form) on an `@agent`, a `pipe`/`aggregate`, or a `@flow`, and composeai raises `BudgetExceededError` the instant a call pushes the running total past the cap.

## `Budget(usd=, tokens=)`

```python
import composeai as compose
from composeai import Budget


@compose.agent(model="anthropic/claude-sonnet-5")
def researcher(topic: str) -> str:
    """You are a researcher."""
    return compose.prompt(f"Summarize: {topic}")


researcher.run("quantum computing", budget=Budget(usd=0.50, tokens=200_000))
```

At least one of `usd`/`tokens` must be set — an empty `Budget()` couldn't enforce anything, so the constructor raises `ConfigError` immediately. Pass either alone, or both:

```python
Budget(usd=1.0)                     # dollars only
Budget(tokens=100_000)              # tokens only
Budget(usd=1.0, tokens=100_000)     # both -- either crossing the line trips it
```

Enforcement happens after every LLM call across the whole run's subtree, not just at the top level: an `@agent` nested inside a `pipe`, an `aggregate` branch, or a step inside a `@flow` all count against a budget passed to an enclosing `.run()` call. `check_budgets()` walks every active budget on the stack each time — a nested, tighter budget and an enclosing, looser one are both live at once (see "Nesting" below).

## What counts

`tokens` is input+output tokens combined, summed across every LLM call in the subtree. This is exact regardless of pricing — a token budget sees every call's usage no matter what model it hit.

`usd` only counts calls with a known price. An adapter that can't price a call reports `Usage(cost_usd=None)`, and that call's cost is treated as `0` for budgeting purposes — a `usd` budget simply can't see spend it has no price for. This is the same unpriced-model caveat `compose costs` has: cost is never fabricated, so a call with no registered price is invisible to `Budget(usd=...)`, not silently counted as free spend against you.

If you're on an OpenAI-compatible server with a real bill (ollama.com, a hosted vLLM instance) and want `usd` to actually see that spend, register a price — either `input_price=`/`output_price=` on `compose.openai_compatible(...)`, or `composeai.register_price(provider, model, composeai.ModelPrice(input=..., output=...))` directly. See [providers](providers.md) for both forms. **Pass `tokens` too if you need a hard cap regardless of pricing** — it's the one budget dimension that never depends on a price table.

## Cumulative spend across `resume()`

A `budget=` passed to a run's original `.run()` call stays enforced across every later `resume()` of that run: spend already persisted by earlier attempts counts against the cap, and replayed steps themselves cost nothing (they don't re-issue LLM calls, so they add zero to the running total). A run can't dodge its budget by pausing or crashing and getting resumed with the clock reset.

Worked example — a two-step flow under a 15-token budget, each step scripted to cost exactly 10 tokens (5 in + 5 out) via `FakeModel`:

```python
import composeai as compose
from composeai import Budget, resume
from composeai.messages import Usage
from composeai.testing import FakeModel

model = FakeModel(["first", "second"], usage=Usage(input_tokens=5, output_tokens=5))


@compose.agent(model=model)
def spender(prompt: str) -> str:
    """Spend some tokens."""
    return prompt


@compose.flow
def two_step() -> str:
    spender("one")                       # 10 tokens -- lifetime total 10, under the 15 cap
    if not compose.approve("continue"):
        return "stopped"
    spender("two")                       # 10 more -- lifetime total 20, over the cap
    return "done"


run = two_step.run(budget=Budget(tokens=15))
run.status   # "paused" -- the first spend landed at 10, still under the cap

# resume() re-attempts the flow: the first step replays (0 new tokens,
# journaled), then the second step actually runs, adding 10 more on top of
# the persisted 10 from the paused attempt -- lifetime total 20 exceeds the
# 15-token cap, so this raises instead of quietly completing:
resume(run.id, answers={"continue": True})   # raises BudgetExceededError
```

Without this accumulation, a resumed attempt would start counting from zero and could complete happily inside a cap the run's *total* spend has already blown past. The same rule applies one level down: a nested `@agent`'s *own* `budget=` (passed on its own `.run()` call, not the flow's) is cumulative across a pause/resume of that same step specifically — only that step's earlier attempts count against its own cap, never a sibling step's spend.

## Per-call agent budgets vs. flow budgets (nesting)

A budget passed to an `@agent`'s own `.run()`/`.stream()` call and a budget passed to an enclosing `pipe`/`aggregate`/`@flow`'s `.run()` are both enforced simultaneously — nesting doesn't replace the outer cap, it adds a second one:

```python
import composeai as compose
from composeai import Budget
from composeai.messages import Usage
from composeai.testing import FakeModel

inner_model = FakeModel(["inner out"], usage=Usage(input_tokens=10, output_tokens=10))
outer_model = FakeModel(["outer out"], usage=Usage(input_tokens=10, output_tokens=10))


@compose.agent(model=inner_model)
def inner_agent(x: str) -> str:
    """Inner."""
    return x


@compose.agent(model=outer_model)
def outer_agent(x: str) -> str:
    """Outer."""
    return x


def middle_stage(x: str) -> str:
    # A tighter budget than the enclosing pipe's -- both are checked.
    run = inner_agent.run(x, budget=Budget(tokens=5))
    return run.output


pipeline = compose.pipe(outer_agent, middle_stage)
pipeline.run("go", budget=Budget(tokens=1000))   # generous outer cap...
# ...but inner_agent's own 5-token budget trips first, since it's tighter.
```

Whichever budget's cap is crossed first raises — an inner, tighter budget commonly trips before an outer, looser one ever would, but there's no ordering rule beyond "whoever crosses their own line first."

## `resume(budget=)` override

`resume(run_id, budget=...)` replaces the run's stored budget for this attempt **and every later one** — a plain last-write-wins update to the row, not journaled (the journal is first-write-wins, which would otherwise freeze the very first override forever):

```python
resume(run.id, answers={"continue": True}, budget=Budget(usd=5.0))
```

Prior attempts' real spend still counts against the new cap — raising the limit doesn't erase what was already spent, it only moves the ceiling. Omitting `budget=` (the default, `None`) keeps whatever budget was already stored; there's no way to *clear* a budget via `resume()` once one is set.

## `BudgetExceededError` handling

`BudgetExceededError` is a `ComposeError`, raised by the budget check immediately after the LLM call that crossed the line — the call itself already happened (and was already billed/journaled); the error just stops the run from making another one. Its message names the span that tripped it, plus the configured cap and the usage that exceeded it:

```python
from composeai.errors import BudgetExceededError

try:
    researcher.run("quantum computing", budget=Budget(usd=0.50))
except BudgetExceededError as exc:
    print(exc)
    # budget exceeded on agent 'researcher': usd budget=0.5, used=$0.6231
```

Inside a `@flow`, an uncaught `BudgetExceededError` fails that run the same way any other uncaught exception would — the run's row is marked `"failed"` and the exception propagates out of `.run()`/`resume()`. A common pattern is to catch it at the call site, inspect what already ran (`compose trace <run_id>`), and either raise the cap and `resume()`, or accept the partial result and stop.

## See also

[agents](agents.md) covers `.run()`/`.stream()`'s other keyword-only arguments; [composition](composition.md) covers `budget=` on a top-level `pipe`/`aggregate` call; [flows](flows.md) covers `resume()` in full, including `answers=`/`allow_code_change=`; [providers](providers.md) covers `input_price=`/`register_price` for making `usd` spend visible on an unpriced or compatible-server model.
