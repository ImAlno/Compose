# Static typing

composeai's composition surface is fully static-typed: `@agent`/`@task`/`@flow`/`@tool` preserve their function's signature, `pipe`/`>>`/`aggregate` thread the input and output types through every stage, and `.run()` hands back a `Run[R]` whose `.output` is your real return type — so a type checker sees the same contract the runtime enforces, and catches a wiring bug in your editor before you ever run it.

## What the checker sees

`@agent` (bare or parenthesized) infers an `AgentFunction[P, R]` that keeps the decorated function's whole parameter list (names included) and its return type; `>>` composes two stages into a `Pipeline[In, Out]`; `.run()` returns a `Run[R]`. Under [pyright/Pylance](#checker-support), the inferred types are:

```python
import dataclasses

import composeai as compose
from composeai.testing import FakeModel


@dataclasses.dataclass
class Facts:
    items: list[str]


@compose.agent
def extract(text: str) -> Facts:
    """Extract facts."""
    return compose.prompt(text)


@compose.agent(model=FakeModel(["unused"]))
def summarize(facts: Facts) -> str:
    """Summarize."""
    return compose.prompt(",".join(facts.items))


reveal_type(extract)                     # AgentFunction[(text: str), Facts]
reveal_type(summarize)                   # AgentFunction[(facts: Facts), str]
reveal_type(extract >> summarize)        # Pipeline[str, str]
reveal_type(extract.run("hi"))           # Run[Facts]
reveal_type(extract.run("hi").output)    # Facts
```

Because the parameter list is preserved, both call forms type-check and calling with the wrong type is a real static error:

```python
extract("hi")        # ok -> Facts
extract(text="hi")   # ok -> Facts
extract(42)          # error: Literal[42] is not assignable to parameter "text" of type "str"
```

`pipe()` infers the same `Pipeline[In, Out]` from the actual stage callables, and `aggregate()` solves its common input type into an `Aggregate[In]` (its output is always `dict[str, Any]`, since branches return heterogeneous types):

```python
import composeai as compose


def to_len(x: str) -> int:
    return len(x)


def to_float(x: int) -> float:
    return x * 1.0


reveal_type(compose.pipe(to_len, to_float))       # Pipeline[str, float]
reveal_type(compose.aggregate(a=to_len, b=to_len))  # Aggregate[str]
```

## The pipe ladder and `>>`

`pipe()` is typed as an overload ladder covering arities **2 through 9** — every rung fully infers `In`/`Out` from the stages you pass — with **no** `Any` fallback overload. That is deliberate: a fallback would silently turn both wrong wiring and an over-length call into an untyped `Pipeline[Any, Any]` with no diagnostic. Without it, a 10-stage `pipe()` is a static no-overload-match (`reportCallIssue`, *"No overloads for `pipe` match the provided arguments"*) even though it builds and runs fine — the ladder stops at nine.

Past nine stages, chain with `>>`, which is unlimited: a `Pipeline` is itself a stage, so `pipe(...) >> next >> next >> ...` keeps threading types with no arity cap.

```python
import composeai as compose


def si(x: str) -> int:
    return len(x)


def if_(x: int) -> float:
    return x * 1.0


def fb(x: float) -> bool:
    return x > 0


def by(x: bool) -> bytes:
    return b"1" if x else b"0"


def bs(x: bytes) -> str:
    return x.decode()


# 12 stages -- well past the arity-9 ladder; binary `>>` has no cap.
long_chain = (
    compose.pipe(si, if_) >> fb >> by >> bs >> si >> if_ >> fb >> by >> bs >> si >> if_
)
reveal_type(long_chain)   # Pipeline[str, float]
```

**Diagnostic quality — prefer `pipe()` over `>>` for the wiring you want checked most precisely.** A mismatch inside the `pipe()` ladder is a `reportArgumentType` that names the exact offending stage parameter (*"Argument of type … cannot be assigned to parameter `s2` of type `Stage[…]`"*). The same mismatch via `>>` surfaces as the terser `reportOperatorIssue` — *"Operator '>>' not supported for types …"* — which flags the line but not the specific parameter. A `>>` on an agent that isn't single-positional-argument (a two-parameter or zero-parameter `@agent`) is also a hard `reportOperatorIssue`, by design: only a single-input agent composes.

## Runtime boundary validation

The static contract has a runtime twin. Every value crossing a `pipe`/`aggregate`/`map` stage boundary is validated on entry against that stage's declared **input** type (pydantic strict mode), at the single dispatch chokepoint, and the coerced value is what flows onward. It is **entry-only** (each stage's input; the pipeline's own input counts as the first stage's), on by default, and a strict no-op for any `Any`/unannotated boundary.

Strict mode keeps the ergonomically useful coercions and rejects the surprising ones:

| Value produced upstream | Stage input type | What happens |
| :--- | :--- | :--- |
| a `dict`, e.g. `{"name": "ai"}` | a pydantic model / dataclass | instantiated into the model (the plain-callable-returns-a-dict idiom keeps working) |
| an instance of the type (or a subclass) | that type | passes, **same object** — identity preserved, never copied [^bool] |
| `int` | `float` | widened to `float` (`3` → `3.0`) |
| `str` | `int` | **`StageTypeError`** — a lossy scalar coercion is refused |
| `float` | `int` | **`StageTypeError`** — lossy, refused |
| `bool` | `int` / `float` | **`StageTypeError`** — the `bool`-subclasses-`int` exception, see below |
| anything | `Any` / unannotated | not validated at all — the exact object flows through untouched |

[^bool]: The one snag in "a subclass passes": `bool` *is* a subclass of `int`, but pydantic strict mode deliberately refuses a `bool` crossing an `int` (or `float`) boundary — `True` reaching an `int`-typed stage is a `StageTypeError`, not a pass-through, and an `int` reaching a `bool`-typed stage is refused too. A boolean is only ever valid at a `bool` boundary.

A mismatch raises `StageTypeError`, naming the boundary it failed at (`pipeline input`, `stage handoff`, `aggregate branch '<name>'`, or `map item <i>`), the stage, the expected type, and pydantic's own field-level detail:

```python
import composeai as compose
from composeai.errors import StageTypeError
from pydantic import BaseModel


class Topic(BaseModel):
    name: str


def head(t: Topic) -> Topic:
    return t


def tail(t: Topic) -> Topic:
    return t


try:
    compose.pipe(head, tail)(42)   # 42 is not a Topic
except StageTypeError as exc:
    print(exc)
    # pipeline input: stage 'head' expected Topic but got an incompatible value -- ...
```

`StageTypeError` is the **runtime** twin of the build-time [`CompositionTypeError`](composition.md): both subclass `ComposeError` *and* `TypeError`, but they mean different things and are kept distinct on purpose. `CompositionTypeError` is raised by `pipe()` for a mismatched *annotation* pair — before anything runs, before any API spend — and fails identically on every retry (a wiring bug). `StageTypeError` is raised while running, for one concrete *value* that didn't validate — and a bad-data bug may not recur. Catch them separately when you need to react differently.

To opt a boundary out of validation entirely, annotate it `Any` (or leave it unannotated) — a deliberately duck-typed stage then pays no validation cost and sees the raw object.

## Paused runs and `Run[R]`

`Run[R].output` is typed as `R` — the **completed-run** output type. A paused or failed run has no output of type `R`, so **never guard on `output`**: because `R` doesn't include `None`, a type checker narrows `run.output` to `Never` inside an `is None` branch (and to `R` outside it), which is worse than useless.

```python
import composeai as compose


@compose.flow
def publish(topic: str) -> str:
    if compose.approve("publish", payload={"topic": topic}):
        return f"published: {topic}"
    return "held"


run = publish.run("ai")

reveal_type(run.output)          # str
if run.output is None:           # DON'T -- R has no None member
    reveal_type(run.output)      # Never
```

Discriminate on `status`/`pending` instead — the fields that actually distinguish the states:

```python
run = publish.run("ai")

if run.status == "paused":
    reveal_type(run.pending)     # Interrupt | None -- answer it, then resume(run.id, ...)
elif run.status == "completed":
    print(run.output)            # a real str here
```

If you genuinely don't know the output type at a site — the classic case is [`resume()`](flows.md), which revives any durable run from a `run_id` alone and is explicitly `-> Run[Any]` — annotate with a **bare** `Run`. A bare (unsubscripted) `Run` means `Run[Any]` (its type parameter carries a PEP 696 `default=Any`), so `.output` is `Any` and an `is None` guard is fine again:

```python
import composeai as compose
from composeai import Run, resume


def continue_it(run_id: str) -> None:
    run: Run = resume(run_id)     # bare Run == Run[Any]
    reveal_type(run.output)       # Any
    if run.output is None:        # fine on Run[Any]
        return
    print(run.output)
```

## Escape hatches and limitations

The typed surface is best-effort where a ParamSpec or overload can't express the whole truth. Known edges:

- **`.run()`-family parameters are loosely typed.** `.run()`/`.arun()`/`.stream()`/`.astream()`/`resume()` all return the precise `Run[R]`/`RunStream[R]`/`AsyncRunStream[R]`, but their *arguments* are `*args: Any` — pyright rejects a keyword-only `budget=` wedged between `*args: P.args` and `**kwargs: P.kwargs`, so the parameter list can't be pinned. The **return** type (the deliverable) is what's typed; the direct call form (`agent(x)`) is what statically checks your arguments.

  ```python
  extract(42)       # error -- direct call checks the argument
  extract.run(42)   # NOT flagged -- .run()'s params are loose; still returns Run[Facts]
  ```

- **Annotating with the object types needs a submodule import.** `Run` and `MapResult` are exported from the top-level `composeai`, but the stage/decorator object types are not — import them from their modules to write an annotation:

  ```python
  from composeai.agentfn import AgentFunction
  from composeai.combinators import Pipeline, Aggregate
  from composeai.flow import Flow, Task
  from composeai.tools import Tool
  from composeai.runs import Run, RunStream, AsyncRunStream
  ```

  Each carries PEP 696 defaults, so a *bare* `Pipeline`/`AgentFunction`/`Flow`/… annotation means `[..., Any]` rather than tripping `reportMissingTypeArgument` under strict mode.

- **An overloaded body loses its `AgentFunction` type.** Decorating a function that carries `@overload` siblings makes the checker report the overload set (not an `AgentFunction[P, R]`), so `.run()`/`.stream()`/`>>` aren't statically visible on it — it still works at runtime. Annotate the result explicitly, or keep agent bodies un-overloaded.

- **A class-body `@agent` keeps `self` in its signature.** `AgentFunction` is not a descriptor, so an `@agent` defined as a method infers `AgentFunction[(self: C, text: str), R]` and does not bind `self` on attribute access. Prefer module-level agent functions.

- **`>>` with a bare lambda is a strict-mode papercut.** Under `strict` pyright, a `lambda` on the right of `>>` triggers `reportUnknownLambdaType` on its parameter even though inference still succeeds (the chain's type is correct). Use a named function in strict codebases, or annotate the lambda's parameter.

- **`aggregate()` accepts a mismatched typed branch statically.** The signature's escape-hatch arm (which keeps bare-lambda branches diagnostic-free) also lets a branch whose input type disagrees with its siblings through the checker; `AggIn` is solved from the first branch. Runtime is unaffected — `Aggregate` falls back to an `Any` input type when its branches disagree, so nothing is lost, just not statically caught.

- **A `Protocol` boundary must be `@runtime_checkable`.** Runtime validation checks a `Protocol`-typed boundary with `isinstance`, which is only legal against a [`@runtime_checkable`](https://docs.python.org/3/library/typing.html#typing.runtime_checkable) protocol. Annotate a stage input with a plain (non-`runtime_checkable`) `Protocol` and `pipe()`/`aggregate()` refuses it at construction with a `ConfigError` naming the fix — decorate the protocol `@runtime_checkable`, or annotate the boundary `Any` to opt out of validation. A `@runtime_checkable` protocol works fully: a structurally-conforming value passes, and a non-conforming one is a `StageTypeError` at the boundary.

- **An unresolvable string annotation isn't validated.** If a stage's input annotation can't be resolved to a real type — the classic case is `from __future__ import annotations` plus a model defined in an enclosing function's local scope, which `typing.get_type_hints` can't see, so it stays a bare string — that boundary silently degrades to unvalidated pass-through, exactly like `Any`. The value flows through untouched rather than crashing an otherwise-valid run. Define such models at module scope if you want the boundary validated.

## Checker support

**[pyright](https://microsoft.github.io/pyright/) / Pylance is the contract.** The whole typed public surface is pinned by `typing_extensions.assert_type` in `tests/typing_surface.py`, which the release gate type-checks with pyright (`reportUnnecessaryTypeIgnoreComment` on, so a negative case that stops firing resurfaces as a fresh error). If pyright infers it, it's guaranteed.

**mypy is a best-effort cross-check, not the contract.** A non-blocking mypy pass over `tests/typing_surface.py` runs at release, but mypy diverges on the composition surface: it solves the contravariant `Stage` input type to `Any` across `>>`/`pipe()`, inferring `Pipeline[Any, Out]`/`Aggregate[Any]` where pyright propagates the concrete left-operand input type. Those divergences are catalogued, with counts and rationale, in `tests/typing_surface.py`'s module docstring — every one is a known checker difference, not a defect in composeai, and none changes runtime behavior. On the negative cases (things that *must* be rejected) mypy and pyright agree — both reject them.

## Migration notes

Two v0.5.0 behavior changes are worth calling out when upgrading:

- **Annotated stage boundaries now validate and coerce at runtime.** Previously a value crossing a `pipe`/`aggregate`/`map` stage boundary was passed through unchecked; now it's strict-validated against the declared input type (see [Runtime boundary validation](#runtime-boundary-validation) above). This is on by default. If a stage was deliberately duck-typed — relying on a value that doesn't match its annotation reaching the body — annotate that boundary `Any` to opt out and restore the old pass-through.

- **…and some now fail at *build* time (or at `map()`/`amap()` entry).** Each boundary's validator is built eagerly when `pipe()`/`aggregate()` constructs the pipeline — or, since `map()`/`amap()` have no construction step, once at each `map()`/`amap()` call *before any item dispatches* — so a stage whose input annotation pydantic can't build a validator for now raises a `ConfigError` *there* instead of building silently and only surfacing later (on `map()`, that would be a raw per-item failure on every item — which `on_error="collect"` would mask). The most common case is a `Protocol` input that isn't `@runtime_checkable` (it can't be an `isinstance` target): a pipeline (or `map()` call) that built and ran in 0.4.1 with such a boundary now refuses up front, and the error names the fix (decorate it `@runtime_checkable`, or annotate the boundary `Any` to opt out). This is by design — the same "no wasted API spend on a wiring bug" ethos as `pipe()`'s compile-time type check.

- **A nested combinator call joins the enclosing run.** Calling a `pipe()`/`aggregate()` result directly (the bare `stage(x)` sugar) from inside a `@flow` body now adopts that run — one trace, usage rolled up, budgets cumulative, pauses resuming through it — instead of starting a separate run. See [composition](composition.md#inside-a-flow) for the full contract (an explicit `.run()`/`.stream()` still mints its own independent run).

## See also

[composition](composition.md) covers `pipe`/`aggregate`/`map` and the build-time `CompositionTypeError`; [agents](agents.md) covers the `@agent` idiom whose signature the `AgentFunction[P, R]` type preserves; [flows](flows.md) covers `@flow`, `resume()`, and the paused-run states `status`/`pending` discriminate; [testing](testing.md) covers the `assert_type` pattern `tests/typing_surface.py` uses to pin these types.
