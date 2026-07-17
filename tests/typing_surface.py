"""Static-typing surface showcase for composeai v0.5.0 (Plan B, Task 6).

This file is a *pure static-typing artifact* -- the composeai analogue of
PydanticAI's ``tests/typed_agent.py``. It contains **zero pytest functions**
and is therefore never collected or executed by the suite (pytest's default
``python_files`` is ``test_*.py``/``*_test.py``; ``typing_surface.py`` matches
neither), so the exact suite count is unchanged by its existence. Its whole job
is to be *type-checked*: it exercises the ENTIRE typed public surface in one
place so a single ``pyright`` run (the release gate -- ``[tool.pyright]`` in
pyproject.toml already ``include``s ``tests``) proves the surface still infers
what the design promises.

How it verifies, two ways at once (the pydantic PR #10092 / PydanticAI
convention):

* **Positive shapes** use ``typing_extensions.assert_type`` -- a hard, symmetric
  equality assertion on the inferred type. If any surface stops inferring the
  pinned type, ``assert_type`` fails *in this same gate*.
* **Negative cases** (things that MUST be rejected -- ``pipeline(42)``, wrong
  wiring, a 2-param agent ``>>``, a 10-stage ``pipe()``, a wrong-arg ``@tool``
  call) are ordinary lines carrying a rule-scoped ``# pyright: ignore[<rule>]``
  -- never bare. With ``reportUnnecessaryTypeIgnoreComment = true`` (also in
  ``[tool.pyright]``), a wrong rule name OR a diagnostic that stops firing both
  resurface as a fresh error here, so every negative is self-verifying too.

Engine-driving calls (``.run()``/``.arun()``/``.stream()``/``.astream()``/
``resume()``/``__call__``) live inside **uncalled** helper functions
(``_surface_*``) -- pyright still analyzes their bodies, but nothing ever drives
the real agent/durable engine even if this module were imported. The decorated
definitions and the composition-time-only ``pipe()``/``>>`` builds sit at module
scope. ``assert_type`` is itself a runtime no-op (it returns its first argument
unchanged), so none of this has any runtime effect regardless.

The prototype ``reveal_type`` transcript
(``plans/superpowers/research/2026-07-16-typing/prototype/reveals_basic.txt``)
is the acceptance oracle these pins reproduce on the REAL classes; the full
line-by-line parity audit lives in this task's report. Deliberate,
spec-adjudicated deviations from the prototype -- all of which are the
prototype exploring an arm the design then rejected -- are:

* **T1 (PEP 696 default):** ``Run``/``MapResult``/``Pipeline``/``Aggregate``/
  ``AgentFunction``/``Task``/``Flow``/``Tool`` TypeVars carry ``default=Any``,
  so a *bare* annotation means ``[..., Any]`` (``Run[Any]``, ...) rather than
  the prototype's pre-default ``Run[Unknown]``.
* **T3 (no ``Any`` fallback):** the ``pipe()`` ladder is arities 2..9 with NO
  ``Any`` fallback overload, so wrong wiring / a multi-param stage / a 10-stage
  call are *diagnostics* here, not the prototype ``pipe()``'s silent
  ``Pipeline[Any, Any]``. ``aggregate()``'s ``Stage[AggIn, Any] |
  Callable[[Any], Any]`` shape (keeps bare-lambda branches clean; accepts a
  mismatched typed branch) and ``map``/``amap``'s async-stage
  (``Callable[[A], Awaitable[B]]``) overload arms are the other two Task-3
  adjudications (neither surface was in the prototype).
* **Variant A (``__rshift__``):** an ``AgentFunction`` composes with ``>>`` only
  when single-positional-arg (``self: AgentFunction[[X], R2]``); a 2-/0-param
  agent is a hard ``reportOperatorIssue`` -- chosen over the prototype's B/C/D/E
  arms, which silently mis-typed multi-arg composition.

Known mypy divergences (best-effort, non-blocking smoke)
--------------------------------------------------------

A best-effort, non-blocking mypy pass over just this file is a Plan C release
gate (``scripts/release.sh`` runs ``mypy tests/typing_surface.py || echo ...``
with visible output; ``[tool.mypy]`` in pyproject.toml scopes it to this file
with ``follow_imports = "silent"`` so it checks the surface against composeai's
REAL types without demanding the whole repo be mypy-clean). **pyright is the
contract; mypy is a cross-checker smoke test only.**

As of mypy 2.3.0 this file reports **25 errors** (``mypy`` from the repo root).
All 25 are EXPECTED and fall into four classes -- none is a defect in this file
or in ``src``, and none is fixable without either weakening a pyright
``assert_type`` pin or hand-annotating away the very inference this file exists
to showcase (the Plan B final review predicted the Stage-protocol variance
complaints; the directive is to document, not contort). No line was changed to
appease mypy.

====================  =====  ==========================================================  ========
mypy error class      count  why it diverges from pyright                                verdict
====================  =====  ==========================================================  ========
``[assert-type]``     9      mypy solves the contravariant ``Stage.In`` to ``Any``       expected
                             across ``>>`` / ``pipe()`` composition, inferring
                             ``Pipeline[Any, Out]`` / ``Aggregate[Any]`` where pyright
                             propagates the left operand's concrete input type -- the
                             Stage-protocol variance the Plan B review predicted.
``[var-annotated]``   8      Downstream of the row above: mypy will not *infer* a var    expected
                             whose solved type is a partial-``Any`` generic (or a
                             rejected ``>>`` result) and demands an explicit
                             annotation. Adding one would turn this inference showcase
                             into a hand-annotated file.
``[operator]``        1      ``plain-fn >> agent``: mypy does not resolve the reflected  expected
                             ``>>`` into ``AgentFunction.__rrshift__`` (it reports
                             ``Stage[Never, Never]``) -- a known mypy limit with
                             reflected operators vs protocol self-types; pyright gives
                             ``Pipeline[int, Facts]``.
negative cases        7      The ``# pyright: ignore``-marked negatives: ``[arg-type]``  expected
                             x3, ``[misc]`` x3, ``[call-overload]`` x1
                             (``pipeline(42)``, wrong wiring, 10-stage ``pipe()``,
                             2-/0-param agent ``>>``, wrong-arg ``@tool``). mypy
                             honours no scoped-ignore so they count -- but it REJECTS
                             every one, the same verdict as pyright.
====================  =====  ==========================================================  ========
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any

from typing_extensions import assert_type

import composeai as compose
from composeai import Budget, MapResult, Run, agent, flow, resume, task, tool
from composeai.agentfn import AgentFunction
from composeai.combinators import Aggregate, Pipeline
from composeai.flow import Flow, Task
from composeai.runs import AsyncRunStream, RunStream
from composeai.testing import FakeModel
from composeai.tools import Tool


@dataclasses.dataclass
class Facts:
    items: list[str]


@dataclasses.dataclass
class Report:
    title: str
    body: str


# ===========================================================================
# @agent -- both decorator forms (bare + parenthesized), ParamSpec[P, R]
# Prototype oracle: probe_chain.py:24-25, probe_multiparam.py:12.
# ===========================================================================


@agent
def sfc_extract(text: str) -> Facts:
    return Facts(items=[text])


@agent(model=FakeModel(["unused"]), retries=2)
def sfc_extract_cfg(text: str) -> Facts:
    return Facts(items=[text])


@agent(model=FakeModel(["unused"]))
def sfc_summarize(facts: Facts) -> str:
    return ",".join(facts.items)


@agent
def sfc_two_param(a: str, b: int) -> str:
    return a * b


@agent
def sfc_no_param() -> str:
    return "x"


@agent
def sfc_make_int(text: str) -> int:
    return len(text)


# ===========================================================================
# Plain-function stages forming a 5-cycle str->int->float->bool->bytes->str,
# so pipe()/`>>` chains of any length can be wired from them.
# ===========================================================================


def sfc_si(x: str) -> int:
    return len(x)


def sfc_if(x: int) -> float:
    return x * 1.0


def sfc_fb(x: float) -> bool:
    return x > 0


def sfc_by(x: bool) -> bytes:
    return b"1" if x else b"0"


def sfc_bs(x: bytes) -> str:
    return x.decode()


def sfc_shout(s: str) -> str:
    return s.upper()


def sfc_wants_bool(b: bool) -> str:  # for the wrong-wiring negative
    return str(b)


# --- @agent: keyword + positional calls stay typed; bare annotation is Any ---


def _surface_agent_calls() -> None:
    # ParamSpec preserves the parameter name, so BOTH call forms type-check and
    # return `R` (probe_lambda.py:59-62). Bare and parenthesized forms agree.
    assert_type(sfc_extract("hi"), Facts)
    assert_type(sfc_extract(text="hi"), Facts)
    assert_type(sfc_extract_cfg("hi"), Facts)
    assert_type(sfc_extract_cfg(text="hi"), Facts)


def _surface_agent_bare_annotation_is_any(af: AgentFunction) -> None:
    # A bare (unsubscripted) `AgentFunction` means `AgentFunction[..., Any]`
    # (T1 PEP 696 default on P/R), never `reportMissingTypeArgument`.
    assert_type(af("anything", 1, 2), Any)


# --- @agent Run[R] surface: run / arun / stream / astream + .output ---------


def _surface_agent_run() -> None:
    # `.run(...)`/`.stream(...)` -> `Run[R]`/`RunStream[R]` (R=Facts). The
    # keyword-only `budget=` leaves the `Run[R]` return -- the deliverable --
    # untouched.
    assert_type(sfc_extract.run("hi"), Run[Facts])
    assert_type(sfc_extract.run("hi").output, Facts)
    assert_type(sfc_extract.run("hi", budget=Budget(tokens=10)).output, Facts)
    assert_type(sfc_extract.stream("hi"), RunStream[Facts])
    assert_type(sfc_extract.stream("hi").run, Run[Facts])


async def _surface_agent_arun_astream() -> None:
    assert_type(await sfc_extract.arun("hi"), Run[Facts])
    assert_type((await sfc_extract.arun("hi")).output, Facts)
    assert_type(sfc_extract.astream("hi"), AsyncRunStream[Facts])
    assert_type((await sfc_extract.astream("hi").run()).output, Facts)


# ===========================================================================
# `>>` composition -- Variant A (single-arg agents compose; multi/0-arg reject)
# Prototype oracle: probe_chain.py:37-49,87-94; probe_multiparam.py:29-39.
# ===========================================================================

# agent >> agent (bare + parenthesized both infer AgentFunction[(text: str), Facts]).
sfc_p2 = sfc_extract >> sfc_summarize
assert_type(sfc_p2, Pipeline[str, str])
sfc_p2_cfg = sfc_extract_cfg >> sfc_summarize
assert_type(sfc_p2_cfg, Pipeline[str, str])

# agent >> agent >> plain-fn (second `>>` is Pipeline.__rshift__).
sfc_p3 = sfc_extract >> sfc_summarize >> sfc_shout
assert_type(sfc_p3, Pipeline[str, str])

# plain-fn >> agent resolves through AgentFunction.__rrshift__ (the agent has
# no left operand to own the `>>`): sfc_prefix (int->str) feeds sfc_extract
# (str->Facts) -> Pipeline[int, Facts].
def sfc_prefix(n: int) -> str:
    return str(n)


sfc_pr = sfc_prefix >> sfc_extract
assert_type(sfc_pr, Pipeline[int, Facts])


# A 6-stage `>>` chain built purely from plain functions, seeded by a Pipeline
# (a plain-fn `>>` plain-fn has no operator; the seed supplies __rshift__).
sfc_p6c = sfc_extract >> sfc_summarize >> sfc_si >> sfc_if >> sfc_fb >> sfc_by
assert_type(sfc_p6c, Pipeline[str, bytes])

# lambda as the final stage: Stage's positional-only `__call__` lets a bare
# lambda match; `len` fixes the result to int (probe_chain.py:94).
sfc_p_lambda = sfc_extract >> sfc_summarize >> (lambda s: len(s))
assert_type(sfc_p_lambda, Pipeline[str, int])


def _surface_pipeline_call_returns_out() -> None:
    assert_type(sfc_p2("hello"), str)
    assert_type(sfc_p6c("hello"), bytes)


# ===========================================================================
# pipe() overload ladder -- arities 2..9, fully inferred (NO Any fallback).
# Prototype oracle: probe_chain.py:82-90 (past-ladder via `>>`).
# ===========================================================================

# Arity 2 (bottom rung).
sfc_pipe2 = compose.pipe(sfc_si, sfc_if)
assert_type(sfc_pipe2, Pipeline[str, float])

# Arity 9 (top rung): str -> int -> float -> bool -> bytes -> str -> int ->
# float -> bool -> bytes ... nine stages ending at bytes.
sfc_pipe9 = compose.pipe(
    sfc_si, sfc_if, sfc_fb, sfc_by, sfc_bs, sfc_si, sfc_if, sfc_fb, sfc_by
)
assert_type(sfc_pipe9, Pipeline[str, bytes])

# pipe() output then `>>` keeps threading types (probe_chain.py:82).
sfc_pipe_then_rshift = compose.pipe(sfc_si, sfc_if, sfc_fb, sfc_by, sfc_bs) >> sfc_si
assert_type(sfc_pipe_then_rshift, Pipeline[str, int])


# --- `>>` chain PAST the ladder length (12 stages -- binary `>>` is unlimited)
# The documented continuation once a chain exceeds the arity-9 ladder.
sfc_long = (
    compose.pipe(sfc_si, sfc_if)
    >> sfc_fb
    >> sfc_by
    >> sfc_bs
    >> sfc_si
    >> sfc_if
    >> sfc_fb
    >> sfc_by
    >> sfc_bs
    >> sfc_si
    >> sfc_if
)
assert_type(sfc_long, Pipeline[str, float])


# --- Pipeline Run[R] surface: run / arun / stream / astream + .output --------


def _surface_pipeline_run() -> None:
    assert_type(sfc_pipe2.run("hi"), Run[float])
    assert_type(sfc_pipe2.run("hi").output, float)
    assert_type(sfc_pipe2.run("hi", budget=Budget(tokens=5)).output, float)
    assert_type(sfc_pipe2.stream("hi"), RunStream[float])
    assert_type(sfc_pipe2.stream("hi").run, Run[float])


async def _surface_pipeline_arun_astream() -> None:
    assert_type(await sfc_pipe2.arun("hi"), Run[float])
    assert_type((await sfc_pipe2.arun("hi")).output, float)
    assert_type(sfc_pipe2.astream("hi"), AsyncRunStream[float])
    assert_type((await sfc_pipe2.astream("hi").run()).output, float)


# ===========================================================================
# aggregate() -- typed branches solve `AggIn`; bare-lambda branches stay clean
# T3 adjudication (not in the prototype): both must be diagnostic-free.
# ===========================================================================


def sfc_branch_len(x: str) -> int:
    return len(x)


def sfc_branch_upper(x: str) -> str:
    return x.upper()


# Typed branches: common input `str` is solved through the `Stage[AggIn, Any]`
# arm -> `Aggregate[str]`. Diagnostic-free.
sfc_agg_typed = compose.aggregate(a=sfc_branch_len, b=sfc_branch_upper)
assert_type(sfc_agg_typed, Aggregate[str])

# The `timeout_per_branch` keyword doesn't disturb inference.
sfc_agg_timeout = compose.aggregate(timeout_per_branch=1.0, only=sfc_branch_len)
assert_type(sfc_agg_timeout, Aggregate[str])

# Bare-lambda branches: the `| Callable[[Any], Any]` escape-hatch arm gives each
# lambda parameter `Any` (so `x + 1` / `x * 2` are NOT `reportOperatorIssue`),
# and `AggIn` falls back to its `default=Any` -> `Aggregate[Any]`. Both branch
# bodies below MUST be diagnostic-free -- that is the whole point of the arm.
sfc_agg_lambda = compose.aggregate(inc=lambda x: x + 1, dbl=lambda x: x * 2)
assert_type(sfc_agg_lambda, Aggregate[Any])


# aggregate `>>` composes as a `Stage[In, dict[str, Any]]`; a bare `dict`-typed
# reducer keeps the real `>>` path clean (see test_typing_inline's note).
def sfc_reduce(counts: dict) -> int:
    return sum(counts.values())


sfc_agg_pipe = sfc_agg_typed >> sfc_reduce
assert_type(sfc_agg_pipe, Pipeline[str, int])


def _surface_aggregate_run() -> None:
    # Aggregate output is always `dict[str, Any]` (heterogeneous branches).
    assert_type(sfc_agg_typed.run("hi"), Run[dict[str, Any]])
    assert_type(sfc_agg_typed.run("hi").output, dict[str, Any])


# ===========================================================================
# map / amap -- all four overload arms: {sync,async} x {raise,collect}.
# T3 adjudication (not in the prototype): the async arm unwraps the awaitable
# so `B` is the AWAITED value, not the returned Coroutine.
# ===========================================================================


async def sfc_async_stage(x: str) -> str:
    return x.upper()


def _surface_map_overloads() -> None:
    # sync stage, on_error="raise" (default) -> list[B]
    assert_type(compose.map(sfc_si, ["a", "b"]), list[int])
    assert_type(compose.map(sfc_si, ["a", "b"], on_error="raise"), list[int])
    # sync stage, on_error="collect" -> list[MapResult[B]]
    assert_type(
        compose.map(sfc_si, ["a", "b"], max_workers=2, on_error="collect"),
        list[MapResult[int]],
    )
    # async stage, on_error="raise" -> list[B] (B is the awaited str)
    assert_type(compose.map(sfc_async_stage, ["a", "b"]), list[str])
    # async stage, on_error="collect" -> list[MapResult[B]]
    assert_type(
        compose.map(sfc_async_stage, ["a", "b"], on_error="collect"),
        list[MapResult[str]],
    )


async def _surface_amap_overloads() -> None:
    # amap carries the identical four-arm overload set.
    assert_type(await compose.amap(sfc_si, ["a", "b"]), list[int])
    assert_type(
        await compose.amap(sfc_si, ["a", "b"], on_error="collect"),
        list[MapResult[int]],
    )
    assert_type(await compose.amap(sfc_async_stage, ["a", "b"]), list[str])
    assert_type(
        await compose.amap(sfc_async_stage, ["a", "b"], on_error="collect"),
        list[MapResult[str]],
    )


# ===========================================================================
# @task / @flow / @tool -- both decorator forms; Task/Flow/Tool[P, R].
# ===========================================================================


@task
def sfc_task_step(text: str) -> Facts:
    return Facts(items=[text])


@task(retries=2)
def sfc_task_cfg(text: str) -> int:
    return len(text)


@flow
def sfc_flow_report(topic: str) -> Report:
    return Report(title=topic, body=topic * 2)


@flow(name="sfc_flow_cfg")
def sfc_flow_cfg(topic: str) -> Report:
    return Report(title=topic, body=topic)


@tool
def sfc_tool_search(query: str) -> str:
    """Search for something.

    Args:
        query: the search query.
    """
    return query.upper()


@tool(name="sfc_tool_lookup")
def sfc_tool_lookup(key: str) -> int:
    """Look up a value by key.

    Args:
        key: the lookup key.
    """
    return len(key)


def _surface_task_flow_tool_calls() -> None:
    # @task/@tool execute directly (no `Run` wrapper on `__call__`).
    assert_type(sfc_task_step("hi"), Facts)
    assert_type(sfc_task_step(text="hi"), Facts)
    assert_type(sfc_task_cfg("hi"), int)
    assert_type(sfc_tool_search("q"), str)
    assert_type(sfc_tool_search(query="q"), str)
    assert_type(sfc_tool_lookup("k"), int)
    # @flow `__call__ -> R` is sugar for `.run(x).output`.
    assert_type(sfc_flow_report("ai"), Report)
    assert_type(sfc_flow_report(topic="ai"), Report)
    assert_type(sfc_flow_cfg("ai"), Report)


async def _surface_task_arun() -> None:
    # `Task.arun -> R` directly (async twin of `__call__`, no `Run` wrapper).
    assert_type(await sfc_task_step.arun("hi"), Facts)


# --- @flow Run[R] surface: run / arun / stream + .output --------------------


def _surface_flow_run() -> None:
    assert_type(sfc_flow_report.run("ai"), Run[Report])
    assert_type(sfc_flow_report.run("ai").output, Report)
    assert_type(sfc_flow_report.run("ai", budget=Budget(tokens=10)).output, Report)
    assert_type(sfc_flow_report.stream("ai"), RunStream[Report])
    assert_type(sfc_flow_report.stream("ai").run, Run[Report])


async def _surface_flow_arun() -> None:
    assert_type(await sfc_flow_report.arun("ai"), Run[Report])
    assert_type((await sfc_flow_report.arun("ai")).output, Report)


def _surface_task_flow_tool_bare_annotations(t: Task, f: Flow, tl: Tool) -> None:
    # Bare Task/Flow/Tool mean `[..., Any]` (T1 PEP 696 default on P/R).
    assert_type(t("x"), Any)
    assert_type(f("x"), Any)
    assert_type(tl("x"), Any)


# ===========================================================================
# Flow pause surface: resume() / aresume() are explicitly `-> Run[Any]`
# (the output type is not statically known from a run_id alone).
# ===========================================================================


def _surface_resume() -> None:
    r = resume("some-run-id")
    assert_type(r, Run[Any])
    assert_type(r.output, Any)
    r2 = resume("some-run-id", {"approval:x": True}, budget=Budget(tokens=5))
    assert_type(r2, Run[Any])


async def _surface_aresume() -> None:
    r = await compose.aresume("some-run-id")
    assert_type(r, Run[Any])
    assert_type(r.output, Any)


# ===========================================================================
# Deterministic helpers usable in @flow bodies -- typed return values.
# ===========================================================================


def _surface_now_random() -> None:
    assert_type(compose.now(), datetime.datetime)
    assert_type(compose.random(), float)


async def _surface_anow_arandom() -> None:
    assert_type(await compose.anow(), datetime.datetime)
    assert_type(await compose.arandom(), float)


# ===========================================================================
# NEGATIVE surface -- each MUST be rejected; rule-scoped ignores are
# self-verifying under reportUnnecessaryTypeIgnoreComment.
# ===========================================================================


def _neg_pipeline_wrong_call_arg(p: Pipeline[str, int]) -> None:
    # `pipeline(42)` on a `Pipeline[str, int]`: int is not assignable to str.
    result = p(42)  # pyright: ignore[reportArgumentType]
    del result


def _neg_pipe_wrong_wiring() -> None:
    # sfc_si returns int; sfc_wants_bool expects bool (int is NOT assignable to
    # bool) -- the ladder rejects the second stage (probe_errors.py:30 class).
    bad = compose.pipe(sfc_si, sfc_wants_bool)  # pyright: ignore[reportArgumentType]
    del bad


def _neg_pipe_ten_stages_exceeds_ladder() -> None:
    # 10 well-wired stages build fine at runtime but exceed the arity-9 ladder
    # -- a static no-overload-match, BY DESIGN (no `Any` fallback; docs point
    # to `>>` for longer chains).
    bad = compose.pipe(  # pyright: ignore[reportCallIssue]
        sfc_si, sfc_if, sfc_fb, sfc_by, sfc_bs, sfc_si, sfc_if, sfc_fb, sfc_by, sfc_bs
    )
    del bad


def _neg_two_param_agent_rshift() -> None:
    # Variant A: a 2-positional-arg agent's `self` is not `AgentFunction[[X],
    # R2]`, so no `>>` overload applies -> the operator itself is unsupported.
    bad = sfc_two_param >> sfc_shout  # pyright: ignore[reportOperatorIssue]
    del bad


def _neg_zero_param_agent_rshift() -> None:
    # Same hard reject for a 0-param agent.
    bad = sfc_no_param >> sfc_shout  # pyright: ignore[reportOperatorIssue]
    del bad


def _neg_agent_wrong_call_arg() -> None:
    # `sfc_extract` expects `text: str`; passing 42 is a real static error.
    result = sfc_extract(42)  # pyright: ignore[reportArgumentType]
    del result


def _neg_tool_wrong_call_arg() -> None:
    # `sfc_tool_search` expects `query: str`; passing 42 is a real static error.
    result = sfc_tool_search(42)  # pyright: ignore[reportArgumentType]
    del result
