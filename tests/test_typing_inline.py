"""Static-typing pins for v0.5.0 Plan B, Task 1: ``Run[R]``/``MapResult[T]``/
``RunStream[R]``/``AsyncRunStream[R]`` generics.

This file is checked by the *same* ``pyright`` invocation the release gate
already runs (``[tool.pyright]`` in pyproject.toml includes ``tests``) --
no new CI step, no new dependency. Positive shape assertions use
``typing_extensions.assert_type`` (a runtime no-op: it just returns its
first argument unchanged, so these also execute harmlessly under pytest).
Negative (should-be-rejected) assignments are marked with rule-scoped
``# pyright: ignore[<rule>]`` comments -- never bare -- so a wrong rule
name or a diagnostic that stops firing both resurface as a new
``reportUnnecessaryTypeIgnoreComment`` error in this same gate (see
``plans/superpowers/research/2026-07-16-typing/typing-research.md``).

The PEP 696 default (``typing_extensions.TypeVar("R", default=Any)``) is
what makes a *bare* ``Run``/``MapResult`` annotation mean ``Run[Any]`` /
``MapResult[Any]`` under strict mode too -- copied verbatim from
``plans/superpowers/research/2026-07-16-typing/prototype/probe_run_default.py``,
which proved this shape gives zero new diagnostics in both basic and
strict pyright modes.

Runtime pins (dataclass shape / pickle / journal round-trip) prove the
``Generic[...]`` base classes are pure erasure -- zero behavioral change --
exactly as the research (interactive probes, not saved to a file) found.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any

import pytest
from typing_extensions import assert_type

from composeai._encoding import from_jsonable, to_jsonable
from composeai.agentfn import AgentFunction, agent
from composeai.combinators import Aggregate, MapResult, Pipeline
from composeai.combinators import aggregate as _aggregate
from composeai.combinators import amap as _amap
from composeai.combinators import map as _map
from composeai.combinators import pipe as _pipe
from composeai.errors import CompositionTypeError
from composeai.flow import Flow, Task
from composeai.flow import flow as _flow
from composeai.flow import resume as _resume
from composeai.flow import task as _task
from composeai.messages import Usage
from composeai.runs import AsyncRunStream, Budget, Run, RunStream
from composeai.testing import FakeModel
from composeai.tools import Tool
from composeai.tools import tool as _tool
from composeai.tracing import Trace


def _run(output: Any) -> Run:
    """Build a ``Run`` whose ``output`` argument is itself ``Any``-typed --
    composeai's own internal producers never know a caller's real output
    type, so this is the "bare construction" shape the 41 internal
    producer/consumer sites in agentfn.py/combinators.py/flow.py/runs.py
    are in."""
    return Run(
        id="r1",
        status="completed",
        output=output,
        usage=Usage(),
        trace=Trace(trace_id="t1"),
        messages=[],
    )


# --- positive: bare Run means Run[Any] (PEP 696 default) --------------------


def test_bare_run_output_is_any() -> None:
    # Calling `Run(...)` with an `Any`-typed `output` argument -- the shape
    # every internal producer site uses -- infers `Run[Any]`.
    run = _run(1)
    assert_type(run.output, Any)

    # A variable explicitly annotated with the *bare* (unsubscripted) `Run`
    # name -- the exact case that regresses to `reportMissingTypeArgument`
    # under strict mode without the `default=Any` TypeVar (see
    # probe_run_default.py). Bare `Run` must mean `Run[Any]`.
    bare_run: Run = Run(
        id="r2",
        status="completed",
        output=3.14,
        usage=Usage(),
        trace=Trace(trace_id="t2"),
        messages=[],
    )
    assert_type(bare_run.output, Any)


def test_run_str_annotated_variable_output_is_str() -> None:
    typed_run: Run[str] = Run(
        id="r3",
        status="completed",
        output="hi",
        usage=Usage(),
        trace=Trace(trace_id="t3"),
        messages=[],
    )
    assert_type(typed_run.output, str)

    # A `Run[str]` is accepted anywhere a bare `Run` is expected (bare means
    # `Run[Any]`, and every concrete `Run[X]` is assignable to `Run[Any]`).
    def handle(run: Run) -> None:
        assert_type(run.output, Any)

    handle(typed_run)


def test_run_wrong_output_type_is_rejected() -> None:
    # Negative case: `Run(output=42, ...)` infers `Run[int]`; assigning that
    # to a variable declared `Run[str]` is a real static error (R is
    # invariant) -- rule-scoped, not bare, so
    # `reportUnnecessaryTypeIgnoreComment` catches it if this ever stops
    # firing (e.g. a future refactor accidentally widens `output`'s type).
    bad_run: Run[str] = Run(  # pyright: ignore[reportAssignmentType]
        id="r4",
        status="completed",
        output=42,
        usage=Usage(),
        trace=Trace(trace_id="t4"),
        messages=[],
    )
    del bad_run


# --- positive: MapResult[T] ---------------------------------------------------


def test_map_result_int_value_is_int_or_none() -> None:
    ok: MapResult[int] = MapResult(ok=True, value=5)
    assert_type(ok.value, int | None)

    missing: MapResult[int] = MapResult(ok=False, error="boom", error_type="ValueError")
    assert_type(missing.value, int | None)


def test_map_result_wrong_value_type_is_rejected() -> None:
    bad: MapResult[int] = MapResult(  # pyright: ignore[reportAssignmentType]
        ok=True, value="not an int"
    )
    del bad


# --- positive: RunStream[R] / AsyncRunStream[R] -------------------------------
# Pure static-check helpers (never called) -- pyright still analyzes their
# bodies even though pytest never executes them, exactly like
# probe_run_default.py's `handle_d`.


def _check_run_stream_typing(rs: RunStream[str]) -> None:
    assert_type(rs.run, Run[str])


async def _check_async_run_stream_typing(ars: AsyncRunStream[int]) -> None:
    result = await ars.run()
    assert_type(result, Run[int])


def _check_bare_stream_typing(rs: RunStream, ars: AsyncRunStream) -> None:
    assert_type(rs.run, Run[Any])


# --- runtime pins: Generic is pure erasure ------------------------------------


def test_dataclass_fields_unchanged_by_generic() -> None:
    names = [f.name for f in dataclasses.fields(Run)]
    assert names == ["id", "status", "output", "usage", "trace", "messages", "pending"]

    mr_names = [f.name for f in dataclasses.fields(MapResult)]
    assert mr_names == ["ok", "value", "error", "error_type"]


def test_run_pickle_round_trip() -> None:
    run = _run({"facts": ["a", "b"]})
    restored = pickle.loads(pickle.dumps(run))
    assert restored == run
    assert type(restored) is Run


def test_run_journal_round_trip() -> None:
    run = _run({"facts": ["a", "b"]})
    restored = from_jsonable(to_jsonable(run))
    assert restored == run
    assert type(restored) is Run


def test_bare_isinstance_and_runtime_subscription_unaffected() -> None:
    run = _run(1)
    # Generic doesn't change runtime class identity: isinstance against the
    # bare (unsubscripted) class still works exactly as before.
    assert isinstance(run, Run)

    # Generic[R] supplies `__class_getitem__` for free -- `Run[str]` is now
    # subscriptable (it used to raise `TypeError: 'type' object is not
    # subscriptable`), and instantiating through the subscripted alias
    # still produces a genuine, bare `Run` instance at runtime.
    instance = Run[str](
        id="r5",
        status="completed",
        output="hi",
        usage=Usage(),
        trace=Trace(trace_id="t5"),
        messages=[],
    )
    assert type(instance) is Run


# --- v0.5.0 Plan B, Task 2: Stage protocol + Pipeline[In, Out]/Aggregate[In] --
#
# ``pipe()`` itself stays untyped this task (its overload ladder, which
# infers ``In``/``Out`` from the actual stage callables, is Task 3) -- these
# pins assert exactly what IS expressible now: a bare ``Pipeline``/
# ``Aggregate`` from ``pipe()``/direct construction means ``Pipeline[Any,
# Any]``/``Aggregate[Any]`` (PEP 696 ``default=Any``, same as ``Run``/
# ``RunStream`` in Task 1 above), while ``>>`` still genuinely solves
# ``NewOut``/``NewIn`` from the OTHER operand's own signature -- including a
# lambda, whose parameter type is inferred from the ``Stage`` protocol's
# positional-only ``__call__`` (the load-bearing ``/`` -- see
# ``composeai.combinators.Stage``'s docstring). Positive shapes mirror
# ``plans/superpowers/research/2026-07-16-typing/prototype/probe_chain.py``
# lines 38-49 (chained ``>>``) and 83-94 (``>>`` picking up after a
# fixed-arity chain; lambda context-sensitive inference), scoped down to
# what doesn't depend on ``pipe()``'s own ladder.


def _str_to_int(x: str) -> int:
    return len(x)


def _int_to_bool(x: int) -> bool:
    return x > 0


def _bool_to_str(x: bool) -> str:
    return str(x)


def _str_identity(x: str) -> str:
    return x


# --- positive: pipe()'s overload ladder now infers In/Out (Task 3) ----------
# These two were Task-2 pins asserting `pipe()` stayed untyped (bare
# `Pipeline[Any, Any]`); Task 3's arity-2..9 ladder now infers `In`/`Out` from
# the actual stage callables, so they assert the real inferred chain instead.


def test_pipe_two_stages_infers_pipeline() -> None:
    # `pipe(str->int, int->bool)` now infers `Pipeline[str, bool]` through the
    # arity-2 ladder overload (`s1: Stage[A, B], s2: Stage[B, C], / ->
    # Pipeline[A, C]`) -- prototype probe_chain.py:29 parity.
    p = _pipe(_str_to_int, _int_to_bool)
    assert_type(p, Pipeline[str, bool])


def test_pipe_then_rshift_still_infers_new_out() -> None:
    # The ladder infers `p`'s `In`/`Out` (`str`/`bool`), then `>>` genuinely
    # solves `NewOut` from `_bool_to_str`'s own signature -- the whole chain
    # types end-to-end now (`Pipeline[str, str]`).
    p = _pipe(_str_to_int, _int_to_bool)
    p2 = p >> _bool_to_str
    assert_type(p2, Pipeline[str, str])


# --- positive: direct construction pins In/Out, >> keeps solving NewOut/NewIn


def test_pipeline_direct_construction_pins_in_out() -> None:
    # `Pipeline.__init__` never references `In`/`Out` (unlike `Run.__init__`,
    # whose `output: R` field genuinely solves `R`) -- `stages: tuple[Stage,
    # ...]` doesn't pin anything, so only an explicit annotation does. This
    # type-checks because `Pipeline[Any, Any]`'s `Any` arguments are freely
    # assignable to any `Pipeline[X, Y]` (`Any` bypasses pyright's
    # generic-argument invariance) -- not because the constructor infers it.
    p: Pipeline[str, int] = Pipeline((_str_to_int,))
    assert_type(p, Pipeline[str, int])


def test_pipeline_rshift_after_direct_construction_infers_full_chain() -> None:
    p: Pipeline[str, int] = Pipeline((_str_to_int,))
    p2 = p >> _int_to_bool
    assert_type(p2, Pipeline[str, bool])


def test_pipeline_rshift_lambda_type_checks() -> None:
    # A bare, unannotated lambda as the RHS of `>>` still type-checks against
    # `Stage[Out, NewOut]` (matching probe_chain.py's `extract >> summarize >>
    # (lambda s: len(s))` shape) thanks to the positional-only `/` in
    # `Stage.__call__`. Verified independently (against both this class and
    # a byte-for-byte copy of prototype/proto.py) that pyright does NOT
    # actually thread `Out` into the lambda's own parameter as expected-type
    # context THROUGH the `>>` operator specifically -- `len`'s fixed `int`
    # return is what makes this resolve to `int` below, not genuine
    # parameter-type inference (swapping in a `str`- or `int`-only method
    # like `.upper()`/`.bit_length()` yields `NewOut = Unknown` instead, for
    # both this class and proto.py's own `Pipeline`, and for the exact
    # `extract >> summarize >> (lambda ...)` chain probe_chain.py uses -- a
    # real pyright limitation, not something this task's design regresses).
    # An explicit `p.__rshift__(lambda n: n + 1)` method call (not the `>>`
    # operator) DOES get genuine parameter-type inference, for contrast.
    p: Pipeline[str, int] = Pipeline((_str_to_int,))
    p_lambda = p >> (lambda n: len(str(n)))
    assert_type(p_lambda, Pipeline[str, int])


def test_pipeline_rrshift_infers_new_in() -> None:
    p: Pipeline[int, bool] = Pipeline((_int_to_bool,))
    # `_str_to_int` (a plain function) has no `__rshift__` of its own, so
    # `_str_to_int >> p` resolves through `Pipeline.__rrshift__`.
    p2 = _str_to_int >> p
    assert_type(p2, Pipeline[str, bool])


# --- positive: Aggregate[In] -------------------------------------------------


def test_aggregate_direct_construction_and_rshift() -> None:
    def branch_a(x: str) -> int:
        return len(x)

    def branch_b(x: str) -> int:
        return len(x) * 2

    agg: Aggregate[str] = Aggregate({"a": branch_a, "b": branch_b})
    assert_type(agg, Aggregate[str])

    # `total`'s parameter is bare `dict` (not `dict[str, Any]`): `Aggregate`'s
    # own runtime `.output_type` is the bare `dict` CLASS (never
    # parameterized -- branches can return unrelated types, see
    # `Aggregate.__init__`), and `pipe()`'s own composition-time check
    # (`_types_compatible`, unrelated to this task -- untouched) compares
    # runtime type objects by `==`, under which `dict != dict[str, Any]`.
    # `agg >> total` actually runs `pipe(agg, total)` for real (unlike the
    # `_check_*` helpers below, which stay uncalled specifically to dodge
    # engine execution) -- keeping `total`'s annotation as bare `dict`
    # exercises the real `>>` path without tripping that pre-existing,
    # out-of-scope runtime mismatch. Statically, bare `dict` still satisfies
    # `Stage[dict[str, Any], NewOut]`'s contravariant parameter (unparameterized
    # `dict` behaves as `dict[Unknown, Unknown]`, compatible with anything).
    def total(counts: dict) -> int:
        return sum(counts.values())

    p = agg >> total
    assert_type(p, Pipeline[str, int])


# --- negative: rule-scoped, never executed -----------------------------------
#
# Pure static-check helpers (never called by pytest, same pattern as
# `_check_run_stream_typing` above) -- pyright still analyzes their bodies.
# Actually *calling* `Pipeline.__call__`/`.run()` for real would execute the
# pipeline through the full engine (spans, a `runs` row, ...), disproportionate
# for what's purely a static-typing pin.


def _check_pipeline_call_wrong_argument_type(pipeline_str_int: Pipeline[str, int]) -> None:
    result = pipeline_str_int(42)  # pyright: ignore[reportArgumentType]
    del result


def _check_pipeline_rshift_incompatible_next_stage(p: Pipeline[str, int]) -> None:
    def wants_bool(b: bool) -> str:
        return str(b)

    # `p`'s Out (`int`) doesn't structurally satisfy `wants_bool`'s `bool`
    # parameter (int isn't assignable to bool) -- no overload of `>>`
    # matches, so pyright reports the operator itself as unsupported rather
    # than a plain argument-type mismatch.
    bad = p >> wants_bool  # pyright: ignore[reportOperatorIssue]
    del bad


def test_pipeline_assignment_mismatched_out_is_rejected() -> None:
    # Negative case: `Pipeline[str, str]` assigned to a `Pipeline[str, int]`-
    # declared variable is a real static error -- `str`/`int` are unrelated,
    # so this fails regardless of `Out`'s declared covariance.
    p_str_str: Pipeline[str, str] = Pipeline((_str_identity,))
    bad: Pipeline[str, int] = p_str_str  # pyright: ignore[reportAssignmentType]
    del bad


# --- v0.5.0 Plan B, Task 3: pipe() ladder + aggregate() + map/amap overloads --
#
# Prototype ground truth:
# plans/superpowers/research/2026-07-16-typing/prototype/{proto,probe_chain,
# probe_errors}.py, plus the scratch aggregate/map probes recorded in this
# task's report. The ladder is arities 2..9 (EIGHT overloads) with NO `Any`
# fallback: a 10-stage `pipe()` is a static no-overload-match by design (docs
# point to `>>` for longer chains). Every negative below is rule-scoped
# (never bare) so a wrong rule or a diagnostic that stops firing resurfaces as
# `reportUnnecessaryTypeIgnoreComment` in this same gate.


# 9-stage chain: str -> int -> float -> bool -> bytes -> str -> int -> float
# -> bool -> (str via g9). g10 pushes it to 10 stages (past the ladder).
def _g1(x: str) -> int:
    return len(x)


def _g2(x: int) -> float:
    return x * 1.0


def _g3(x: float) -> bool:
    return x > 0


def _g4(x: bool) -> bytes:
    return b"1" if x else b"0"


def _g5(x: bytes) -> str:
    return x.decode()


def _g6(x: str) -> int:
    return len(x)


def _g7(x: int) -> float:
    return x * 1.0


def _g8(x: float) -> bool:
    return x > 0


def _g9(x: bool) -> str:
    return str(x)


def _g10(x: str) -> int:
    return len(x)


def _wants_bool(b: bool) -> str:
    return str(b)


def _two_param(a: str, b: int) -> str:
    return a * b


# --- positive: the ladder infers In/Out for arities 2..9 --------------------


def test_pipe_nine_stages_infers() -> None:
    # 9 stages is the top rung of the ladder: `pipe(g1..g9)` infers
    # `Pipeline[str, str]` (A from the first stage's input, last from g9's
    # return). Builds for real (composition-time check only -- no stage runs).
    p = _pipe(_g1, _g2, _g3, _g4, _g5, _g6, _g7, _g8, _g9)
    assert_type(p, Pipeline[str, str])


def test_pipe_ten_stages_exceeds_ladder() -> None:
    # 10 well-wired stages BUILD fine at runtime (the `*stages` impl accepts
    # any count) but exceed the arity-9 ladder -- a static no-overload-match
    # (`reportCallIssue`), BY DESIGN (no `Any` fallback overload; docs point to
    # `>>` for longer chains). The ignore pins that exact rule.
    p = _pipe(_g1, _g2, _g3, _g4, _g5, _g6, _g7, _g8, _g9, _g10)  # pyright: ignore[reportCallIssue]
    assert isinstance(p, Pipeline)


# --- negative: wrong wiring / arity, rejected in-ladder ----------------------


def test_pipe_wrong_wiring_in_ladder_is_rejected() -> None:
    # `_str_to_int` returns `int`; `_wants_bool` expects `bool` (int is NOT
    # assignable to bool) -- the ladder rejects the second stage with
    # `reportArgumentType` (prototype probe_errors.py:30's diagnostic class).
    # Runtime agrees: `pipe()` raises `CompositionTypeError` for the same
    # int->bool mismatch before any stage runs.
    with pytest.raises(CompositionTypeError):
        _pipe(_str_to_int, _wants_bool)  # pyright: ignore[reportArgumentType]


def test_pipe_two_param_stage_is_rejected() -> None:
    # A 2-positional-param callable does NOT satisfy `Stage[A, B]`'s single
    # positional-only `__call__` -- the ladder rejects it with
    # `reportArgumentType` ("Extra parameter"). (A real `@agent` erases arity
    # via `__call__(*args: Any) -> Any`, so this is pinned with a plain
    # 2-param function, which is what actually reaches the ladder.) Runtime
    # agrees: `str` output vs `int` input raises `CompositionTypeError`.
    with pytest.raises(CompositionTypeError):
        _pipe(_two_param, _int_to_bool)  # pyright: ignore[reportArgumentType]


# --- positive/negative: aggregate() infers In from its branches --------------


def test_aggregate_same_input_infers() -> None:
    def a(x: str) -> int:
        return len(x)

    def b(x: str) -> str:
        return x.upper()

    # Same-input branches: `aggregate(**branches: Stage[AggIn, Any]) ->
    # Aggregate[AggIn]` solves `AggIn` from the branches (`str`).
    agg = _aggregate(a=a, b=b)
    assert_type(agg, Aggregate[str])

    # The `timeout_per_branch` keyword doesn't disturb inference, and a single
    # branch still pins `AggIn`.
    agg2 = _aggregate(timeout_per_branch=1.0, only=a)
    assert_type(agg2, Aggregate[str])


def test_aggregate_mismatched_input_is_accepted() -> None:
    def a(x: str) -> int:
        return len(x)

    def c(x: int) -> str:
        return str(x)

    # Mismatched-input branches are ACCEPTED statically (finding I1): the
    # `Stage[AggIn, Any] | Callable[[Any], Any]` signature's escape-hatch arm
    # -- required to keep bare-lambda branches clean -- also accepts a
    # mismatched typed branch (pyright can't distinguish the two), so `c`
    # routes through the `Callable[[Any], Any]` arm rather than being rejected.
    # `AggIn` is still solved from the first branch, so this infers
    # `Aggregate[str]`. Runtime does NOT raise here (unlike `pipe()`):
    # `Aggregate.input_type` just falls back to `Any` when branches disagree,
    # so there is no runtime cost to the lost static rejection.
    agg = _aggregate(a=a, c=c)
    assert_type(agg, Aggregate[str])
    assert isinstance(agg, Aggregate)


# --- map / amap: on_error overloads select the element type ------------------
# Static-only pins (uncalled helpers, same convention as
# `_check_run_stream_typing` above) -- calling `map()`/`amap()` would drive the
# full engine, disproportionate for a static-typing pin. pyright still analyzes
# these bodies.


async def _async_str_to_str(x: str) -> str:
    return x.upper()


def _check_map_element_types() -> None:
    # `on_error="raise"` (the default overload) -> `list[B]`.
    raised = _map(_str_to_int, ["a", "b"])
    assert_type(raised, list[int])

    raised_explicit = _map(_str_to_int, ["a", "b"], on_error="raise")
    assert_type(raised_explicit, list[int])

    # `on_error="collect"` (no default -- must be passed) -> `list[MapResult[B]]`.
    collected = _map(_str_to_int, ["a", "b"], max_workers=2, on_error="collect")
    assert_type(collected, list[MapResult[int]])


def _check_map_async_stage_element_types() -> None:
    # An `async def` stage matches the async-stage overload pair (finding I2),
    # which unwraps the coroutine so `B` is the AWAITED value map() collects --
    # `list[str]`, not `list[Coroutine[..., str]]`.
    raised = _map(_async_str_to_str, ["a", "b"])
    assert_type(raised, list[str])

    collected = _map(_async_str_to_str, ["a", "b"], on_error="collect")
    assert_type(collected, list[MapResult[str]])


async def _check_amap_element_types() -> None:
    raised = await _amap(_str_to_int, ["a", "b"])
    assert_type(raised, list[int])

    collected = await _amap(_str_to_int, ["a", "b"], on_error="collect")
    assert_type(collected, list[MapResult[int]])


async def _check_amap_async_stage_element_types() -> None:
    # amap's async-stage overload twin (finding I2): same coroutine-unwrapping.
    raised = await _amap(_async_str_to_str, ["a", "b"])
    assert_type(raised, list[str])

    collected = await _amap(_async_str_to_str, ["a", "b"], on_error="collect")
    assert_type(collected, list[MapResult[str]])


# --- v0.5.0 Plan B, Task 4: AgentFunction[P, R] + @agent dual-form + Variant-A >>
#
# Prototype ground truth:
# plans/superpowers/research/2026-07-16-typing/prototype/{proto,probe_chain,
# probe_lambda,probe_variants}.py. `@agent` is now a two-overload dual form:
# bare `@agent` -> `AgentFunction[P, R]` (P the decorated function's whole
# parameter list, names included; R its structured-output type) and
# `@agent(model=..., ...)` -> a decorator returning the same. `P`/`R` carry PEP
# 696 defaults (probed: pyright 1.1.411 accepts `ParamSpec("P", default=...)` on
# pythonVersion 3.10), so a bare `AgentFunction` annotation means
# `AgentFunction[..., Any]`. `>>` is Variant A: only a SINGLE-positional-arg
# agent composes (`self: AgentFunction[[X], R2]`); a 2-param or 0-param agent is
# a hard `reportOperatorIssue`. `.run(...)`/`.arun(...)` return `Run[R]`,
# `.stream(...)` a `RunStream[R]`, `.astream(...)` an `AsyncRunStream[R]` --
# their PARAMETER typing stays loose (`*args: Any`) because pyright rejects a
# keyword-only `budget` between `*args: P.args` and `**kwargs: P.kwargs`
# (`reportGeneralTypeIssues`); the RETURN type is the deliverable. Every
# negative below is rule-scoped so a wrong rule or a diagnostic that stops
# firing resurfaces as `reportUnnecessaryTypeIgnoreComment` in this same gate.


@dataclasses.dataclass
class Facts:
    items: list[str]


# Bare `@agent` (model resolved lazily -- never touched here, only constructed).
@agent
def ty_extract(text: str) -> Facts:
    return Facts(items=[text])


# Parenthesized `@agent(...)`: same inferred `AgentFunction[(text: str), Facts]`.
@agent(model=FakeModel(["unused"]), retries=2)
def ty_extract_cfg(text: str) -> Facts:
    return Facts(items=[text])


@agent(model=FakeModel(["unused"]))
def ty_summarize(facts: Facts) -> str:
    return ",".join(facts.items)


@agent
def ty_two_param(a: str, b: int) -> str:
    return a * b


@agent
def ty_no_param() -> str:
    return "x"


def ty_shout(s: str) -> str:
    return s.upper()


def ty_make_text(n: int) -> str:
    return str(n)


# --- positive: >> chains (executed -- building a Pipeline is composition-time
# only, no engine/API; assert_type is a runtime no-op). ------------------------


def test_agent_bare_form_rshift_infers_pipeline() -> None:
    # `extract >> summarize` composes a bare-decorated single-arg agent
    # (`AgentFunction[(text: str), Facts]`) with another agent via Variant-A
    # `__rshift__` -- `X`=str, `R2`=Facts, `NewOut`=str -> `Pipeline[str, str]`
    # (probe_chain.py:38 parity).
    p = ty_extract >> ty_summarize
    assert_type(p, Pipeline[str, str])


def test_agent_with_args_form_rshift_infers_pipeline() -> None:
    # The parenthesized form infers the identical shape, so it composes the same.
    p = ty_extract_cfg >> ty_summarize
    assert_type(p, Pipeline[str, str])


def test_agent_three_stage_rshift_chain_infers() -> None:
    # `agent >> agent >> named_fn`: the first `>>` yields `Pipeline[str, str]`
    # (AgentFunction.__rshift__), the second is `Pipeline.__rshift__` picking up
    # the plain function `ty_shout` -- the whole chain types end-to-end
    # (probe_chain.py:49).
    p3 = ty_extract >> ty_summarize >> ty_shout
    assert_type(p3, Pipeline[str, str])


def test_agent_rrshift_from_plain_function_infers() -> None:
    # `plain_fn >> agent`: `ty_make_text` has no `__rshift__`, so this resolves
    # through `AgentFunction.__rrshift__` -- `X`=str, `R2`=Facts, `NewIn`=int ->
    # `Pipeline[int, Facts]`.
    pr = ty_make_text >> ty_extract
    assert_type(pr, Pipeline[int, Facts])


# --- static-only pins (uncalled helpers -- pyright analyzes their bodies, but
# pytest never runs them, so calling `ty_extract(...)`/`.run(...)`/`.stream(...)`
# here never drives the agent engine; same convention as `_check_*` above). ----


def _check_agent_bare_keyword_and_positional_call_typed() -> None:
    # ParamSpec preserves the parameter name `text`, so a keyword call
    # type-checks (probe_lambda.py:59) and both call forms return `Facts` --
    # pinning bare `@agent` infers `AgentFunction[(text: str), Facts]`.
    assert_type(ty_extract(text="hi"), Facts)
    assert_type(ty_extract("hi"), Facts)


def _check_agent_with_args_keyword_and_positional_call_typed() -> None:
    # The parenthesized form pins the same `AgentFunction[(text: str), Facts]`.
    assert_type(ty_extract_cfg(text="hi"), Facts)
    assert_type(ty_extract_cfg("hi"), Facts)


def _check_agent_call_wrong_argument_type_rejected() -> None:
    # `ty_extract` expects `text: str`; passing `42` is a real static error
    # (probe_lambda.py's keyword-typed call, inverted). Rule-scoped.
    result = ty_extract(42)  # pyright: ignore[reportArgumentType]
    del result


def _check_agent_run_returns_typed_run() -> None:
    # `.run(...)` returns `Run[R]` (`R`=Facts), so `.output` is `Facts` -- with
    # or without the keyword-only `budget` (whose loose parameter typing doesn't
    # disturb the `Run[R]` return, the actual deliverable).
    assert_type(ty_extract.run("hi").output, Facts)
    assert_type(ty_extract.run("hi", budget=Budget(tokens=10)).output, Facts)


def _check_agent_stream_typed() -> None:
    # `.stream(...)` returns `RunStream[R]`, whose `.run` is `Run[R]`.
    assert_type(ty_extract.stream("hi").run, Run[Facts])


async def _check_agent_arun_and_astream_typed() -> None:
    # `.arun(...)` -> `Run[R]`; `.astream(...)` -> `AsyncRunStream[R]`, whose
    # awaited `.run()` is `Run[R]`.
    assert_type((await ty_extract.arun("hi")).output, Facts)
    assert_type((await ty_extract.astream("hi").run()).output, Facts)


def _check_two_param_agent_rshift_rejected() -> None:
    # Variant A: a 2-positional-arg agent's `self` is `AgentFunction[(a: str,
    # b: int), str]`, which does NOT match `self: AgentFunction[[X], R2]` -- no
    # `__rshift__` overload applies, so pyright reports the operator itself as
    # unsupported (probe_variants.py's 2-param hard reject). Rule-scoped.
    bad = ty_two_param >> ty_shout  # pyright: ignore[reportOperatorIssue]
    del bad


def _check_zero_param_agent_rshift_rejected() -> None:
    # Same hard reject for a 0-param agent (`self` is `AgentFunction[(), str]`).
    bad = ty_no_param >> ty_shout  # pyright: ignore[reportOperatorIssue]
    del bad


def _check_bare_agent_function_annotation_is_any() -> None:
    # A bare (unsubscripted) `AgentFunction` annotation means
    # `AgentFunction[..., Any]` (PEP 696 defaults on `P`/`R`) -- the case that
    # would trip `reportMissingTypeArgument` under strict mode without them.
    def handle(af: AgentFunction) -> None: ...

    handle(ty_extract)


# --- v0.5.0 Plan B, Task 5: Task[P, R] / Flow[P, R] / Tool[P, R] + dual-form ---
#
# Same shapes as Task 4's `@agent` (ParamSpec `P` + `TypeVar` `R`, both with PEP
# 696 defaults so a bare `Task`/`Flow`/`Tool` annotation means `[..., Any]`; a
# two-overload dual form: bare `@task`/`@flow`/`@tool` -> the typed object, or
# parenthesized with the decorator's own real kwargs -> a decorator returning
# it). `Task.__call__ -> R` (a `@task` executes directly -- it has no `.run()`);
# `Flow.__call__ -> R` with `Flow.run/arun -> Run[R]` and `.stream -> RunStream[R]`;
# `Tool.__call__` is ParamSpec-typed (its model-facing `.execute`/`.aexecute`
# stay loose). `resume()` is explicitly `-> Run[Any]`. NO `Flow.__rshift__` was
# added: a `Flow` is not a first-class pipe stage today (not in
# `combinators._stage_input_type`'s isinstance set, no `.input_type`/
# `.output_type`, no existing `__rshift__`), so adding one would be new runtime
# behavior -- YAGNI, zero-runtime-change rule. Every negative is rule-scoped so
# a wrong rule or a diagnostic that stops firing resurfaces as
# `reportUnnecessaryTypeIgnoreComment` in this same gate.


@dataclasses.dataclass
class Report:
    title: str
    body: str


# Bare `@task` -> `Task[(text: str), Facts]`.
@_task
def ty_task_step(text: str) -> Facts:
    return Facts(items=[text])


# Parenthesized `@task(retries=...)` -> the same inferred `Task[(text: str), int]`.
@_task(retries=2)
def ty_task_cfg(text: str) -> int:
    return len(text)


# Bare `@flow` -> `Flow[(topic: str), Report]`.
@_flow
def ty_flow_report(topic: str) -> Report:
    return Report(title=topic, body=topic * 2)


# Parenthesized `@flow(name=...)` -> the same inferred `Flow[(topic: str), Report]`.
@_flow(name="ty_flow_cfg")
def ty_flow_cfg(topic: str) -> Report:
    return Report(title=topic, body=topic)


# Bare `@tool` -> `Tool[(query: str), str]` (a docstring is required by @tool).
@_tool
def ty_tool_search(query: str) -> str:
    """Search for something.

    Args:
        query: the search query.
    """
    return query.upper()


# Parenthesized `@tool(name=...)` -> the same inferred `Tool[(key: str), int]`.
@_tool(name="ty_tool_lookup")
def ty_tool_lookup(key: str) -> int:
    """Look up a value by key.

    Args:
        key: the lookup key.
    """
    return len(key)


# --- positive: @tool calls are ParamSpec-typed (executed -- `Tool.__call__` is
# a pure passthrough to the wrapped fn, no engine/spans; assert_type is a
# runtime no-op). --------------------------------------------------------------


def test_tool_bare_and_with_args_call_typed() -> None:
    # ParamSpec preserves the parameter name `query`, so a keyword call
    # type-checks and both call forms return the fn's own return type (`str`).
    assert_type(ty_tool_search("x"), str)
    assert_type(ty_tool_search(query="x"), str)
    # The parenthesized `@tool(name=...)` form infers the same shape.
    assert_type(ty_tool_lookup("k"), int)
    assert_type(ty_tool_lookup(key="k"), int)


def test_task_flow_tool_bare_annotations_are_any() -> None:
    # A bare (unsubscripted) `Task`/`Flow`/`Tool` annotation means `[..., Any]`
    # (PEP 696 defaults on `P`/`R`) -- the case that would trip
    # `reportMissingTypeArgument` under strict mode without them.
    def h_task(t: Task) -> None: ...
    def h_flow(f: Flow) -> None: ...
    def h_tool(t: Tool) -> None: ...

    h_task(ty_task_step)
    h_flow(ty_flow_report)
    h_tool(ty_tool_search)


def test_task_flow_tool_generic_runtime_erasure() -> None:
    # `Generic[P, R]` is erasure-only: runtime class identity is unchanged, so
    # isinstance against the bare class still works exactly as before.
    assert isinstance(ty_task_step, Task)
    assert isinstance(ty_flow_report, Flow)
    assert isinstance(ty_tool_search, Tool)

    # Generic supplies `__class_getitem__` for free -- the subscripted alias is
    # now usable (it used to raise `TypeError: 'type' object is not
    # subscriptable`). The first type argument is a ParamSpec, so it takes a
    # parameter LIST (`[str]`), not a bare type.
    assert Task[[str], Facts] is not None
    assert Flow[[str], Report] is not None
    assert Tool[[str], str] is not None


# --- static-only pins (uncalled helpers -- pyright analyzes their bodies, but
# pytest never runs them, so a `@task`/`@flow` body or the durable engine is
# never actually driven here; same convention as the Task-4 `_check_*` above). -


def _check_task_call_typed() -> None:
    # `Task.__call__ -> R` (a `@task` executes directly -- no `.run()`): both a
    # positional and a keyword call return the fn's own return type.
    assert_type(ty_task_step("hi"), Facts)
    assert_type(ty_task_step(text="hi"), Facts)
    # The parenthesized `@task(retries=...)` form pins the same `(text: str) -> int`.
    assert_type(ty_task_cfg("hi"), int)


async def _check_task_arun_typed() -> None:
    # `Task.arun(...)` is the async twin of `__call__` -- it returns `R` directly
    # (a `@task` has no `Run` wrapper), so `await`ing it yields `Facts`.
    assert_type(await ty_task_step.arun("hi"), Facts)


def _check_task_call_wrong_argument_type_rejected() -> None:
    # `ty_task_step` expects `text: str`; passing `42` is a real static error.
    result = ty_task_step(42)  # pyright: ignore[reportArgumentType]
    del result


def _check_flow_call_and_run_typed() -> None:
    # `Flow.__call__ -> R` (sugar for `.run(x).output`); `.run(...)` returns
    # `Run[R]` (`R`=Report), so `.output` is `Report` -- with or without the
    # keyword-only `budget` (whose loose parameter typing doesn't disturb the
    # `Run[R]` return, the actual deliverable). `.stream(...)` returns
    # `RunStream[R]`, whose `.run` is `Run[R]`.
    assert_type(ty_flow_report("ai"), Report)
    assert_type(ty_flow_report(topic="ai"), Report)
    assert_type(ty_flow_report.run("ai").output, Report)
    assert_type(ty_flow_report.run("ai", budget=Budget(tokens=10)).output, Report)
    assert_type(ty_flow_report.stream("ai").run, Run[Report])
    # The parenthesized `@flow(name=...)` form pins the same shape.
    assert_type(ty_flow_cfg("ai").title, str)


async def _check_flow_arun_typed() -> None:
    # `Flow.arun(...)` -> `Run[R]`, so its `.output` is `Report`.
    assert_type((await ty_flow_report.arun("ai")).output, Report)


def _check_flow_call_wrong_argument_type_rejected() -> None:
    # `ty_flow_report` expects `topic: str`; passing `123` is a real static error.
    result = ty_flow_report(123)  # pyright: ignore[reportArgumentType]
    del result


def _check_tool_call_wrong_argument_type_rejected() -> None:
    # `ty_tool_search` expects `query: str`; passing `42` is a real static error.
    result = ty_tool_search(42)  # pyright: ignore[reportArgumentType]
    del result


def _check_resume_returns_run_any() -> None:
    # `resume(run_id)` is explicitly `-> Run[Any]` (it can revive any kind of
    # durable run -- a `@flow` or a standalone `@agent` -- whose output type is
    # not statically known at the resume site), so `.output` is `Any`.
    r = _resume("some-run-id")
    assert_type(r, Run[Any])
    assert_type(r.output, Any)
