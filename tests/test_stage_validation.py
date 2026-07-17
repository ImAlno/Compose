"""Tests for runtime stage-boundary validation (v0.5.0 Plan C, Task 1).

``combinators._ainvoke_stage`` is the single dispatch chokepoint every
``Pipeline`` stage, ``Aggregate`` branch, and ``map()`` item funnels
through. On entry it validates the incoming value against the stage's
declared input type with pydantic's strict mode, coerces it, and passes the
coerced value onward. A mismatch is a :class:`~composeai.errors.StageTypeError`
(the runtime twin of the build-time ``CompositionTypeError``). An ``Any``/
missing annotation is a hard no-op -- pydantic is never touched and object
identity is preserved.

Also covers the ``_types_compatible`` bare-origin gap fix (Plan B binding
I1): a bare ``dict`` output must compose with a ``dict[str, Any]``-annotated
reducer, both at build time and at runtime.

Deliberately NOT using ``from __future__ import annotations`` (same reason
as ``test_combinators.py``): stage annotations must resolve to real class
objects, not inert forward-reference strings.
"""

import asyncio
import dataclasses
from typing import Any, Protocol, runtime_checkable

import pytest
from pydantic import BaseModel

import composeai as compose
from composeai.combinators import _types_compatible as types_compatible
from composeai.combinators import aggregate, pipe
from composeai.errors import ComposeError, ConfigError, StageTypeError


class Topic(BaseModel):
    name: str


class Other(BaseModel):
    value: int


# --- StageTypeError shape ------------------------------------------------------


def test_stage_type_error_is_compose_error_and_type_error():
    assert issubclass(StageTypeError, ComposeError)
    assert issubclass(StageTypeError, TypeError)
    with pytest.raises(TypeError):
        raise StageTypeError("boom")
    with pytest.raises(ComposeError):
        raise StageTypeError("boom")


# --- _types_compatible: bare-origin <-> parameterized twin (I1 gap fix) --------


def test_types_compatible_bare_dict_feeds_parameterized_dict():
    assert types_compatible(dict, dict[str, Any]) is True


def test_types_compatible_parameterized_dict_feeds_bare_dict():
    assert types_compatible(dict[str, Any], dict) is True


def test_types_compatible_bare_list_feeds_parameterized_list_both_ways():
    assert types_compatible(list, list[int]) is True
    assert types_compatible(list[int], list) is True


def test_types_compatible_two_distinct_parameterizations_still_reject():
    # The fix is MINIMAL: it only relaxes the bare-vs-parameterized case.
    # Two fully-parameterized generics still compare by == only.
    assert types_compatible(list[str], list[int]) is False
    assert types_compatible(dict[str, int], dict[str, str]) is False


def test_types_compatible_bare_origin_of_unrelated_generic_rejects():
    assert types_compatible(str, list[int]) is False
    assert types_compatible(dict, list[int]) is False


# --- pipeline input boundary ---------------------------------------------------


def test_pipeline_input_mismatch_raises_stage_type_error_naming_boundary_and_type():
    def head(t: Topic) -> Topic:
        return t

    def tail(t: Topic) -> Topic:
        return t

    p = pipe(head, tail)
    with pytest.raises(StageTypeError) as exc_info:
        # Intentional wrong-type call: a Topic-input pipeline, an int arg. The
        # static rejection (pyright) is the build-time twin of the runtime
        # StageTypeError this test pins; rule-scoped ignore keeps both honest.
        p(42)  # pyright: ignore[reportArgumentType]
    message = str(exc_info.value)
    assert "pipeline input" in message
    assert "head" in message
    assert "Topic" in message


def test_pipeline_input_correct_instance_passes_through_identically():
    captured: dict[str, Any] = {}

    def head(t: Topic) -> Topic:
        captured["seen"] = t
        return t

    def tail(t: Topic) -> Topic:
        return t

    original = Topic(name="ai")
    out = pipe(head, tail)(original)
    # Strict validation preserves identity for already-correct instances -- the
    # exact same object reaches the stage body and comes back out.
    assert captured["seen"] is original
    assert out is original


def test_dict_is_coerced_to_model_at_stage_entry():
    captured: dict[str, Any] = {}

    def head(t: Topic) -> Topic:
        captured["seen"] = t
        return t

    def tail(t: Topic) -> Topic:
        return t

    # A dict for a Topic-typed pipeline: statically a type-lie (pyright rejects
    # it), but the runtime coerces dict->Topic -- the documented plain-callable-
    # returns-a-dict idiom this feature keeps ergonomic.
    out = pipe(head, tail)({"name": "quantum"})  # pyright: ignore[reportArgumentType]
    # dict -> model instantiation is allowed even in strict mode; the coerced
    # Topic instance -- not the raw dict -- is what the stage body receives.
    assert isinstance(captured["seen"], Topic)
    assert captured["seen"].name == "quantum"
    assert isinstance(out, Topic)


# --- stage-handoff boundary ----------------------------------------------------


def test_stage_handoff_lossy_scalar_coercion_is_rejected():
    def head(x: Any) -> Any:
        return "3"  # a str flowing toward an int-typed stage

    def tail(n: int) -> int:
        return n

    with pytest.raises(StageTypeError) as exc_info:
        pipe(head, tail)("ignored")
    message = str(exc_info.value)
    assert "stage handoff" in message
    assert "tail" in message
    assert "int" in message


def test_int_widens_to_float_at_stage_entry():
    captured: dict[str, Any] = {}

    def head(x: Any) -> Any:
        return 3

    def tail(f: float) -> float:
        captured["seen"] = f
        return f

    out = pipe(head, tail)("ignored")
    # int -> float is a safe widening pydantic permits even in strict mode.
    assert out == 3.0
    assert isinstance(captured["seen"], float)


# --- Any / missing annotation is a hard no-op ----------------------------------


def test_any_annotated_stage_sees_the_raw_value_untouched():
    captured: dict[str, Any] = {}

    def head(x: Any) -> Any:
        captured["seen"] = x
        return x

    def tail(x: Any) -> Any:
        return x

    sentinel = object()
    pipe(head, tail)(sentinel)
    # Any short-circuits before pydantic is ever constructed -- the exact
    # object flows through, identity intact.
    assert captured["seen"] is sentinel


def test_unannotated_stage_is_a_no_op():
    captured: dict[str, Any] = {}

    def head(x):
        captured["seen"] = x
        return x

    def tail(x):
        return x

    sentinel = object()
    pipe(head, tail)(sentinel)
    assert captured["seen"] is sentinel


# --- aggregate branch boundary -------------------------------------------------


def test_aggregate_branch_mismatch_raises_naming_the_branch():
    def as_str(x: str) -> str:
        return x

    def as_int(n: int) -> int:
        return n

    agg = aggregate(good=as_str, bad=as_int)
    with pytest.raises(StageTypeError) as exc_info:
        agg("hello")
    message = str(exc_info.value)
    assert "aggregate branch 'bad'" in message
    assert "int" in message


# --- map item boundary + on_error="collect" ------------------------------------


def test_map_collect_catches_stage_type_error_as_per_item_failure():
    def to_int(n: int) -> int:
        return n

    # "two" is a deliberate wrong-typed item (an int stage); pyright rejects the
    # mixed list, which is the static twin of the per-item StageTypeError below.
    results = compose.map(to_int, [1, "two", 3], on_error="collect")  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert results[0].ok is True
    assert results[0].value == 1
    assert results[1].ok is False
    assert results[1].error_type == "StageTypeError"
    assert results[2].ok is True
    assert results[2].value == 3


def test_map_item_boundary_named_in_raise_mode():
    def to_int(n: int) -> int:
        return n

    with pytest.raises(StageTypeError) as exc_info:
        # "bad" is a deliberate wrong-typed item (see the collect-mode twin).
        compose.map(to_int, [1, "bad", 3])  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert "map item 1" in str(exc_info.value)


# --- build-time eager warm -> ConfigError --------------------------------------


class _Arbitrary:
    """A plain, non-pydantic class."""


@dataclasses.dataclass
class ArbitraryDataclass:
    # A stdlib dataclass whose field is an arbitrary (non-pydantic) type:
    # `TypeAdapter(ArbitraryDataclass)` raises PydanticSchemaGenerationError,
    # and the arbitrary_types_allowed fallback is itself illegal (config can't
    # be passed to a dataclass), so no runtime validator can ever be built.
    thing: _Arbitrary


def test_unbuildable_stage_input_type_raises_config_error_at_build_time():
    def head(d: ArbitraryDataclass) -> str:
        return "x"

    def tail(x: str) -> str:
        return x

    with pytest.raises(ConfigError):
        pipe(head, tail)


def test_unbuildable_aggregate_branch_input_raises_config_error_at_build_time():
    def branch(d: ArbitraryDataclass) -> str:
        return "x"

    with pytest.raises(ConfigError):
        aggregate(only=branch)


# --- the flagship I1 gap: agg >> dict[str, Any] reducer ------------------------


def test_aggregate_composed_with_parameterized_dict_reducer_composes_and_runs():
    def sec(x: str) -> str:
        return f"sec:{x}"

    def perf(x: str) -> str:
        return f"perf:{x}"

    agg = aggregate(a=sec, b=perf)

    def reduce_it(d: dict[str, Any]) -> int:
        return len(d)

    # Composition must NOT raise (the I1 trap): Aggregate.output_type is bare
    # `dict`, the reducer input is `dict[str, Any]`.
    pipeline = agg >> reduce_it
    # And it must actually run end to end.
    assert pipeline("hi") == 2


# --- I-2: Protocol input annotations -------------------------------------------
#
# A non-`runtime_checkable` Protocol cannot be an `isinstance` target, so
# `TypeAdapter(Proto, arbitrary_types_allowed=True)` (the fallback for any
# non-pydantic class) raises a raw `pydantic_core.SchemaError` -- a sibling of
# `PydanticSchemaGenerationError`/`PydanticUserError` that used to escape the
# eager-warm net raw. It must now surface as a build-time `ConfigError` whose
# message points at `@runtime_checkable`. With the decorator, the fallback
# builds an isinstance validator and everything works.


class Quacks(Protocol):
    def quack(self) -> str: ...


@runtime_checkable
class RuntimeQuacks(Protocol):
    def quack(self) -> str: ...


class Duck:
    def quack(self) -> str:
        return "quack"


def test_non_runtime_checkable_protocol_input_raises_config_error_mentioning_runtime_checkable():
    def wants(d: Quacks) -> str:
        return d.quack()

    def tail(x: str) -> str:
        return x

    with pytest.raises(ConfigError) as exc_info:
        pipe(wants, tail)
    message = str(exc_info.value)
    # ConfigError, not a raw pydantic_core.SchemaError, and it names the fix.
    assert "runtime_checkable" in message
    assert "wants" in message  # the offending stage
    assert "Quacks" in message  # the offending type


def test_non_runtime_checkable_protocol_aggregate_branch_raises_config_error():
    def branch(d: Quacks) -> str:
        return d.quack()

    with pytest.raises(ConfigError) as exc_info:
        aggregate(only=branch)
    assert "runtime_checkable" in str(exc_info.value)


def test_runtime_checkable_protocol_builds_duck_passes_and_wrong_value_raises():
    def wants(d: RuntimeQuacks) -> str:
        return d.quack()

    def tail(x: str) -> str:
        return x

    # Builds cleanly: the arbitrary_types fallback yields an isinstance
    # validator (isinstance works against a @runtime_checkable Protocol).
    pipeline = pipe(wants, tail)
    # A structurally-conforming duck validates and reaches the body.
    assert pipeline(Duck()) == "quack"
    # A non-conforming value is a runtime StageTypeError at the boundary.
    with pytest.raises(StageTypeError):
        pipeline(42)  # pyright: ignore[reportArgumentType]


# --- I-3: map()/amap() warm the input adapter at entry -------------------------
#
# map()/amap() have no construction step (unlike pipe()/aggregate()), so an
# unbuildable `fn` input annotation (e.g. a non-`runtime_checkable` Protocol)
# used to surface as a raw pydantic_core.SchemaError at dispatch -- and, worst
# of all, `on_error="collect"` masked it as a per-ITEM failure on EVERY item
# with no pointer to the annotation. Warming the adapter at map()/amap() entry
# makes it fail ONCE, up front, with the same ConfigError + `@runtime_checkable`
# hint pipe()/aggregate() give -- BEFORE any item dispatches, so collect mode
# can never mask it.


def test_map_non_runtime_checkable_protocol_fn_raises_config_error_at_call():
    def wants(d: Quacks) -> str:
        return d.quack()

    # `on_error="collect"` would otherwise mask the config error as a per-item
    # failure on EVERY item -- the warm makes it a single ConfigError up front,
    # before any item dispatches, naming the fix.
    with pytest.raises(ConfigError) as exc_info:
        # Duck structurally conforms to Quacks, so this is a clean static call;
        # the ConfigError is a *build*-shape failure of the annotation itself.
        compose.map(wants, [Duck(), Duck()], on_error="collect")
    message = str(exc_info.value)
    assert "runtime_checkable" in message
    assert "wants" in message  # the offending stage
    assert "Quacks" in message  # the offending type


def test_amap_non_runtime_checkable_protocol_fn_raises_config_error_at_call():
    def wants(d: Quacks) -> str:
        return d.quack()

    async def drive():
        return await compose.amap(wants, [Duck(), Duck()], on_error="collect")

    with pytest.raises(ConfigError) as exc_info:
        asyncio.run(drive())
    assert "runtime_checkable" in str(exc_info.value)


def test_map_runtime_checkable_protocol_fn_works_and_wrong_item_raises_map_item():
    def wants(d: RuntimeQuacks) -> str:
        return d.quack()

    # A @runtime_checkable Protocol fn still warms and runs cleanly through map.
    results = compose.map(wants, [Duck(), Duck()])
    assert results == ["quack", "quack"]

    # A wrong item is a runtime StageTypeError at the boundary, named "map item".
    with pytest.raises(StageTypeError) as exc_info:
        # 42 is a deliberate non-conforming item -- pyright rejects the mixed
        # list, the static twin of the per-item StageTypeError below.
        compose.map(wants, [Duck(), 42])  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert "map item 1" in str(exc_info.value)
