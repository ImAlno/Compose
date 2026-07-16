"""Tests for ``composeai._schema.register_annotation_types`` (Phase 7).

``@task``/``@flow``/``@agent`` decoration must register every pydantic
model / dataclass / enum reachable from a function's annotations (params +
return, walked recursively through ``typing.get_args``) so a *fresh
process* -- one that never encoded the value itself, e.g. a resumed flow
started via ``python other_script.py`` -- can still decode journaled data
referencing those types without composeai ever importing anything
dynamically (the whole point of the registry: no pickle, no dynamic
import).
"""

import dataclasses
import enum
import sys
import types

from pydantic import BaseModel

from composeai._encoding import _REGISTRY, from_jsonable, to_jsonable
from composeai._schema import register_annotation_types, seal_schema


def _unregister(cls: type) -> None:
    tag = f"{cls.__module__}:{cls.__qualname__}"
    _REGISTRY.pop(tag, None)


class Inner(BaseModel):
    n: int


class Outer(BaseModel):
    inner: Inner
    label: str


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


@dataclasses.dataclass
class Point:
    x: int
    y: int


def test_register_annotation_types_registers_param_and_return_pydantic_models():
    def fn(a: Outer) -> Inner:
        raise NotImplementedError

    _unregister(Outer)
    _unregister(Inner)
    register_annotation_types(fn)

    encoded_outer = to_jsonable(Outer(inner=Inner(n=1), label="x"))
    # to_jsonable() also auto-registers; undo it so decoding below proves
    # *our* registration, not that side effect.
    _unregister(Outer)
    _unregister(Inner)
    register_annotation_types(fn)
    decoded = from_jsonable(encoded_outer)
    assert decoded == Outer(inner=Inner(n=1), label="x")


def test_register_annotation_types_walks_nested_pydantic_fields():
    def fn(x: Outer) -> None:
        raise NotImplementedError

    _unregister(Outer)
    _unregister(Inner)
    register_annotation_types(fn)

    # Inner is only reachable by walking Outer's model_fields -- prove it
    # was registered too, not just the top-level annotation.
    encoded_inner = to_jsonable(Inner(n=5))
    _unregister(Inner)
    register_annotation_types(fn)
    decoded = from_jsonable(encoded_inner)
    assert decoded == Inner(n=5)


def test_register_annotation_types_registers_enum():
    def fn(c: Color) -> str:
        raise NotImplementedError

    _unregister(Color)
    register_annotation_types(fn)
    encoded = to_jsonable(Color.RED)
    _unregister(Color)
    register_annotation_types(fn)
    decoded = from_jsonable(encoded)
    assert decoded is Color.RED


def test_register_annotation_types_registers_dataclass():
    def fn(p: Point) -> None:
        raise NotImplementedError

    _unregister(Point)
    register_annotation_types(fn)
    encoded = to_jsonable(Point(x=1, y=2))
    _unregister(Point)
    register_annotation_types(fn)
    decoded = from_jsonable(encoded)
    assert decoded == Point(x=1, y=2)


def test_register_annotation_types_walks_generic_containers():
    def fn(items: list[Inner]) -> dict[str, Outer]:
        raise NotImplementedError

    _unregister(Inner)
    _unregister(Outer)
    register_annotation_types(fn)

    assert from_jsonable(to_jsonable(Inner(n=9))) == Inner(n=9)
    _unregister(Inner)
    _unregister(Outer)
    register_annotation_types(fn)
    assert from_jsonable(to_jsonable(Outer(inner=Inner(n=9), label="y"))) == Outer(
        inner=Inner(n=9), label="y"
    )


def test_register_annotation_types_ignores_plain_builtin_annotations():
    def fn(a: int, b: str) -> bool:
        raise NotImplementedError

    # Must not raise for plain builtins.
    register_annotation_types(fn)


# --- dataclass fields under `from __future__ import annotations` -----------
#
# `from __future__ import annotations` is a per-module compiler directive
# (must be the module's first statement), so reproducing it needs a real,
# separate module -- built here via `exec()` into a fresh `types.ModuleType`
# registered in `sys.modules` (rather than a plain dict namespace) so
# `typing.get_type_hints` resolves names against the *same* globals the
# classes were defined with, the same way a real imported module would.


def _make_module_with_future_annotations(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)  # noqa: S102
    return module


_FUTURE_ANNOTATIONS_SOURCE = '''
from __future__ import annotations

import dataclasses
import enum


class FutureColor(enum.Enum):
    RED = "red"
    BLUE = "blue"


@dataclasses.dataclass
class FutureInner:
    color: FutureColor


@dataclasses.dataclass
class FutureOuter:
    inner: FutureInner
    label: str
'''


def test_register_annotation_types_recurses_into_dataclass_field_types_under_future_annotations():
    """Regression: `_register_type_recursive`'s dataclass branch used to
    recurse using `dataclasses.fields(annotation)[i].type` directly -- under
    `from __future__ import annotations` (postponed evaluation, which
    composeai's own source uses throughout), that's the *raw string*
    annotation, not the resolved class. `Outer` got registered; `Inner`
    (and anything reachable only through it) silently never did, so a fresh
    process that only imports the module (never encodes an `Inner` itself)
    couldn't decode a journaled value referencing it."""
    module = _make_module_with_future_annotations(
        "composeai_test_future_annotations_mod", _FUTURE_ANNOTATIONS_SOURCE
    )
    FutureOuter = module.FutureOuter
    FutureInner = module.FutureInner
    FutureColor = module.FutureColor

    # Confirms the actual root cause: an unresolved string, not a class.
    assert dataclasses.fields(FutureOuter)[0].type == "FutureInner"

    def fn(o: FutureOuter) -> None:  # pyright: ignore[reportInvalidTypeForm]
        raise NotImplementedError

    _unregister(FutureOuter)
    _unregister(FutureInner)
    _unregister(FutureColor)
    register_annotation_types(fn)

    # FutureInner (and, recursively, FutureColor) must be registered too --
    # not just the top-level FutureOuter annotation -- simulating a fresh
    # process that decodes an Inner value it never itself encoded.
    encoded_inner = to_jsonable(FutureInner(color=FutureColor.RED))
    _unregister(FutureInner)
    _unregister(FutureColor)
    register_annotation_types(fn)
    decoded = from_jsonable(encoded_inner)
    assert decoded == FutureInner(color=FutureColor.RED)


# --- seal_schema: dict[str, V] schemas must keep their value-type constraint --


def test_seal_schema_preserves_dict_value_schema_additional_properties():
    """Regression: seal_schema used to unconditionally overwrite
    `additionalProperties` with `False` whenever `type == "object"` --
    including the *schema-valued* `additionalProperties` pydantic generates
    for an open mapping like `dict[str, int]` (describing the permitted
    value type, not a boolean flag). That collapsed a perfectly usable
    `dict[str, int]` schema into one that only ever validates `{}`."""
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    seal_schema(schema)
    assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}


def test_seal_schema_still_seals_fixed_shape_objects_with_no_additional_properties():
    """The original, still-intended behavior: a fixed-shape object schema
    (pydantic model/dataclass -- no additionalProperties key at all, or an
    existing boolean one) still gets `additionalProperties: False` forced,
    everywhere in the tree including under $defs."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "$defs": {"Nested": {"type": "object", "properties": {"y": {"type": "string"}}}},
    }
    seal_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Nested"]["additionalProperties"] is False


def test_seal_schema_recurses_into_the_preserved_dict_value_schema_too():
    """A dict[str, SomeModel] value schema is itself sealed recursively --
    only the *outer* additionalProperties (the dict-shape marker) is
    protected from being overwritten, not exempted from sealing itself."""
    schema = {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
        },
    }
    seal_schema(schema)
    assert schema["additionalProperties"]["additionalProperties"] is False


# --- register_module_types: the whole-module companion to register_annotation_types --


def test_register_module_types_scans_a_module_namespace():
    import types as _types

    from pydantic import BaseModel

    from composeai import register_module_types
    from composeai._encoding import _REGISTRY, _type_tag

    class ScannedWidget(BaseModel):
        name: str

    fake_module = _types.ModuleType("fake_app_schemas")
    # setattr(), not `fake_module.ScannedWidget = ...`: ModuleType has no
    # statically-declared attributes, so a direct assignment is a pyright
    # reportAttributeAccessIssue; setattr()'s Any-typed second arg sidesteps
    # that at the cost of ruff's B010 (constant-attribute setattr), silenced
    # here since this dynamic-namespace test is exactly the justified case.
    setattr(fake_module, "ScannedWidget", ScannedWidget)  # noqa: B010

    _REGISTRY.pop(_type_tag(ScannedWidget), None)
    register_module_types(fake_module)
    assert _REGISTRY[_type_tag(ScannedWidget)] is ScannedWidget


def test_register_serializable_is_public():
    import composeai

    assert "register_serializable" in composeai.__all__
    assert "register_module_types" in composeai.__all__
