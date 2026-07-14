"""Shared JSON Schema post-processing for ``@tool`` and ``@agent`` structured output.

Both derive their schemas from pydantic (``create_model`` / ``TypeAdapter``) and
both need the same treatment before handing a schema to a model: strict
``additionalProperties: false`` everywhere -- including nested ``$defs`` --
and no pydantic-generated ``title`` noise (models rely on ``description``,
not ``title``, and a stray auto-title only wastes tokens).
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
from collections.abc import Callable
from typing import Any, get_args, get_type_hints

from pydantic import BaseModel

from ._encoding import register_serializable


def resolve_annotations(fn: Callable[..., Any], *, include_extras: bool = True) -> dict[str, Any]:
    """Best-effort ``typing.get_type_hints`` for ``fn``.

    ``get_type_hints`` resolves string-form annotations (e.g. under
    ``from __future__ import annotations``) against ``fn.__globals__``, but
    it cannot see names from an enclosing function's local/closure scope --
    a common pattern in tests that define a small model right next to the
    function using it. When that lookup fails, fall back to each
    parameter's raw annotation from :func:`inspect.signature`: on every
    supported Python version, an annotation that's already a real object
    (the common case without the ``__future__`` import, and -- since
    Python 3.14's lazy annotation evaluation -- even local/closure names
    under it) survives untouched; only an unresolved string annotation
    combining *both* the ``__future__`` import *and* a local/closure name
    would still come back as a plain string here.
    """
    try:
        return get_type_hints(fn, include_extras=include_extras)
    except NameError:
        pass

    sig = inspect.signature(fn)
    raw: dict[str, Any] = {
        name: param.annotation
        for name, param in sig.parameters.items()
        if param.annotation is not inspect.Parameter.empty
    }
    if sig.return_annotation is not inspect.Signature.empty:
        raw["return"] = sig.return_annotation
    return raw


def register_annotation_types(fn: Callable[..., Any]) -> None:
    """Register every pydantic model / dataclass / enum reachable from ``fn``'s annotations.

    Walks :func:`resolve_annotations` (every parameter plus the return
    type), recursing into generic args (``typing.get_args`` -- e.g. the
    ``Inner`` in ``list[Inner]`` or ``dict[str, Outer]``) and, for
    pydantic models and dataclasses, into their own field types. Called by
    ``@task``/``@flow``/``@agent`` decoration so a *fresh process* (e.g. a
    resumed flow started via a new ``python`` invocation) can decode
    journaled values referencing these types via
    :func:`~composeai._encoding.from_jsonable` without composeai ever
    importing a type dynamically -- that would be a code-execution hole,
    so decoding an unregistered type always raises instead (see
    :mod:`composeai._encoding`).
    """
    hints = resolve_annotations(fn, include_extras=True)
    seen: set[type] = set()
    for value in hints.values():
        _register_type_recursive(value, seen)


def register_module_types(module: Any) -> None:
    """Register every pydantic model / dataclass / enum in ``module``'s namespace.

    The whole-module companion to :func:`register_annotation_types`, for
    processes that decode data they never encoded -- ``compose --import``
    (see :mod:`composeai.cli`) and app-side trace tooling. Scans
    ``vars(module)`` and recurses into field types the same way decoration-
    time registration does. Re-exported types register under their *true*
    defining module (the registry key comes from ``cls.__module__``), so
    scanning a barrel/schemas module that re-exports types works. Never
    imports anything itself -- the caller chose to import ``module``.
    """
    seen: set[type] = set()
    for value in vars(module).values():
        _register_type_recursive(value, seen)


def _register_type_recursive(annotation: Any, seen: set[type]) -> None:
    if isinstance(annotation, type):
        if annotation in seen:
            return
        seen.add(annotation)
        if issubclass(annotation, BaseModel):
            register_serializable(annotation)
            for field_info in annotation.model_fields.values():
                _register_type_recursive(field_info.annotation, seen)
        elif dataclasses.is_dataclass(annotation):
            register_serializable(annotation)
            # `dataclasses.fields(annotation)[i].type` is the field's *raw*
            # annotation -- under `from __future__ import annotations` (or
            # any other postponed-evaluation source), that's an unresolved
            # string (e.g. `"Inner"`), not the actual class object. Recursing
            # on it directly means `isinstance(..., type)` is always False
            # and `get_args(...)` always empty, so a dataclass field typed
            # with another custom dataclass/enum/pydantic model silently
            # never gets registered -- exactly the gap the pydantic branch
            # above doesn't have, since `model_fields[...].annotation` is
            # always pydantic-resolved to a real type object already.
            # `typing.get_type_hints` resolves the same way `model_fields`
            # does; fall back to the raw (possibly-string) annotation if
            # resolution itself fails (e.g. a name only in a caller's local
            # scope -- see `resolve_annotations`'s own docstring for the
            # same trade-off).
            try:
                hints = get_type_hints(annotation, include_extras=True)
            except NameError:
                hints = {}
            for field in dataclasses.fields(annotation):
                _register_type_recursive(hints.get(field.name, field.type), seen)
        elif issubclass(annotation, enum.Enum):
            register_serializable(annotation)
        return

    for arg in get_args(annotation):
        _register_type_recursive(arg, seen)


def seal_schema(schema: Any) -> None:
    """Recursively strip auto-generated ``title`` keys and seal every object schema.

    Mutates ``schema`` in place. "Object schema" means any nested dict with
    ``"type": "object"`` -- including entries under ``$defs`` -- which gets
    ``additionalProperties: false`` set, *unless* it already has a
    dict-valued ``additionalProperties`` (a real schema, not a boolean --
    what pydantic generates for an open mapping like ``dict[str, V]``/
    ``Mapping[str, V]``, describing the permitted *value* type). Overwriting
    that with ``False`` would turn a schema that legitimately accepts any
    string-keyed dict of ``V`` into one that only ever matches ``{}`` --
    silently forbidding the model from ever populating the field/argument
    it's meant to constrain, not forbid. A boolean (or absent)
    ``additionalProperties`` is still always forced to ``False`` (the
    fixed-shape-object case this was designed for -- pydantic models and
    dataclasses). Non-dict values (e.g. entries of a ``"required"`` list)
    are left untouched.
    """
    if not isinstance(schema, dict):
        return
    # Auto-generated titles are always strings; a dict under a "title" key is
    # a *property named title* (inside a "properties" map) and must survive.
    if isinstance(schema.get("title"), str):
        schema.pop("title")
    if schema.get("type") == "object" and not isinstance(schema.get("additionalProperties"), dict):
        schema["additionalProperties"] = False
    for value in schema.values():
        if isinstance(value, dict):
            seal_schema(value)
        elif isinstance(value, list):
            for item in value:
                seal_schema(item)
