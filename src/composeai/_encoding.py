"""JSON-safe encode/decode for journal and trace persistence.

``to_jsonable``/``from_jsonable`` turn a restricted set of Python values
into plain ``dict``/``list``/scalar JSON trees (and back), tagging
non-trivial types with a reserved ``"$kind"`` key so they can be
rehydrated. Deliberately **no pickle** and **no dynamic import**:
decoding data that references an unregistered type raises
:class:`~composeai.errors.SerializationError` naming the missing type
rather than importing it -- guessing and importing arbitrary dotted
paths found in persisted data is a code-execution hole (this exact
issue was found and fixed in a prior audit). Callers must import the
module that defines a type -- or call :func:`register_serializable` --
before decoding data that references it.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
from typing import Any, TypeVar

from pydantic import BaseModel

from .errors import SerializationError

_KIND = "$kind"
_TYPE = "$type"
_VALUE = "value"

# Type registry: never used to import anything, only to look up classes
# that some part of this process has already imported and registered.
_REGISTRY: dict[str, type] = {}

_C = TypeVar("_C", bound=type)


def register_serializable(cls: _C) -> _C:
    """Register ``cls`` so :func:`from_jsonable` can rehydrate it by tag.

    Encoding a pydantic model, dataclass, or enum auto-registers its
    class. Call this explicitly in a process that decodes data it never
    encoded (e.g. a fresh journal reader started in a new process).
    Returns ``cls`` unchanged, so it can also be used as a decorator.

    Typed ``_C bound=type`` (not plain ``type``) so decorating a
    ``Generic`` dataclass (e.g. ``combinators.MapResult``) preserves its
    subscriptability to static checkers -- a bare ``-> type`` return would
    erase it to plain ``type``, making ``MapResult[int]`` a pyright error
    even though ``Generic`` supplies ``__class_getitem__`` at runtime.
    """
    _REGISTRY[_type_tag(cls)] = cls
    return cls


def _type_tag(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _index_path(path: str, index: int) -> str:
    return f"{path}[{index}]"


def _is_namedtuple(obj: Any) -> bool:
    """Best-effort, standard-library-idiomatic NamedTuple detection.

    ``isinstance(obj, tuple)`` is true for NamedTuple instances too (they're
    tuple subclasses) -- this must be checked *before* the generic tuple
    branch, or a NamedTuple silently degrades to a plain tuple.
    """
    return (
        isinstance(obj, tuple)
        and type(obj) is not tuple
        and hasattr(type(obj), "_fields")
        and hasattr(type(obj), "_asdict")
    )


def to_jsonable(obj: Any) -> Any:
    """Convert ``obj`` into a plain JSON-safe tree.

    Raises :class:`SerializationError` naming the offending path if any
    value nested in ``obj`` isn't supported.
    """
    return _encode(obj, "")


def _encode(obj: Any, path: str) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, BaseModel):
        register_serializable(type(obj))
        return {
            _KIND: "pydantic",
            _TYPE: _type_tag(type(obj)),
            _VALUE: obj.model_dump(mode="json"),
        }

    if isinstance(obj, enum.Enum):
        register_serializable(type(obj))
        return {_KIND: "enum", _TYPE: _type_tag(type(obj)), _VALUE: obj.name}

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        register_serializable(type(obj))
        non_init = [field.name for field in dataclasses.fields(obj) if not field.init]
        if non_init:
            raise SerializationError(
                f"{path or '<root>'}: dataclass {type(obj).__name__!r} has non-init "
                f"field(s) {non_init!r} (declared with field(init=False), typically "
                "computed in __post_init__) -- these can't round-trip the journal: "
                "encoding would capture the computed value, but decoding calls "
                "cls(**fields), which rejects any keyword the real __init__ doesn't "
                "accept. Exclude non-init fields from what you journal (e.g. store a "
                "plain dict/dataclass of only the init fields), or give this "
                "dataclass a classmethod constructor you call explicitly on decode."
            )
        value = {
            field.name: _encode(getattr(obj, field.name), _child_path(path, field.name))
            for field in dataclasses.fields(obj)
        }
        return {_KIND: "dataclass", _TYPE: _type_tag(type(obj)), _VALUE: value}

    if isinstance(obj, datetime.datetime):
        return {_KIND: "datetime", _VALUE: obj.isoformat()}
    if isinstance(obj, datetime.date):
        return {_KIND: "date", _VALUE: obj.isoformat()}
    if isinstance(obj, datetime.time):
        return {_KIND: "time", _VALUE: obj.isoformat()}

    if _is_namedtuple(obj):
        raise SerializationError(
            f"{path or '<root>'}: {type(obj).__name__!r} is a NamedTuple, which the "
            "journal encoder doesn't support -- it's a tuple subclass, so encoding it "
            "as a plain tuple would silently lose field-name access on decode (an "
            "AttributeError on the replayed value where the original worked). Convert "
            "it to a dataclass, a dict, or a plain tuple before journaling it."
        )
    if isinstance(obj, tuple):
        return {
            _KIND: "tuple",
            _VALUE: [_encode(v, _index_path(path, i)) for i, v in enumerate(obj)],
        }
    if isinstance(obj, frozenset):
        return {
            _KIND: "frozenset",
            _VALUE: [_encode(v, _index_path(path, i)) for i, v in enumerate(obj)],
        }
    if isinstance(obj, set):
        return {
            _KIND: "set",
            _VALUE: [_encode(v, _index_path(path, i)) for i, v in enumerate(obj)],
        }

    if isinstance(obj, list):
        return [_encode(v, _index_path(path, i)) for i, v in enumerate(obj)]

    if isinstance(obj, dict):
        for key in obj:
            if not isinstance(key, str):
                raise SerializationError(
                    f"{path or '<root>'}: dict keys must be str, got {key!r}"
                )
        encoded = {k: _encode(v, _child_path(path, k)) for k, v in obj.items()}
        if _KIND in obj:
            # A plain dict that happens to use our reserved key: escape it
            # so decoding is unambiguous.
            return {_KIND: "dict", _VALUE: encoded}
        return encoded

    raise SerializationError(f"{path or '<root>'}: {type(obj)!r} is not JSON-serializable")


def from_jsonable(data: Any) -> Any:
    """Inverse of :func:`to_jsonable`.

    Raises :class:`SerializationError` if a ``$type`` tag isn't in the
    registry (see :func:`register_serializable`). Never imports modules
    to resolve a type.
    """
    if isinstance(data, dict):
        if _KIND in data:
            return _decode_tagged(data)
        return {k: from_jsonable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [from_jsonable(v) for v in data]
    return data


def _decode_tagged(data: dict[str, Any]) -> Any:
    kind = data.get(_KIND)
    try:
        if kind == "dict":
            return {k: from_jsonable(v) for k, v in data[_VALUE].items()}
        if kind == "pydantic":
            cls = _resolve_type(data[_TYPE])
            value = {k: from_jsonable(v) for k, v in data[_VALUE].items()}
            return cls.model_validate(value)
        if kind == "dataclass":
            cls = _resolve_type(data[_TYPE])
            value = {k: from_jsonable(v) for k, v in data[_VALUE].items()}
            return cls(**value)
        if kind == "enum":
            cls = _resolve_type(data[_TYPE])
            return cls[data[_VALUE]]
        if kind == "datetime":
            return datetime.datetime.fromisoformat(data[_VALUE])
        if kind == "date":
            return datetime.date.fromisoformat(data[_VALUE])
        if kind == "time":
            return datetime.time.fromisoformat(data[_VALUE])
        if kind == "tuple":
            return tuple(from_jsonable(v) for v in data[_VALUE])
        if kind == "set":
            return {from_jsonable(v) for v in data[_VALUE]}
        if kind == "frozenset":
            return frozenset(from_jsonable(v) for v in data[_VALUE])
    except SerializationError:
        raise  # already the right type (e.g. _resolve_type's "unregistered type") -- pass through
    except (KeyError, TypeError) as exc:
        # A tagged value missing an expected key (e.g. `{"$kind": "dataclass",
        # "value": {...}}` with no "$type" -- DB corruption, a manual edit, a
        # future schema/version mismatch, ...) used to raise a raw KeyError/
        # TypeError here instead of this module's own SerializationError, so
        # callers that specifically catch SerializationError (the module's
        # documented contract) didn't catch this.
        raise SerializationError(
            f"corrupted journal/state row (kind={kind!r}): {type(exc).__name__}: {exc}"
        ) from exc

    raise SerializationError(f"Unknown {_KIND!r} tag: {kind!r}")


def _resolve_type(tag: str) -> Any:
    cls = _REGISTRY.get(tag)
    if cls is not None:
        return cls
    module_hint = tag.split(":", 1)[0]
    raise SerializationError(
        f"Cannot decode unregistered type {tag!r}: import the module that "
        f"defines it (e.g. `import {module_hint}`) or call "
        f"register_serializable(...) before decoding this data. "
        f"Types are never imported automatically, by design (security)."
    )
