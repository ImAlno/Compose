import dataclasses
import datetime
import enum
import sys

import pytest
from pydantic import BaseModel

from composeai import _encoding
from composeai._encoding import from_jsonable, register_serializable, to_jsonable
from composeai.errors import SerializationError


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


@dataclasses.dataclass
class Point:
    x: int
    y: int


class Person(BaseModel):
    name: str
    age: int


class NotSerializable:
    """A plain object with no encoding support."""


# --- passthrough ---


def test_passthrough_scalars_and_none():
    for value in [None, True, False, 0, 1, -5, 1.5, "text", ""]:
        assert to_jsonable(value) == value


def test_passthrough_list_and_plain_str_keyed_dict():
    data = {"a": [1, 2, {"b": "c", "d": None}]}
    assert to_jsonable(data) == data
    assert from_jsonable(to_jsonable(data)) == data


# --- tagged kinds: round trip ---


def test_pydantic_round_trip():
    person = Person(name="Ada", age=30)
    encoded = to_jsonable(person)
    assert encoded["$kind"] == "pydantic"
    assert encoded["$type"] == f"{Person.__module__}:{Person.__qualname__}"
    assert encoded["value"] == {"name": "Ada", "age": 30}
    decoded = from_jsonable(encoded)
    assert decoded == person
    assert isinstance(decoded, Person)


def test_dataclass_round_trip():
    point = Point(1, 2)
    encoded = to_jsonable(point)
    assert encoded["$kind"] == "dataclass"
    assert encoded["$type"] == f"{Point.__module__}:{Point.__qualname__}"
    assert encoded["value"] == {"x": 1, "y": 2}
    decoded = from_jsonable(encoded)
    assert decoded == point
    assert isinstance(decoded, Point)


def test_enum_round_trip():
    encoded = to_jsonable(Color.RED)
    assert encoded == {
        "$kind": "enum",
        "$type": f"{Color.__module__}:{Color.__qualname__}",
        "value": "RED",
    }
    decoded = from_jsonable(encoded)
    assert decoded is Color.RED


def test_datetime_round_trip():
    dt = datetime.datetime(2026, 7, 12, 10, 30, 0)
    encoded = to_jsonable(dt)
    assert encoded == {"$kind": "datetime", "value": dt.isoformat()}
    assert from_jsonable(encoded) == dt


def test_date_round_trip():
    d = datetime.date(2026, 7, 12)
    encoded = to_jsonable(d)
    assert encoded == {"$kind": "date", "value": d.isoformat()}
    assert from_jsonable(encoded) == d


def test_time_round_trip():
    t = datetime.time(10, 30, 0)
    encoded = to_jsonable(t)
    assert encoded == {"$kind": "time", "value": t.isoformat()}
    assert from_jsonable(encoded) == t


def test_tuple_round_trip():
    encoded = to_jsonable((1, "a", None))
    assert encoded == {"$kind": "tuple", "value": [1, "a", None]}
    decoded = from_jsonable(encoded)
    assert decoded == (1, "a", None)
    assert isinstance(decoded, tuple)


def test_namedtuple_encoding_raises_serialization_error_instead_of_degrading():
    """Regression: `isinstance(obj, tuple)` fired for NamedTuple instances
    too (they're tuple subclasses) before any NamedTuple-specific handling,
    silently encoding them as a plain tuple -- decoding always reconstructed
    a plain tuple, losing field-name access. An explicit error at encode
    time is safer than a silent degrade that only breaks later, on
    ``result.field_name``."""
    import typing

    class Point3(typing.NamedTuple):
        x: int
        y: int

    with pytest.raises(SerializationError, match="NamedTuple"):
        to_jsonable(Point3(1, 2))

    # A plain tuple (not a NamedTuple) is unaffected.
    assert to_jsonable((1, 2)) == {"$kind": "tuple", "value": [1, 2]}


def test_set_round_trip():
    encoded = to_jsonable({1, 2, 3})
    assert encoded["$kind"] == "set"
    assert sorted(encoded["value"]) == [1, 2, 3]
    decoded = from_jsonable(encoded)
    assert decoded == {1, 2, 3}
    assert isinstance(decoded, set)


def test_frozenset_round_trip():
    encoded = to_jsonable(frozenset({1, 2}))
    assert encoded["$kind"] == "frozenset"
    decoded = from_jsonable(encoded)
    assert decoded == frozenset({1, 2})
    assert isinstance(decoded, frozenset)


def test_nested_structures_round_trip():
    @dataclasses.dataclass
    class Wrapper:
        point: Point
        color: Color
        when: datetime.datetime
        tags: tuple[str, ...]

    original = Wrapper(
        point=Point(1, 2),
        color=Color.BLUE,
        when=datetime.datetime(2026, 1, 1),
        tags=("a", "b"),
    )
    encoded = to_jsonable(original)
    # Nested dataclass field must itself be tagged (recursive to_jsonable).
    assert encoded["value"]["point"]["$kind"] == "dataclass"
    assert encoded["value"]["color"]["$kind"] == "enum"
    decoded = from_jsonable(encoded)
    assert decoded == original


def test_list_of_mixed_tagged_values_round_trip():
    data = [Color.RED, Point(1, 2), datetime.date(2020, 1, 1), (1, 2)]
    encoded = to_jsonable(data)
    decoded = from_jsonable(encoded)
    assert decoded == data


# --- $kind collision escape ---


def test_kind_collision_is_escaped_and_round_trips():
    tricky = {"$kind": "not_a_real_tag", "value": 42}
    encoded = to_jsonable(tricky)
    assert encoded == {
        "$kind": "dict",
        "value": {"$kind": "not_a_real_tag", "value": 42},
    }
    decoded = from_jsonable(encoded)
    assert decoded == tricky


def test_nested_kind_collision_inside_normal_dict():
    data = {"outer": {"$kind": "enum", "value": "sneaky"}}
    encoded = to_jsonable(data)
    decoded = from_jsonable(encoded)
    assert decoded == data


# --- SerializationError path naming ---


def test_serialization_error_names_nested_path():
    data = {"pages": [0, 1, 2, {"meta": NotSerializable()}]}
    with pytest.raises(SerializationError) as exc_info:
        to_jsonable(data)
    assert "pages[3].meta" in str(exc_info.value)


def test_serialization_error_at_root_has_no_stray_prefix():
    with pytest.raises(SerializationError) as exc_info:
        to_jsonable(NotSerializable())
    message = str(exc_info.value)
    assert "NotSerializable" in message


def test_serialization_error_for_non_str_dict_keys():
    with pytest.raises(SerializationError):
        to_jsonable({1: "a"})


# --- decode-unknown-type: raises, never imports ---


def test_decode_unknown_type_raises_and_does_not_import_module():
    fake_module = "nonexistent_module_xyz"
    assert fake_module not in sys.modules
    payload = {"$kind": "enum", "$type": f"{fake_module}:Foo", "value": "X"}
    with pytest.raises(SerializationError) as exc_info:
        from_jsonable(payload)
    assert fake_module not in sys.modules
    assert fake_module in str(exc_info.value)


def test_decode_unknown_dataclass_type_raises():
    payload = {"$kind": "dataclass", "$type": "nonexistent_module_xyz:Bar", "value": {}}
    with pytest.raises(SerializationError):
        from_jsonable(payload)


# --- corrupted rows: KeyError/TypeError -> SerializationError -----------------


def test_corrupted_row_missing_type_key_raises_serialization_error_not_keyerror():
    """Regression: a tagged value missing an expected key (DB corruption, a
    manual edit, a future schema/version mismatch) used to raise a raw,
    untyped KeyError instead of this module's own SerializationError, so
    callers that specifically catch SerializationError (the documented
    contract) didn't catch this."""
    payload = {"$kind": "dataclass", "value": {}}  # no "$type"
    with pytest.raises(SerializationError):
        from_jsonable(payload)


def test_corrupted_row_missing_value_key_raises_serialization_error():
    payload = {"$kind": "enum", "$type": f"{Color.__module__}:{Color.__qualname__}"}
    with pytest.raises(SerializationError):
        from_jsonable(payload)


def test_corrupted_row_dataclass_constructor_type_error_raises_serialization_error():
    """A tagged dataclass payload whose fields don't match the real
    __init__ (e.g. an extra/renamed key from a stale schema) raises
    SerializationError, not a raw TypeError from `cls(**value)`."""
    payload = {
        "$kind": "dataclass",
        "$type": f"{Point.__module__}:{Point.__qualname__}",
        "value": {"x": 1, "y": 2, "unexpected_extra_field": 3},
    }
    with pytest.raises(SerializationError):
        from_jsonable(payload)


# --- init=False dataclass fields: rejected at encode, not decode --------------


def test_dataclass_with_init_false_field_raises_serialization_error_at_encode():
    """Regression: a dataclass with a computed `field(init=False)` field
    encoded successfully (capturing the computed value), but decoding
    raised an unhandled `TypeError: __init__() got an unexpected keyword
    argument` -- the encoder walked every field with no filter on
    `field.init`, and the decoder passed every encoded field name as a
    constructor keyword, including ones the real __init__ doesn't accept.
    Now this is rejected at *encode* time, naming the limitation, instead
    of round-tripping successfully and only failing later on decode."""

    @dataclasses.dataclass
    class Derived:
        x: int
        y: int = dataclasses.field(init=False)

        def __post_init__(self):
            self.y = self.x * 2

    with pytest.raises(SerializationError, match="init=False"):
        to_jsonable(Derived(x=5))


# --- register_serializable ---


def test_register_serializable_enables_decode_in_fresh_registry():
    @dataclasses.dataclass
    class LocalThing:
        value: int

    encoded = to_jsonable(LocalThing(5))  # auto-registers LocalThing
    tag = encoded["$type"]

    # Simulate a fresh process/registry that never encoded LocalThing.
    saved = _encoding._REGISTRY.pop(tag)
    try:
        with pytest.raises(SerializationError):
            from_jsonable(encoded)

        register_serializable(LocalThing)
        decoded = from_jsonable(encoded)
        assert decoded == LocalThing(5)
    finally:
        _encoding._REGISTRY[tag] = saved


# --- integration with composeai.messages ---


def test_encodes_and_decodes_message_and_usage():
    from composeai.messages import Message, Usage

    msg = Message.user("hello")
    usage = Usage(input_tokens=5, cost_usd=0.02)
    payload = {"message": msg, "usage": usage}

    encoded = to_jsonable(payload)
    decoded = from_jsonable(encoded)

    assert decoded["message"] == msg
    assert decoded["usage"] == usage
