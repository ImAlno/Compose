"""Tests for ``composeai.tools.tool`` (Phase 3)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from composeai.errors import ConfigError
from composeai.tools import Tool, tool


def _all_titles(schema):
    """Collect every 'title' key found anywhere (recursively) in a schema."""
    found = []

    def _walk(node):
        if isinstance(node, dict):
            if "title" in node:
                found.append(node["title"])
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return found


def _all_object_schemas(schema):
    """Collect every nested dict with type == "object" (including $defs)."""
    found = []

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                found.append(node)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return found


# --- basic decoration: bare vs. with-args ------------------------------------


def test_tool_bare_produces_tool_instance():
    @tool
    def search(query: str) -> str:
        """Search the web."""
        return f"results for {query}"

    assert isinstance(search, Tool)
    assert search.spec.name == "search"


def test_tool_with_args_produces_tool_instance():
    @tool(name="websearch", description="Custom description.", requires_approval=True)
    def search(query: str) -> str:
        """Search the web."""
        return f"results for {query}"

    assert isinstance(search, Tool)
    assert search.spec.name == "websearch"
    assert search.spec.description == "Custom description."
    assert search.spec.requires_approval is True


def test_tool_still_callable_as_plain_function():
    @tool
    def double(x: int) -> int:
        """Double a number.

        Args:
            x: The number to double.
        """
        return x * 2

    assert double(5) == 10
    assert double(x=5) == 10


# --- spec: name / description / requires_approval defaults ------------------


def test_tool_name_defaults_to_function_name():
    @tool
    def my_tool_fn(x: int) -> int:
        """Do a thing."""
        return x

    assert my_tool_fn.spec.name == "my_tool_fn"


def test_tool_description_defaults_to_docstring_minus_args_section():
    @tool
    def search(query: str, max_results: int = 5) -> str:
        """Search the web.

        Args:
            query: What to search for.
            max_results: Maximum number of results.
        """
        return query

    assert search.spec.description == "Search the web."
    assert "Args:" not in search.spec.description


def test_tool_description_dedented_multiline():
    @tool
    def search(query: str) -> str:
        """Search the web.

        This does a full web search and returns text snippets.

        Args:
            query: What to search for.
        """
        return query

    assert search.spec.description == (
        "Search the web.\n\nThis does a full web search and returns text snippets."
    )


def test_tool_requires_approval_defaults_false():
    @tool
    def search(query: str) -> str:
        """Search the web."""
        return query

    assert search.spec.requires_approval is False


def test_tool_strict_is_always_true():
    @tool
    def search(query: str) -> str:
        """Search the web."""
        return query

    assert search.spec.strict is True


# --- schema generation: types / defaults / required --------------------------


def test_schema_required_param_has_no_default():
    @tool
    def search(query: str, max_results: int = 5) -> str:
        """Search the web.

        Args:
            query: What to search for.
            max_results: Maximum number of results.
        """
        return query

    schema = search.input_schema
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["max_results"]["type"] == "integer"
    assert schema["properties"]["max_results"]["default"] == 5


def test_schema_multiple_required_params():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        return a + b

    schema = add.spec.input_schema
    assert set(schema["required"]) == {"a", "b"}


# --- nested pydantic param ----------------------------------------------------


class _Coordinates(BaseModel):
    lat: float
    lon: float


def test_schema_nested_pydantic_param():
    @tool
    def locate(place: str, coords: _Coordinates) -> str:
        """Locate a place.

        Args:
            place: The place name.
            coords: Its coordinates.
        """
        return place

    schema = locate.spec.input_schema
    assert "$defs" in schema
    assert "_Coordinates" in schema["$defs"]
    nested = schema["$defs"]["_Coordinates"]
    assert nested["type"] == "object"
    assert nested["additionalProperties"] is False
    assert "lat" in nested["properties"]


# --- docstring arg descriptions merged ---------------------------------------


def test_docstring_arg_descriptions_merged_into_schema():
    @tool
    def search(query: str, max_results: int = 5) -> str:
        """Search the web.

        Args:
            query: What to search for.
            max_results: Maximum number of results.
        """
        return query

    schema = search.input_schema
    assert schema["properties"]["query"]["description"] == "What to search for."
    assert schema["properties"]["max_results"]["description"] == "Maximum number of results."


def test_docstring_arg_description_continuation_lines_joined():
    @tool
    def search(query: str) -> str:
        """Search the web.

        Args:
            query: What to search for. This can span
                multiple lines in the docstring.
        """
        return query

    schema = search.input_schema
    assert schema["properties"]["query"]["description"] == (
        "What to search for. This can span multiple lines in the docstring."
    )


# --- additionalProperties false incl. $defs; titles stripped ----------------


def test_schema_additional_properties_false_everywhere():
    @tool
    def locate(place: str, coords: _Coordinates) -> str:
        """Locate a place.

        Args:
            place: The place name.
            coords: Its coordinates.
        """
        return place

    schema = locate.spec.input_schema
    object_schemas = _all_object_schemas(schema)
    assert object_schemas  # sanity: found at least the root
    for obj_schema in object_schemas:
        assert obj_schema["additionalProperties"] is False


def test_schema_titles_stripped_everywhere():
    @tool
    def locate(place: str, coords: _Coordinates) -> str:
        """Locate a place.

        Args:
            place: The place name.
            coords: Its coordinates.
        """
        return place

    schema = locate.spec.input_schema
    assert _all_titles(schema) == []


# --- errors: missing docstring / *args / **kwargs ----------------------------


def test_missing_docstring_raises_config_error():
    with pytest.raises(ConfigError, match="docstring"):

        @tool
        def search(query: str) -> str:
            return query


def test_missing_docstring_ok_with_explicit_description():
    @tool(description="Search the web.")
    def search(query: str) -> str:
        return query

    assert search.spec.description == "Search the web."


def test_var_positional_args_raises_config_error():
    with pytest.raises(ConfigError, match=r"\*args"):

        @tool
        def search(*args) -> str:
            """Search the web."""
            return " ".join(args)


def test_var_keyword_args_raises_config_error():
    with pytest.raises(ConfigError, match=r"\*\*kwargs"):

        @tool
        def search(**kwargs) -> str:
            """Search the web."""
            return str(kwargs)


# --- execute: coercion + return encoding -------------------------------------


def test_execute_coerces_string_to_int():
    @tool
    def repeat(text: str, times: int) -> str:
        """Repeat text.

        Args:
            text: The text to repeat.
            times: How many times.
        """
        return text * times

    assert repeat.execute({"text": "ab", "times": "3"}) == "ababab"


def test_execute_returns_str_as_is():
    @tool
    def shout(text: str) -> str:
        """Shout text."""
        return text.upper()

    assert shout.execute({"text": "hi"}) == "HI"


def test_execute_json_encodes_non_str_return():
    @tool
    def add(a: int, b: int) -> int:
        """Add numbers.

        Args:
            a: First.
            b: Second.
        """
        return a + b

    result = add.execute({"a": 2, "b": 3})
    assert isinstance(result, str)
    assert json.loads(result) == 5


def test_execute_json_encodes_dict_return():
    @tool
    def info(name: str) -> dict:
        """Get info.

        Args:
            name: The name.
        """
        return {"name": name, "ok": True}

    result = info.execute({"name": "x"})
    assert json.loads(result) == {"name": "x", "ok": True}


def test_execute_propagates_exceptions():
    @tool
    def boom() -> str:
        """Explode."""
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        boom.execute({})


def test_schema_keeps_property_named_title():
    """Regression: seal_schema must not delete a field literally named `title`."""

    @tool
    def make_article(title: str, body: str) -> str:
        """Create an article.

        Args:
            title: The article title.
            body: The article body.
        """
        return f"{title}: {body}"

    schema = make_article.spec.input_schema
    assert "title" in schema["properties"]
    assert schema["properties"]["title"]["type"] == "string"
    assert "title" in schema["required"]


# --- capstone fix wave A regressions -------------------------------------------


def test_dict_str_int_param_keeps_value_type_schema_not_collapsed_to_empty_object():
    """Regression: seal_schema used to overwrite a dict[str, V] parameter's
    schema-valued additionalProperties with False, collapsing it into a
    schema that only ever validates {}."""

    @tool
    def counter(counts: dict[str, int]) -> str:
        """Report counts.

        Args:
            counts: name -> count mapping
        """
        return str(counts)

    prop = counter.spec.input_schema["properties"]["counts"]
    assert prop["type"] == "object"
    assert prop["additionalProperties"] == {"type": "integer"}


def test_execute_rejects_extra_arguments_not_in_the_schema():
    """Regression: the schema advertised to the model always has
    additionalProperties: false, but local re-validation used pydantic's
    default extra="ignore", silently dropping unknown keys instead of
    rejecting the call -- a quiet mismatch between what's advertised and
    what's enforced."""

    @tool
    def greet(name: str) -> str:
        """Greet someone.

        Args:
            name: who to greet
        """
        return f"hi {name}"

    with pytest.raises(Exception):  # noqa: B017 -- pydantic's ValidationError
        greet.execute({"name": "Ada", "unexpected_extra_key": "boom"})

    # A well-formed call still works.
    assert greet.execute({"name": "Ada"}) == "hi Ada"


def test_unmatched_docstring_arg_name_raises_config_error_at_decoration():
    """Regression: a docstring Args: entry naming a parameter that doesn't
    exist (e.g. after a rename) used to be silently dropped -- no error,
    and the real, now-undocumented parameter shipped with no description."""
    with pytest.raises(ConfigError, match="search_query"):

        @tool
        def search(query: str) -> str:
            """Search for something.

            Args:
                search_query: what to look for
            """
            return query


def test_tool_exposes_input_output_types_for_composition_type_checking():
    """Regression: a @tool used directly as a pipe()/aggregate() stage lost
    composition-time type checking -- input_type/output_type were derived
    from Tool.__call__'s untyped (*args, **kwargs) signature instead of the
    wrapped function's real one, silently defaulting both to Any."""

    @tool
    def search(query: str) -> str:
        """Search for something.

        Args:
            query: what to look for
        """
        return query

    assert search.input_type is str
    assert search.output_type is str
