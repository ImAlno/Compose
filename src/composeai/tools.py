"""``@tool``: turn a plain, typed function into a model-callable :class:`Tool`.

The decorator does three things a hand-rolled tool declaration would
otherwise require boilerplate for: build a strict JSON Schema from the
function's signature (via a throwaway pydantic model), parse a Google-style
docstring for the tool's own description and its per-argument descriptions,
and validate/coerce arguments back into typed kwargs when the tool actually
runs. The decorated object stays a plain callable -- ``search("x")`` still
just calls the function -- with the model-facing machinery attached as
extra attributes.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from typing import Any, overload

from pydantic import BaseModel, ConfigDict, create_model

from ._encoding import to_jsonable
from ._schema import resolve_annotations, seal_schema
from .errors import ConfigError
from .models.base import ToolSpec

_ARG_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]+)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<desc>.*)$"
)


class Tool:
    """The callable produced by ``@tool``.

    Still plainly callable as the original function (``tool_obj(...)``
    bypasses schema validation entirely, same as calling the undecorated
    function). ``.spec`` is the :class:`~composeai.models.base.ToolSpec`
    handed to models; ``.execute`` is what the agent loop calls to run the
    tool from model-supplied JSON arguments.
    """

    def __init__(
        self, fn: Callable[..., Any], spec: ToolSpec, param_model: type[BaseModel]
    ) -> None:
        self.fn = fn
        self.spec = spec
        self._param_model = param_model
        self.__name__ = spec.name
        self.__doc__ = fn.__doc__
        # Composition-time type checking (composeai.combinators.pipe/aggregate)
        # for a @tool used directly as a stage -- derived from `fn`'s own
        # signature, not `Tool.__call__`'s untyped `(*args, **kwargs)` one.
        self.input_type, self.output_type = _tool_types(fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.spec.input_schema

    def execute(self, arguments: dict[str, Any]) -> str:
        """Validate/coerce ``arguments`` through the signature model and call the fn.

        Returns the fn's return value verbatim if it's already a ``str``,
        otherwise JSON-encodes it (via :func:`~composeai._encoding.to_jsonable`).
        Exceptions from validation or from the fn itself propagate
        unchanged -- the agent loop decides how to turn them into a tool
        result.
        """
        validated = self._param_model.model_validate(arguments)
        kwargs = {name: getattr(validated, name) for name in self._param_model.model_fields}
        result = self.fn(**kwargs)
        if isinstance(result, str):
            return result
        return json.dumps(to_jsonable(result))


@overload
def tool(fn: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorate a plain, typed function into a model-callable :class:`Tool`.

    Usable bare (``@tool``) or with arguments (``@tool(name=..., description=...,
    requires_approval=...)``). The function's docstring supplies the tool's
    description and, via a Google-style ``Args:`` section, per-parameter
    descriptions merged into the generated JSON Schema -- unless overridden
    by an explicit ``description=``.
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return _build_tool(
            func, name=name, description=description, requires_approval=requires_approval
        )

    if fn is not None:
        return decorator(fn)
    return decorator


def _build_tool(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    requires_approval: bool,
) -> Tool:
    tool_name = name or fn.__name__
    doc = fn.__doc__

    if description is not None:
        final_description = description
        _, arg_descriptions = _parse_docstring(doc) if doc and doc.strip() else ("", {})
    else:
        if not doc or not doc.strip():
            raise ConfigError(
                f"@tool function {fn.__name__!r} has no docstring: tools need a "
                "description because the model relies on it to decide when (and "
                "how) to call them. Add a docstring, or pass description=... "
                "explicitly to @tool."
            )
        final_description, arg_descriptions = _parse_docstring(doc)

    param_model = _build_param_model(fn, tool_name)
    schema = param_model.model_json_schema()
    _merge_arg_descriptions(fn.__name__, schema, arg_descriptions)
    seal_schema(schema)

    spec = ToolSpec(
        name=tool_name,
        description=final_description,
        input_schema=schema,
        requires_approval=requires_approval,
        strict=True,
    )
    return Tool(fn=fn, spec=spec, param_model=param_model)


def _tool_types(fn: Callable[..., Any]) -> tuple[Any, Any]:
    """``(input_type, output_type)`` for ``fn``'s first parameter/return annotation.

    Powers ``Tool.input_type``/``Tool.output_type`` -- so a ``@tool`` object
    used directly as a ``pipe()``/``aggregate()`` stage (a use the code
    already anticipates -- see ``composeai.combinators``'s ``_stage_name``)
    gets real composition-time type checking instead of silently defaulting
    to ``Any`` for both (which happened when the type-introspection
    machinery there resolved against ``Tool.__call__``'s untyped, variadic
    ``(*args, **kwargs)`` signature instead of the wrapped function's real
    one). Mirrors ``composeai.combinators._plain_callable_types``'s logic
    for a plain callable, duplicated rather than imported to avoid a
    tools<->combinators dependency (``combinators`` already imports
    ``tools``, not the other way around).
    """
    try:
        hints = resolve_annotations(fn, include_extras=True)
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return Any, Any
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    input_type: Any = hints.get(positional[0].name, Any) if positional else Any
    output_type: Any = hints.get("return", Any)
    return input_type, output_type


def _build_param_model(fn: Callable[..., Any], tool_name: str) -> type[BaseModel]:
    sig = inspect.signature(fn)
    hints = resolve_annotations(fn, include_extras=True)

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ConfigError(
                f"@tool function {fn.__name__!r} must not use *args/**kwargs: tool "
                "parameters must be explicit, typed keyword arguments so a JSON "
                "Schema can be generated for them."
            )
        annotation = hints.get(param_name, Any)
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (annotation, ...)
        else:
            fields[param_name] = (annotation, param.default)

    words = [word.capitalize() for word in re.split(r"\W+", tool_name) if word]
    model_name = "".join(words) + "Params"
    # `extra="forbid"`: the schema handed to the model always advertises
    # `additionalProperties: false` (via `seal_schema`) -- the local
    # re-validation `Tool.execute` does at call time must actually enforce
    # that too, not just pydantic's default `extra="ignore"` (which would
    # silently drop any keys not in the schema instead of rejecting the
    # call). A rejection here is still just a normal exception -- the agent
    # loop's existing `except Exception` handling in `_execute_one_tool`
    # already turns it into an `is_error` tool result, same as any other
    # validation failure.
    return create_model(  # type: ignore[call-overload, no-any-return]
        model_name, __config__=ConfigDict(extra="forbid"), **fields
    )


def _merge_arg_descriptions(
    fn_name: str, schema: dict[str, Any], descriptions: dict[str, str]
) -> None:
    """Merge each parsed ``Args:`` description into its matching schema property.

    Raises :class:`~composeai.errors.ConfigError` at decoration time if a
    docstring ``Args:`` entry doesn't match any real parameter name --
    e.g. the function was refactored (a parameter renamed) but the
    docstring wasn't updated. Previously this was silently dropped: the
    stale entry vanished with no error, and the real, now-undocumented
    parameter shipped with no description at all.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    unmatched = sorted(name for name in descriptions if name not in properties)
    if unmatched:
        raise ConfigError(
            f"@tool function {fn_name!r}: docstring Args: section describes "
            f"parameter(s) {unmatched!r}, which don't match any real parameter "
            f"(actual parameters: {sorted(properties)!r}) -- fix the typo, or "
            "update the docstring after a rename."
        )
    for arg_name, desc in descriptions.items():
        prop = properties.get(arg_name)
        if isinstance(prop, dict) and desc:
            prop["description"] = desc


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Parse a Google-style docstring into ``(description, {arg: description})``.

    A minimal stdlib parser: everything before a line reading exactly
    ``Args:`` is the description; within the ``Args:`` section, a line
    indented under it matching ``name: description`` starts a new argument,
    and any more-deeply-indented line that follows is a continuation of it.
    Parsing stops at the next zero-indent line (e.g. a ``Returns:`` section).
    """
    cleaned = inspect.cleandoc(doc)
    lines = cleaned.splitlines()

    args_start: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == "Args:":
            args_start = i
            break

    if args_start is None:
        return cleaned.strip(), {}

    description = "\n".join(lines[:args_start]).rstrip()

    descriptions: dict[str, str] = {}
    current_name: str | None = None
    current_indent: int | None = None

    for line in lines[args_start + 1 :]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent == 0:
            break  # a new top-level section, e.g. "Returns:"

        match = _ARG_LINE_RE.match(line)
        if match and (current_indent is None or indent <= current_indent):
            # Explicit `str` annotation (not just an `assert is not None`) so this
            # doesn't stay `str | Any` -- match.group()'s stub return type -- which
            # would otherwise make every later use of `current_name` re-widen back
            # to `str | None` instead of narrowing to `str`.
            matched_name: str = match.group("name")
            current_name = matched_name
            current_indent = indent
            descriptions[current_name] = match.group("desc").strip()
        elif current_name is not None:
            descriptions[current_name] = (descriptions[current_name] + " " + line.strip()).strip()

    return description, descriptions
