"""OpenAI Responses API adapter.

The ``openai`` SDK is imported only from within this module, and only
lazily (inside :meth:`OpenAIModel._get_client` / :meth:`OpenAIModel.complete`)
so that importing :mod:`composeai` -- or even this module, for a caller who
passes in their own ``client=`` -- never requires the SDK to be installed
unless it's actually used.

Wire shapes were verified against the installed ``openai`` SDK (2.45.0) --
``openai/types/responses/*.py`` -- rather than assumed from memory.

**Reasoning round-trip is limited to what the API returns by default.**
``_build_kwargs`` never sets the request-side ``reasoning`` parameter (no
``summary``/``generate_summary``), and never sets ``include:
["reasoning.encrypted_content"]`` -- both are opt-in per the installed SDK's
``types/shared_params/reasoning.py`` / ``types/responses/
response_create_params.py`` docstrings. Practically: a real reasoning-model
response's ``reasoning`` output item comes back with an empty ``summary``
list (no readable ``ThinkingPart.text``) and no ``encrypted_content``
(so :func:`_echo_reasoning_item`'s cross-turn round trip carries no actual
chain-of-thought payload back to the model -- it still correctly echoes
whatever *is* present, e.g. the item ``id``, just with an empty ``summary``).
Requesting a reasoning summary / encrypted content is a v1 roadmap item, not
implemented here -- see docs/design.md's "Out of scope v1" list -- since it
needs a new request-side configuration surface (this module's ``ModelRequest``
has no ``reasoning``-shaped field to plumb through yet, same gap as
Anthropic's extended-thinking ``thinking`` config -- see models/anthropic.py).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..errors import ConfigError, ProviderError
from ..messages import (
    ContentPart,
    ImagePart,
    Message,
    StopReason,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from .base import ModelRequest, ModelResponse, RawStreamEvent, ToolSpec
from .prices import compute_cost, get_price

# JSON schema response-format names must match a-z, A-Z, 0-9, `_`, `-`, up to
# 64 chars (see ResponseFormatTextJSONSchemaConfigParam in the SDK).
_SCHEMA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DEFAULT_SCHEMA_NAME = "response"

# Disqualifiers for OpenAI's strict-mode JSON Schema subset. Confirmed
# against the installed SDK's own `openai.lib._pydantic._ensure_strict_json_schema`
# (used by client.responses.parse()/client.beta.chat.completions.parse() to
# auto-convert a Pydantic model into a strict-compatible schema): it forces
# every object's `additionalProperties` to `False` and every property into
# `required`, which only makes sense if a schema-valued (non-`False`)
# `additionalProperties`, and a property missing from `required`, are both
# rejected by the real API in strict mode. `default` is confirmed by the
# same helper, which strips `default: None` before sending (a `default` key
# is not part of the strict subset at all -- composeai never strips it,
# since that would mean silently changing what the caller's schema means;
# instead a schema carrying one just doesn't get strict mode). The
# additional numeric/string/object constraint keywords and the restricted
# `format` allow-list below are OpenAI's separately documented "some
# type-specific keywords are not yet supported in strict mode" restrictions
# (platform.openai.com/docs/guides/structured-outputs#supported-schemas) --
# not encoded in the SDK's types, so erring toward the conservative
# (over-disqualifying) direction here is deliberate: the only cost of a
# false positive is an unnecessary `strict: false`, never a crash, whereas a
# false negative would still crash exactly like the bug this fixes.
_STRICT_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "patternProperties",
        "unevaluatedProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "unevaluatedItems",
        "contains",
        "minContains",
        "maxContains",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)
_STRICT_SUPPORTED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"}
)


def _is_strict_compatible(schema: Any) -> bool:
    """Recursively check whether ``schema`` qualifies for strict mode, without mutating it.

    Walks ``properties``/``items``/``anyOf``/``$defs``/etc. (any nested
    dict/list reachable from ``schema``) looking for a disqualifying
    keyword. Returns ``True`` (qualifies) unless a disqualifier is found
    anywhere in the tree -- callers use this to decide whether to send
    ``strict: True`` or ``strict: False``, never to alter the schema itself
    (see the module-level comment above ``_STRICT_UNSUPPORTED_KEYWORDS`` for
    why: mutating a caller's schema to "fix" it would silently change what
    it validates).
    """
    if isinstance(schema, list):
        return all(_is_strict_compatible(item) for item in schema)
    if not isinstance(schema, dict):
        return True

    if "default" in schema:
        return False
    if isinstance(schema.get("additionalProperties"), dict):
        return False
    if not _STRICT_UNSUPPORTED_KEYWORDS.isdisjoint(schema):
        return False
    fmt = schema.get("format")
    if isinstance(fmt, str) and fmt not in _STRICT_SUPPORTED_FORMATS:
        return False
    if schema.get("type") == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            required = schema.get("required")
            required_set = set(required) if isinstance(required, list) else set()
            if set(properties) != required_set:
                return False

    return all(_is_strict_compatible(value) for value in schema.values())


class OpenAIModel:
    """``Model`` adapter for OpenAI's Responses API (``client.responses.create``)."""

    def __init__(
        self,
        model_id: str,
        *,
        client: Any = None,
        api_key: str | None = None,
        max_retries: int = 2,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model_id = model_id
        self._client = client
        self._async_client: Any = None
        self._api_key = api_key
        self._max_retries = max_retries
        self._base_url = base_url
        self._timeout = timeout

        if client is not None:
            # An injected client is a SYNC client: the async twins would
            # silently build an unrelated AsyncX from env/api_key instead.
            # Shadowing the methods with None defeats the engine's getattr
            # discovery (models/base.py), so it falls back to running the
            # injected client's sync complete()/stream() off-thread --
            # exactly the 0.3.x behavior an injected client bought you.
            self.acomplete = None  # pyright: ignore[reportAttributeAccessIssue]
            self.astream = None  # pyright: ignore[reportAttributeAccessIssue]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import openai

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError(
                "Missing OpenAI API key: set the OPENAI_API_KEY environment "
                "variable, or pass api_key=... / client=... to OpenAIModel."
            )
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": self._max_retries,
            "base_url": self._base_url,
        }
        # Only pass timeout when set: the SDK's own default is a NOT_GIVEN
        # sentinel, and an explicit None would mean "no timeout at all".
        if self._timeout is not None:
            client_kwargs["timeout"] = self._timeout
        self._client = openai.OpenAI(**client_kwargs)
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is not None:
            return self._async_client
        import openai

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError(
                "Missing OpenAI API key: set the OPENAI_API_KEY environment "
                "variable, or pass api_key=... / client=... to OpenAIModel."
            )
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": self._max_retries,
            "base_url": self._base_url,
        }
        # Only pass timeout when set: the SDK's own default is a NOT_GIVEN
        # sentinel, and an explicit None would mean "no timeout at all".
        if self._timeout is not None:
            client_kwargs["timeout"] = self._timeout
        self._async_client = openai.AsyncOpenAI(**client_kwargs)
        return self._async_client

    def complete(self, request: ModelRequest) -> ModelResponse:
        import openai

        client = self._get_client()
        kwargs = self._build_kwargs(request)

        try:
            response = client.responses.create(**kwargs)
        except openai.APIError as exc:
            raise ProviderError(str(exc), provider="openai", model=self.model_id) from exc

        return self._finalize(response, request)

    async def acomplete(self, request: ModelRequest) -> ModelResponse:
        import openai

        client = self._get_async_client()
        kwargs = self._build_kwargs(request)

        try:
            response = await client.responses.create(**kwargs)
        except openai.APIError as exc:
            raise ProviderError(str(exc), provider="openai", model=self.model_id) from exc

        return self._finalize(response, request)

    def stream(self, request: ModelRequest) -> Iterator[RawStreamEvent]:
        """Stream a completion, yielding token/tool deltas then ``response_done``.

        Uses the SDK's ``client.responses.stream(...)`` helper, which yields
        the same well-known ``response.*`` events documented for
        ``create(stream=True)`` (``response.output_text.delta``,
        ``response.output_item.added``/``.done``,
        ``response.function_call_arguments.delta``,
        ``response.reasoning_summary_text.delta``, ``response.completed``,
        ...). The final ``response_done`` event carries a
        :class:`ModelResponse` built via the exact same finalization helper
        :meth:`complete` uses, so consuming only that event is equivalent to
        calling ``complete()``.
        """
        import openai

        client = self._get_client()
        kwargs = self._build_kwargs(request)

        final_response: Any = None
        # output_index -> the output item seen at response.output_item.added,
        # so later function_call_arguments.delta/output_item.done events
        # (which don't repeat the call_id/name) can be correlated back.
        item_by_index: dict[int, Any] = {}

        try:
            with client.responses.stream(**kwargs) as stream:
                for raw_event in stream:
                    event_type = getattr(raw_event, "type", None)
                    if event_type == "response.output_text.delta":
                        yield RawStreamEvent(kind="text_delta", text=raw_event.delta)
                    elif event_type == "response.reasoning_summary_text.delta":
                        yield RawStreamEvent(kind="thinking_delta", text=raw_event.delta)
                    elif event_type == "response.output_item.added":
                        item = raw_event.item
                        item_by_index[raw_event.output_index] = item
                        if getattr(item, "type", None) == "function_call":
                            yield RawStreamEvent(
                                kind="tool_call_started",
                                tool_call_id=item.call_id,
                                tool_name=item.name,
                            )
                    elif event_type == "response.function_call_arguments.delta":
                        item = item_by_index.get(raw_event.output_index)
                        yield RawStreamEvent(
                            kind="tool_args_delta",
                            text=raw_event.delta,
                            tool_call_id=getattr(item, "call_id", None),
                            tool_name=getattr(item, "name", None),
                        )
                    elif event_type == "response.output_item.done":
                        item = raw_event.item
                        if getattr(item, "type", None) == "function_call":
                            yield RawStreamEvent(
                                kind="tool_call_finished",
                                tool_call_id=item.call_id,
                                tool_name=item.name,
                            )
                    elif event_type in (
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    ):
                        # response.incomplete/response.failed both carry a
                        # `.response` field of the exact same shape as
                        # response.completed's (verified against the
                        # installed SDK's ResponseIncompleteEvent /
                        # ResponseFailedEvent types) -- a response that
                        # legitimately hit max_output_tokens, or failed
                        # server-side, ends the stream via one of these
                        # instead of response.completed, and must be mapped
                        # the same way complete() maps the equivalent
                        # non-streamed response (via _stop_reason's
                        # status=="incomplete" handling, or _finalize's
                        # status=="failed" check below) rather than falling
                        # through to the "stream ended early" ProviderError.
                        final_response = raw_event.response
                    # Every other event type (response.created,
                    # .in_progress, content_part.*, audio.*, web_search.*,
                    # ...) carries nothing new here.
        except openai.APIError as exc:
            raise ProviderError(str(exc), provider="openai", model=self.model_id) from exc

        if final_response is None:
            raise ProviderError(
                "OpenAI's response stream ended without a response.completed "
                "event -- the stream was closed or interrupted before finishing",
                provider="openai",
                model=self.model_id,
            )

        response = self._finalize(final_response, request)
        yield RawStreamEvent(kind="response_done", response=response)

    async def astream(self, request: ModelRequest) -> AsyncIterator[RawStreamEvent]:
        """Async twin of :meth:`stream`; see its docstring for behavior."""
        import openai

        client = self._get_async_client()
        kwargs = self._build_kwargs(request)

        final_response: Any = None
        # output_index -> the output item seen at response.output_item.added,
        # so later function_call_arguments.delta/output_item.done events
        # (which don't repeat the call_id/name) can be correlated back.
        item_by_index: dict[int, Any] = {}

        try:
            async with client.responses.stream(**kwargs) as stream:
                async for raw_event in stream:
                    event_type = getattr(raw_event, "type", None)
                    if event_type == "response.output_text.delta":
                        yield RawStreamEvent(kind="text_delta", text=raw_event.delta)
                    elif event_type == "response.reasoning_summary_text.delta":
                        yield RawStreamEvent(kind="thinking_delta", text=raw_event.delta)
                    elif event_type == "response.output_item.added":
                        item = raw_event.item
                        item_by_index[raw_event.output_index] = item
                        if getattr(item, "type", None) == "function_call":
                            yield RawStreamEvent(
                                kind="tool_call_started",
                                tool_call_id=item.call_id,
                                tool_name=item.name,
                            )
                    elif event_type == "response.function_call_arguments.delta":
                        item = item_by_index.get(raw_event.output_index)
                        yield RawStreamEvent(
                            kind="tool_args_delta",
                            text=raw_event.delta,
                            tool_call_id=getattr(item, "call_id", None),
                            tool_name=getattr(item, "name", None),
                        )
                    elif event_type == "response.output_item.done":
                        item = raw_event.item
                        if getattr(item, "type", None) == "function_call":
                            yield RawStreamEvent(
                                kind="tool_call_finished",
                                tool_call_id=item.call_id,
                                tool_name=item.name,
                            )
                    elif event_type in (
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    ):
                        # response.incomplete/response.failed both carry a
                        # `.response` field of the exact same shape as
                        # response.completed's (verified against the
                        # installed SDK's ResponseIncompleteEvent /
                        # ResponseFailedEvent types) -- a response that
                        # legitimately hit max_output_tokens, or failed
                        # server-side, ends the stream via one of these
                        # instead of response.completed, and must be mapped
                        # the same way complete() maps the equivalent
                        # non-streamed response (via _stop_reason's
                        # status=="incomplete" handling, or _finalize's
                        # status=="failed" check below) rather than falling
                        # through to the "stream ended early" ProviderError.
                        final_response = raw_event.response
                    # Every other event type (response.created,
                    # .in_progress, content_part.*, audio.*, web_search.*,
                    # ...) carries nothing new here.
        except openai.APIError as exc:
            raise ProviderError(str(exc), provider="openai", model=self.model_id) from exc

        if final_response is None:
            raise ProviderError(
                "OpenAI's response stream ended without a response.completed "
                "event -- the stream was closed or interrupted before finishing",
                provider="openai",
                model=self.model_id,
            )

        response = self._finalize(final_response, request)
        yield RawStreamEvent(kind="response_done", response=response)

    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        api_input = _build_input(request.messages, self.model_id)
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "input": api_input,
            "max_output_tokens": request.max_tokens,
        }
        if request.system is not None:
            kwargs["instructions"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = [_tool_to_param(spec) for spec in request.tools]
        if request.output_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": _schema_name(request.output_schema),
                    "schema": request.output_schema,
                    "strict": _is_strict_compatible(request.output_schema),
                }
            }
        return kwargs

    def _finalize(self, response: Any, request: ModelRequest) -> ModelResponse:
        # A response with status=="failed" is a normal 200-level API
        # response (the SDK does not raise for it -- only a transport-level/
        # HTTP-level failure raises openai.APIError), so it must be handled
        # here rather than relying on the try/except around
        # client.responses.create()/.stream(). Shared by complete() and
        # stream() (stream() now forwards response.failed's `.response` here
        # the same as response.completed's -- see stream()'s event loop) so
        # both paths get the same failure handling.
        if getattr(response, "status", None) == "failed":
            error = getattr(response, "error", None)
            code = getattr(error, "code", None) if error is not None else None
            message = getattr(error, "message", None) if error is not None else None
            detail = (
                f"{code}: {message}" if code and message else (message or code or "unknown error")
            )
            raise ProviderError(
                f"OpenAI response failed: {detail}",
                provider="openai",
                model=self.model_id,
            )

        had_function_call = False
        had_refusal = False
        parts: list[ContentPart] = []
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                had_function_call = True
            for part, is_refusal in _item_to_parts(item, self.model_id):
                parts.append(part)
                had_refusal = had_refusal or is_refusal

        message = Message(role="assistant", parts=parts)
        stop_reason, raw_stop_reason = _stop_reason(response, had_function_call, had_refusal)
        usage = _map_usage(getattr(response, "usage", None), self.model_id)

        # Only attempt to decode structured output on a clean END_TURN: a
        # function-call-only turn legitimately has no output_text item at
        # all (Structured Outputs + function calling is an explicitly
        # supported combination in the real Responses API), and
        # REFUSAL/MAX_TOKENS/OTHER already have their own dedicated handling
        # one level up in agentfn.py's turn loop that a spurious JSON-parse
        # ProviderError would only mask. agentfn.py's _extract_output (the
        # only reader of `.parsed`) is likewise only ever called on the
        # END_TURN branch, so `None` here for every other stop reason is
        # exactly what it expects.
        parsed: dict[str, Any] | None = None
        if request.output_schema is not None and stop_reason == StopReason.END_TURN:
            try:
                parsed = json.loads(message.text)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    "OpenAI returned non-JSON text for a structured-output "
                    f"request: {exc}",
                    provider="openai",
                    model=self.model_id,
                ) from exc

        return ModelResponse(
            message=message,
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
            usage=usage,
            model_id=self.model_id,
            parsed=parsed,
        )


def create_model(model_id: str) -> OpenAIModel:
    """Factory used by :mod:`composeai.models.registry` for ``"openai/..."`` strings."""
    return OpenAIModel(model_id)


# --- request-side mapping ----------------------------------------------------


def _build_input(messages: list[Message], model_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for msg in messages:
        buffer: list[dict[str, Any]] = []
        for part in msg.parts:
            if isinstance(part, TextPart):
                buffer.append({"type": "input_text", "text": part.text})
            elif isinstance(part, ImagePart):
                buffer.append(_image_part_to_content(part))
            elif isinstance(part, ToolCallPart):
                _flush(items, msg.role, buffer)
                items.append(_tool_call_part_to_item(part))
            elif isinstance(part, ToolResultPart):
                _flush(items, msg.role, buffer)
                items.append(_tool_result_part_to_item(part))
            elif isinstance(part, ThinkingPart):
                _flush(items, msg.role, buffer)
                echoed = _echo_reasoning_item(part, model_id)
                if echoed is not None:
                    items.append(echoed)
            else:
                unhandled = f"unhandled ContentPart type: {type(part)!r}"
                raise AssertionError(unhandled)  # pragma: no cover
        _flush(items, msg.role, buffer)
    return items


def _flush(items: list[dict[str, Any]], role: str, buffer: list[dict[str, Any]]) -> None:
    if buffer:
        items.append({"role": role, "content": list(buffer)})
        buffer.clear()


def _image_part_to_content(part: ImagePart) -> dict[str, Any]:
    if part.data is not None:
        image_url = f"data:{part.media_type};base64,{part.data}"
    else:
        image_url = part.url
    return {"type": "input_image", "image_url": image_url, "detail": "auto"}


def _tool_call_part_to_item(part: ToolCallPart) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": part.id,
        "name": part.name,
        "arguments": json.dumps(part.arguments),
    }


def _tool_result_part_to_item(part: ToolResultPart) -> dict[str, Any]:
    # function_call_output has no error flag in the Responses API -- verified
    # against ResponseInputItemParam.FunctionCallOutput -- so we fold
    # is_error into the text the same way the brief calls for.
    output = part.content if not part.is_error else f"ERROR: {part.content}"
    return {"type": "function_call_output", "call_id": part.tool_call_id, "output": output}


def _echo_reasoning_item(part: ThinkingPart, model_id: str) -> dict[str, Any] | None:
    # Only echo a reasoning item back to the same provider + model it came
    # from; other-model/other-provider reasoning is dropped silently.
    if part.provider != "openai" or part.provider_token is None:
        return None
    # A corrupted/hand-edited provider_token (e.g. from a resumed
    # agent_state or a hand-edited cassette) must not surface a raw
    # json.JSONDecodeError/KeyError -- every error composeai raises is a
    # ComposeError (errors.py's module docstring). Mirrors anthropic.py's
    # `_part_to_block` thinking-echo path, which wraps the equivalent
    # json.loads the same way.
    try:
        payload = json.loads(part.provider_token)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"malformed reasoning provider_token for OpenAI model {model_id!r}: {exc}",
            provider="openai",
            model=model_id,
        ) from exc
    if payload.get("model") != model_id:
        return None
    try:
        return payload["item"]
    except KeyError as exc:
        raise ProviderError(
            f"malformed reasoning provider_token for OpenAI model {model_id!r}: "
            f"missing {exc} key",
            provider="openai",
            model=model_id,
        ) from exc


def _tool_to_param(spec: ToolSpec) -> dict[str, Any]:
    # Responses API function tools are flat (unlike Chat Completions, which
    # nests under a "function" key) -- verified against FunctionToolParam.
    # `spec.strict` is always True (tools.py hardcodes it), but the schema
    # it was built from routinely violates OpenAI's strict-mode subset (a
    # defaulted parameter is both missing from `required` and carries a
    # `default` key -- see test_tools.py's
    # test_schema_required_param_has_no_default) -- downgrade to
    # `strict: False` rather than sending a schema OpenAI's strict mode
    # would reject outright; never mutate `spec.input_schema` itself.
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.input_schema,
        "strict": spec.strict and _is_strict_compatible(spec.input_schema),
    }


def _schema_name(schema: dict[str, Any]) -> str:
    title = schema.get("title")
    if isinstance(title, str) and _SCHEMA_NAME_RE.match(title):
        return title
    return _DEFAULT_SCHEMA_NAME


# --- response-side mapping ----------------------------------------------------


def _item_to_parts(item: Any, model_id: str) -> list[tuple[ContentPart, bool]]:
    """Map one Responses API output item to zero or more ``(part, is_refusal)`` pairs."""
    item_type = getattr(item, "type", None)
    if item_type == "message":
        result: list[tuple[ContentPart, bool]] = []
        for content in getattr(item, "content", None) or []:
            content_type = getattr(content, "type", None)
            if content_type == "output_text":
                result.append((TextPart(text=content.text), False))
            elif content_type == "refusal":
                result.append((TextPart(text=content.refusal), True))
            # Other content sub-types (logprobs-only, future additions) are
            # dropped rather than raising.
        return result
    if item_type == "function_call":
        try:
            arguments = json.loads(item.arguments)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"OpenAI returned non-JSON arguments for function_call {item.name!r}: {exc}",
                provider="openai",
                model=model_id,
            ) from exc
        return [(ToolCallPart(id=item.call_id, name=item.name, arguments=arguments), False)]
    if item_type == "reasoning":
        return [(_reasoning_item_to_thinking_part(item, model_id), False)]
    # Unknown/unsupported item types (web search calls, computer calls, mcp
    # calls, future additions, ...) are dropped rather than raising.
    return []


def _reasoning_item_to_thinking_part(item: Any, model_id: str) -> ThinkingPart:
    summary = [
        {"type": "summary_text", "text": getattr(s, "text", "") or ""}
        for s in getattr(item, "summary", None) or []
    ]
    raw_item: dict[str, Any] = {"type": "reasoning", "id": item.id, "summary": summary}

    content = getattr(item, "content", None)
    if content:
        raw_item["content"] = [
            {"type": "reasoning_text", "text": getattr(c, "text", "") or ""} for c in content
        ]

    encrypted = getattr(item, "encrypted_content", None)
    if encrypted is not None:
        raw_item["encrypted_content"] = encrypted

    text = "".join(s["text"] for s in summary)
    if not text and "content" in raw_item:
        text = "".join(c["text"] for c in raw_item["content"])

    token = json.dumps({"item": raw_item, "model": model_id})
    return ThinkingPart(text=text, provider="openai", provider_token=token)


def _stop_reason(
    response: Any, had_function_call: bool, had_refusal: bool
) -> tuple[StopReason, str | None]:
    status = getattr(response, "status", None)
    incomplete_details = getattr(response, "incomplete_details", None)
    incomplete_reason = getattr(incomplete_details, "reason", None) if incomplete_details else None

    if status == "incomplete":
        raw_stop_reason = f"incomplete:{incomplete_reason}" if incomplete_reason else "incomplete"
    else:
        raw_stop_reason = status

    if had_function_call:
        return StopReason.TOOL_USE, raw_stop_reason
    if status == "incomplete" and incomplete_reason == "max_output_tokens":
        return StopReason.MAX_TOKENS, raw_stop_reason
    if had_refusal:
        return StopReason.REFUSAL, raw_stop_reason
    if status == "completed":
        return StopReason.END_TURN, raw_stop_reason
    return StopReason.OTHER, raw_stop_reason


def _map_usage(usage: Any, model_id: str) -> Usage:
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    input_details = getattr(usage, "input_tokens_details", None)
    cache_read_tokens = (getattr(input_details, "cached_tokens", 0) or 0) if input_details else 0

    output_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = (
        (getattr(output_details, "reasoning_tokens", 0) or 0) if output_details else 0
    )

    price = get_price("openai", model_id)
    if price is None:
        cost_usd = None
        cost_complete = False
    else:
        # OpenAI doesn't bill separately for cache *writes* (only cached
        # *reads* get a discount), so cache_creation_tokens stays 0 and no
        # cache-write cost is ever added here -- per the brief.
        #
        # `input_tokens` INCLUDES `cached_tokens` -- verified against the
        # installed SDK's `response_usage.InputTokensDetails` docstring ("a
        # detailed breakdown of the input tokens", i.e. cached_tokens is a
        # subset, not additive) and confirmed unambiguously by the
        # structurally identical `openai.types.realtime.
        # RealtimeResponseUsageInputTokenDetails` docstring: "Cached tokens
        # here are counted as a subset of input tokens, meaning input
        # tokens will include cached and uncached tokens." Billing the full
        # `input_tokens` at the input rate AND separately billing
        # `cache_read_tokens` at the cache_read rate would double-bill the
        # cached slice -- subtract it out of the base bucket first.
        billable_input_tokens = input_tokens - cache_read_tokens
        cost_usd = compute_cost(
            Usage(
                input_tokens=billable_input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
            ),
            price,
            0,
            0,
        )
        cost_complete = True

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=0,
        reasoning_tokens=reasoning_tokens,
        cost_usd=cost_usd,
        cost_complete=cost_complete,
    )
