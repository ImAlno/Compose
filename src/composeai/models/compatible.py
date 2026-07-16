"""OpenAI-compatible Chat Completions adapter (Ollama, vLLM, and similar servers).

The ``openai`` SDK is imported only from within this module, and only
lazily (inside :meth:`OpenAICompatibleModel._get_client` /
:meth:`OpenAICompatibleModel.complete`), same as :mod:`composeai.models.openai`.

Unlike the OpenAI provider (registered via ``"openai/model-id"`` strings),
there is no string form for this adapter -- callers must build a model via
the explicit :func:`openai_compatible` factory (or the
:class:`OpenAICompatibleModel` class directly) and pass the resulting
``Model`` instance to ``@agent(model=...)`` / ``registry.resolve()``, which
already accepts ``Model`` instances as a passthrough.

Wire shapes were verified against the installed ``openai`` SDK (2.45.0) --
``openai/types/chat/*.py`` -- rather than assumed from memory.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
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
from .base import Model, ModelRequest, ModelResponse, RawStreamEvent, ToolSpec
from .prices import ModelPrice, compute_cost, get_price, register_price

_PROVIDER = "openai-compatible"


class _NonJsonContentError(ProviderError):
    """The generic 'accepted the request, returned non-JSON' failure -- the
    ONLY condition schema_mode="auto" demotes on. Module-private: external
    behavior (isinstance ProviderError, message text) is unchanged; the
    subclass exists so complete() can distinguish it from transport errors
    and the reasoning-tokens diagnostic without string-matching messages.
    """


_SCHEMA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DEFAULT_SCHEMA_NAME = "response"

_FINISH_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
    "content_filter": StopReason.REFUSAL,
    # "function_call" (deprecated) and anything unrecognized fall through to
    # OTHER via the .get(..., StopReason.OTHER) lookup below.
}

# Disqualifiers for OpenAI's strict-mode JSON Schema subset -- this adapter
# speaks Chat Completions rather than the Responses API, but a
# strict-mode-enforcing server (vLLM guided decoding, or a real OpenAI
# endpoint reached via base_url) applies the same JSON Schema subset either
# way, so the same check applies. Deliberately duplicated from
# models/openai.py (see tools.py's `_tool_types` for the established
# precedent of this project duplicating a small pure helper across modules
# rather than introducing a cross-adapter import) -- see that module's
# comment above its copy of these constants for what's verified against the
# installed SDK vs. OpenAI's separately-documented strict-mode restrictions.
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

    See models/openai.py's identical function for the full rationale.
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


class OpenAICompatibleModel:
    """``Model`` adapter for any OpenAI-compatible Chat Completions server.

    ``schema_mode="prompt"`` embeds the output schema in the prompt and
    parses the reply leniently instead of relying on ``response_format``
    enforcement. When the reply still doesn't parse, this adapter does
    *not* raise a provider-level error: it returns the response with
    ``parsed=None``, and the agent loop's output extraction raises a
    repairable ``ComposeError`` -- eligible for ``@agent(max_repairs=)``
    corrective turns, not ``retries=`` (which stays provider-error
    territory, e.g. a dropped connection).

    ``schema_mode="auto"`` starts native and demotes permanently to prompt
    mode on the first generic non-JSON structured reply (the server accepted
    ``response_format`` but returned prose anyway) -- that one ``complete()``
    call is retried in prompt mode, and every subsequent call on this
    instance goes straight to prompt mode for the rest of the process, so at
    most one native call is ever wasted per instance. ``stream()`` follows
    the current effective mode but never itself demotes -- a mid-stream
    retry would re-emit already-yielded deltas -- so a streaming caller that
    wants auto-demotion benefits should route at least one ``complete()``
    call through the same instance first. Transport errors (``openai.
    APIError``) and the reasoning-tokens-only diagnostic never demote and
    raise in every mode, including ``"auto"``.
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        client: Any = None,
        api_key: str | None = None,
        max_retries: int = 2,
        timeout: float | None = None,
        schema_mode: str = "native",
    ) -> None:
        if schema_mode not in ("native", "prompt", "auto"):
            raise ConfigError(
                f"schema_mode must be 'native', 'prompt', or 'auto', got {schema_mode!r}"
            )
        self.model_id = model_id
        self._base_url = base_url
        self._client = client
        self._async_client: Any = None
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout
        self._schema_mode = schema_mode
        self._demoted = False

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

        # Many local servers (Ollama, vLLM, ...) require a non-empty key
        # even though they don't check it -- default to a placeholder
        # rather than requiring one like the hosted OpenAI/Anthropic adapters.
        api_key = self._api_key or "unused"
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self._base_url,
            "max_retries": self._max_retries,
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

        # Many local servers (Ollama, vLLM, ...) require a non-empty key
        # even though they don't check it -- default to a placeholder
        # rather than requiring one like the hosted OpenAI/Anthropic adapters.
        api_key = self._api_key or "unused"
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self._base_url,
            "max_retries": self._max_retries,
        }
        # Only pass timeout when set: the SDK's own default is a NOT_GIVEN
        # sentinel, and an explicit None would mean "no timeout at all".
        if self._timeout is not None:
            client_kwargs["timeout"] = self._timeout
        self._async_client = openai.AsyncOpenAI(**client_kwargs)
        return self._async_client

    def _effective_schema_mode(self) -> str:
        # A benign race if two threads demote at once (worst case: one
        # extra native attempt) -- deliberately unsynchronized.
        if self._schema_mode == "auto":
            return "prompt" if self._demoted else "native"
        return self._schema_mode

    def complete(self, request: ModelRequest) -> ModelResponse:
        import openai

        client = self._get_client()
        kwargs = self._build_kwargs(request)

        try:
            response = client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise ProviderError(str(exc), provider=_PROVIDER, model=self.model_id) from exc

        choice = response.choices[0]
        usage = _map_usage(getattr(response, "usage", None), self.model_id)
        try:
            return self._finalize(choice.message, choice.finish_reason, usage, request)
        except _NonJsonContentError:
            if self._schema_mode != "auto" or self._demoted:
                raise
            # The server accepted response_format but returned prose: demote
            # this instance permanently and retry this one call in prompt
            # mode. At most one wasted native call per instance per process.
            self._demoted = True
            # The first (native) call still happened and was real, billed
            # spend -- its usage must not simply be discarded, or trace/costs/
            # Budget would silently under-count one full completion. Keep it
            # and sum it into the retry's usage before finalizing.
            wasted_usage = usage
            kwargs = self._build_kwargs(request)
            try:
                response = client.chat.completions.create(**kwargs)
            except openai.APIError as exc:
                raise ProviderError(str(exc), provider=_PROVIDER, model=self.model_id) from exc
            choice = response.choices[0]
            usage = _map_usage(getattr(response, "usage", None), self.model_id)
            return self._finalize(
                choice.message, choice.finish_reason, wasted_usage + usage, request
            )

    async def acomplete(self, request: ModelRequest) -> ModelResponse:
        import openai

        client = self._get_async_client()
        kwargs = self._build_kwargs(request)

        try:
            response = await client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise ProviderError(str(exc), provider=_PROVIDER, model=self.model_id) from exc

        choice = response.choices[0]
        usage = _map_usage(getattr(response, "usage", None), self.model_id)
        try:
            return self._finalize(choice.message, choice.finish_reason, usage, request)
        except _NonJsonContentError:
            if self._schema_mode != "auto" or self._demoted:
                raise
            # The server accepted response_format but returned prose: demote
            # this instance permanently and retry this one call in prompt
            # mode. At most one wasted native call per instance per process.
            self._demoted = True
            # The first (native) call still happened and was real, billed
            # spend -- its usage must not simply be discarded, or trace/costs/
            # Budget would silently under-count one full completion. Keep it
            # and sum it into the retry's usage before finalizing.
            wasted_usage = usage
            kwargs = self._build_kwargs(request)
            try:
                response = await client.chat.completions.create(**kwargs)
            except openai.APIError as exc:
                raise ProviderError(str(exc), provider=_PROVIDER, model=self.model_id) from exc
            choice = response.choices[0]
            usage = _map_usage(getattr(response, "usage", None), self.model_id)
            return self._finalize(
                choice.message, choice.finish_reason, wasted_usage + usage, request
            )

    def stream(self, request: ModelRequest) -> Iterator[RawStreamEvent]:
        """Stream a completion, yielding token/tool deltas then ``response_done``.

        Uses ``chat.completions.create(stream=True, stream_options=
        {"include_usage": True}, ...)``. Token usage is only sent by the
        server on a final, choice-less chunk when it honors
        ``include_usage``; servers that omit it (some local/compatible
        servers do) degrade to a zero usage with ``cost_complete=False``
        rather than fabricating a ``$0.00`` cost. The final ``response_done``
        event carries a :class:`ModelResponse` built via the exact same
        finalization helper :meth:`complete` uses (replaying the accumulated
        deltas through a synthetic message object with the same shape
        ``complete()``'s SDK response would have), so consuming only that
        event is equivalent to calling ``complete()``.
        """
        import openai

        client = self._get_client()
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        content_chunks: list[str] = []
        refusal_chunks: list[str] = []
        # tool-call index -> accumulated {"id", "name", "arguments"}; dicts
        # preserve insertion order, matching the order calls first appeared.
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage_obj: Any = None
        finished_emitted = False

        try:
            # `create(stream=True)` returns the SDK's own `Stream`, which is
            # itself a context manager (releases the underlying HTTP
            # connection on `__exit__`/`close()`) -- use it as one rather
            # than just iterating over it, same as the Anthropic adapter's
            # `with client.messages.stream(**kwargs) as stream:`. (The SDK
            # also offers a *separate*, higher-level `chat.completions.
            # stream()` helper with its own accumulation/event API -- not
            # used here: it yields structured events in a different shape
            # than the raw `ChatCompletionChunk`s this module parses against
            # verified wire shapes, so adopting it would mean rewriting this
            # method's event mapping wholesale for no behavioral gain.)
            with client.chat.completions.create(**kwargs) as stream:
                for chunk in stream:
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        choice = choices[0]
                        delta = choice.delta
                        if delta.content:
                            content_chunks.append(delta.content)
                            yield RawStreamEvent(kind="text_delta", text=delta.content)
                        if getattr(delta, "refusal", None):
                            refusal_chunks.append(delta.refusal)
                        for tc_delta in delta.tool_calls or []:
                            index = tc_delta.index
                            function = tc_delta.function
                            if index not in tool_calls:
                                name = function.name if function is not None else None
                                tool_calls[index] = {
                                    "id": tc_delta.id,
                                    "name": name,
                                    "arguments": "",
                                }
                                yield RawStreamEvent(
                                    kind="tool_call_started",
                                    tool_call_id=tc_delta.id,
                                    tool_name=name,
                                )
                            fragment = function.arguments if function is not None else None
                            if fragment:
                                tool_calls[index]["arguments"] += fragment
                                yield RawStreamEvent(
                                    kind="tool_args_delta",
                                    text=fragment,
                                    tool_call_id=tool_calls[index]["id"],
                                    tool_name=tool_calls[index]["name"],
                                )
                        if choice.finish_reason is not None:
                            finish_reason = choice.finish_reason
                            if not finished_emitted:
                                finished_emitted = True
                                for state in tool_calls.values():
                                    yield RawStreamEvent(
                                        kind="tool_call_finished",
                                        tool_call_id=state["id"],
                                        tool_name=state["name"],
                                    )
                    if getattr(chunk, "usage", None) is not None:
                        usage_obj = chunk.usage
        except openai.APIError as exc:
            raise ProviderError(str(exc), provider=_PROVIDER, model=self.model_id) from exc

        fake_tool_calls = [
            SimpleNamespace(
                id=state["id"],
                type="function",
                function=SimpleNamespace(name=state["name"], arguments=state["arguments"]),
            )
            for state in tool_calls.values()
        ] or None
        fake_message = SimpleNamespace(
            content="".join(content_chunks) or None,
            refusal="".join(refusal_chunks) or None,
            tool_calls=fake_tool_calls,
        )
        # A missing usage chunk means the server didn't honor include_usage --
        # unknown, not zero, cost -- so cost_complete is forced False here
        # rather than going through _map_usage(None, ...), which would
        # otherwise compute a misleadingly-precise $0.00 for priced models.
        usage = (
            _map_usage(usage_obj, self.model_id)
            if usage_obj is not None
            else Usage(cost_usd=None, cost_complete=False)
        )
        response = self._finalize(fake_message, finish_reason, usage, request)
        yield RawStreamEvent(kind="response_done", response=response)

    async def astream(self, request: ModelRequest) -> AsyncIterator[RawStreamEvent]:
        """Async twin of :meth:`stream`; see its docstring for behavior."""
        import openai

        client = self._get_async_client()
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        content_chunks: list[str] = []
        refusal_chunks: list[str] = []
        # tool-call index -> accumulated {"id", "name", "arguments"}; dicts
        # preserve insertion order, matching the order calls first appeared.
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage_obj: Any = None
        finished_emitted = False

        try:
            # `create(stream=True)` returns the SDK's own `AsyncStream`, which
            # is itself an async context manager (releases the underlying
            # HTTP connection on `__aexit__`/`aclose()`) -- use it as one
            # rather than just iterating over it, same as stream()'s `with
            # client.chat.completions.create(**kwargs) as stream:`. (The SDK
            # also offers a *separate*, higher-level `chat.completions.
            # stream()` helper with its own accumulation/event API -- not
            # used here: it yields structured events in a different shape
            # than the raw `ChatCompletionChunk`s this module parses against
            # verified wire shapes, so adopting it would mean rewriting this
            # method's event mapping wholesale for no behavioral gain.)
            async with await client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        choice = choices[0]
                        delta = choice.delta
                        if delta.content:
                            content_chunks.append(delta.content)
                            yield RawStreamEvent(kind="text_delta", text=delta.content)
                        if getattr(delta, "refusal", None):
                            refusal_chunks.append(delta.refusal)
                        for tc_delta in delta.tool_calls or []:
                            index = tc_delta.index
                            function = tc_delta.function
                            if index not in tool_calls:
                                name = function.name if function is not None else None
                                tool_calls[index] = {
                                    "id": tc_delta.id,
                                    "name": name,
                                    "arguments": "",
                                }
                                yield RawStreamEvent(
                                    kind="tool_call_started",
                                    tool_call_id=tc_delta.id,
                                    tool_name=name,
                                )
                            fragment = function.arguments if function is not None else None
                            if fragment:
                                tool_calls[index]["arguments"] += fragment
                                yield RawStreamEvent(
                                    kind="tool_args_delta",
                                    text=fragment,
                                    tool_call_id=tool_calls[index]["id"],
                                    tool_name=tool_calls[index]["name"],
                                )
                        if choice.finish_reason is not None:
                            finish_reason = choice.finish_reason
                            if not finished_emitted:
                                finished_emitted = True
                                for state in tool_calls.values():
                                    yield RawStreamEvent(
                                        kind="tool_call_finished",
                                        tool_call_id=state["id"],
                                        tool_name=state["name"],
                                    )
                    if getattr(chunk, "usage", None) is not None:
                        usage_obj = chunk.usage
        except openai.APIError as exc:
            raise ProviderError(str(exc), provider=_PROVIDER, model=self.model_id) from exc

        fake_tool_calls = [
            SimpleNamespace(
                id=state["id"],
                type="function",
                function=SimpleNamespace(name=state["name"], arguments=state["arguments"]),
            )
            for state in tool_calls.values()
        ] or None
        fake_message = SimpleNamespace(
            content="".join(content_chunks) or None,
            refusal="".join(refusal_chunks) or None,
            tool_calls=fake_tool_calls,
        )
        # A missing usage chunk means the server didn't honor include_usage --
        # unknown, not zero, cost -- so cost_complete is forced False here
        # rather than going through _map_usage(None, ...), which would
        # otherwise compute a misleadingly-precise $0.00 for priced models.
        usage = (
            _map_usage(usage_obj, self.model_id)
            if usage_obj is not None
            else Usage(cost_usd=None, cost_complete=False)
        )
        response = self._finalize(fake_message, finish_reason, usage, request)
        yield RawStreamEvent(kind="response_done", response=response)

    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        api_messages = _build_messages(request.messages, request.system)
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": api_messages,
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = [_tool_to_param(spec) for spec in request.tools]
        if request.output_schema is not None:
            if self._effective_schema_mode() == "prompt":
                # Servers that accept response_format but silently ignore it
                # (Ollama cloud, some vLLM/LM Studio configs) get the schema
                # embedded in the prompt instead; _finalize parses leniently.
                # Appended to the last USER message -- a second system
                # message proved less reliable against long agent system
                # prompts in live testing.
                _append_schema_instruction(api_messages, request.output_schema)
            else:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": _schema_name(request.output_schema),
                        "schema": request.output_schema,
                        "strict": _is_strict_compatible(request.output_schema),
                    },
                }
        return kwargs

    def _finalize(
        self,
        message_obj: Any,
        finish_reason: str | None,
        usage: Usage,
        request: ModelRequest,
    ) -> ModelResponse:
        parts, had_refusal = _message_to_parts(message_obj, self.model_id)
        message = Message(role="assistant", parts=parts)
        # A genuine refusal (message.refusal set) takes priority over the
        # finish_reason table: servers typically report finish_reason=="stop"
        # for a refusal (Chat Completions has no dedicated refusal finish
        # reason the way it has "content_filter"), so relying on the table
        # alone would silently treat the refusal as a normal successful
        # answer -- see _message_to_parts's docstring.
        if had_refusal:
            stop_reason = StopReason.REFUSAL
        else:
            stop_reason = (
                _FINISH_REASON_MAP.get(finish_reason, StopReason.OTHER)
                if finish_reason is not None
                else StopReason.OTHER
            )

        # Only attempt to decode structured output on a clean END_TURN: a
        # tool_calls-only completion legitimately has message.content == None
        # (message.text == ''), and MAX_TOKENS/REFUSAL/OTHER already have
        # their own dedicated handling one level up in agentfn.py's turn
        # loop that a spurious JSON-parse ProviderError would only mask.
        # agentfn.py's _extract_output (the only reader of `.parsed`) is
        # likewise only ever called on the END_TURN branch, so `None` here
        # for every other stop reason is exactly what it expects.
        parsed: dict[str, Any] | None = None
        if request.output_schema is not None and stop_reason == StopReason.END_TURN:
            text = message.text
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                effective_mode = self._effective_schema_mode()
                if effective_mode == "prompt":
                    parsed = _lenient_json_parse(text)
                if parsed is None:
                    if usage.reasoning_tokens and not text.strip():
                        raise ProviderError(
                            "Model returned only reasoning tokens with empty "
                            "content for a structured-output request -- the "
                            "server likely does not enforce constrained decoding "
                            "for this model; try openai_compatible(..., "
                            "schema_mode='prompt') or raise max_tokens.",
                            provider=_PROVIDER,
                            model=self.model_id,
                        ) from exc
                    if effective_mode != "prompt":
                        raise _NonJsonContentError(
                            "Server returned non-JSON content for a structured-output "
                            f"request: {exc}",
                            provider=_PROVIDER,
                            model=self.model_id,
                        ) from exc
                    # Prompt mode: leave parsed=None and fall through instead of
                    # raising -- ModelResponse(parsed=None) reaches agentfn.py's
                    # _extract_output, which retries json.loads(message.text) and
                    # raises a *repairable* ComposeError, letting
                    # @agent(max_repairs=) recover with a corrective turn instead
                    # of the run failing outright on a single malformed reply.

        return ModelResponse(
            message=message,
            stop_reason=stop_reason,
            raw_stop_reason=finish_reason,
            usage=usage,
            model_id=self.model_id,
            parsed=parsed,
        )


def openai_compatible(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    timeout: float | None = None,
    input_price: float | None = None,
    output_price: float | None = None,
    schema_mode: str = "native",
) -> Model:
    """Build a :class:`Model` for any OpenAI-compatible Chat Completions server.

    ``timeout`` (seconds) bounds each HTTP request at the SDK-client level --
    the only in-flight guard against a hung server; ``@agent(timeout=)`` is a
    turn-boundary watchdog and cannot interrupt a single call. There is no
    ``"provider/model-id"`` string form for this adapter (unlike
    ``"anthropic/..."`` / ``"openai/..."``) since ``base_url`` is required and
    varies per deployment; ``registry.resolve()`` accepts the returned
    ``Model`` instance directly as a passthrough.

    ``input_price``/``output_price`` (USD per **million** tokens, both
    together or neither) register this model's price so ``compose costs``,
    ``Run.usage.cost_usd``, and ``Budget(usd=...)`` work for paid compat
    providers (ollama.com, hosted vLLM, ...) -- without a price, spend is
    invisible to a USD budget (see ``Budget``'s docstring). Registration is
    process-global per ``(provider, model)``: last writer wins.

    ``schema_mode="prompt"`` embeds the output schema in the prompt and
    parses the reply leniently (fence-stripping, first balanced object) --
    for servers that accept but ignore ``response_format`` (Ollama cloud
    does, for every model). Reasoning models burn hidden tokens before
    content in this mode: keep ``max_tokens`` generous. When the reply
    still doesn't parse, that surfaces as a repairable ``ComposeError``
    from the agent loop -- eligible for ``@agent(max_repairs=)`` turns,
    not ``retries=`` (which stays provider-error territory).

    ``schema_mode="auto"`` starts native and permanently demotes to prompt
    mode the first time a call gets back a generic non-JSON structured
    reply, retrying that one call in prompt mode and then staying in prompt
    mode for the rest of this model instance's lifetime -- at most one
    wasted native call ever. ``stream()`` follows the current effective
    mode but never demotes itself. Transport errors and the
    reasoning-tokens-only diagnostic never demote, in any mode.
    """
    if (input_price is None) != (output_price is None):
        raise ConfigError(
            "openai_compatible(): pass both input_price and output_price "
            "(USD per million tokens), or neither"
        )
    if input_price is not None and output_price is not None:
        register_price(_PROVIDER, model, ModelPrice(input=input_price, output=output_price))
    return OpenAICompatibleModel(
        model, base_url=base_url, api_key=api_key, timeout=timeout, schema_mode=schema_mode
    )


# --- request-side mapping ----------------------------------------------------


def _build_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    api_messages: list[dict[str, Any]] = []
    if system is not None:
        api_messages.append({"role": "system", "content": system})

    for msg in messages:
        buffer: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for part in msg.parts:
            if isinstance(part, TextPart):
                buffer.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                buffer.append(_image_part_to_content(part))
            elif isinstance(part, ToolCallPart):
                tool_calls.append(_tool_call_part_to_param(part))
            elif isinstance(part, ToolResultPart):
                _flush(api_messages, msg.role, buffer, tool_calls)
                api_messages.append(_tool_result_part_to_message(part))
            elif isinstance(part, ThinkingPart):
                # Chat Completions has no reasoning-item concept to echo
                # back to -- thinking parts are always dropped, even for
                # same-provider/same-model round trips.
                continue
            else:
                unhandled = f"unhandled ContentPart type: {type(part)!r}"
                raise AssertionError(unhandled)  # pragma: no cover
        _flush(api_messages, msg.role, buffer, tool_calls)
    return api_messages


def _flush(
    api_messages: list[dict[str, Any]],
    role: str,
    buffer: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> None:
    if not buffer and not tool_calls:
        return
    message: dict[str, Any] = {"role": role}
    if buffer:
        message["content"] = list(buffer)
    elif role == "assistant":
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = list(tool_calls)
    api_messages.append(message)
    buffer.clear()
    tool_calls.clear()


def _image_part_to_content(part: ImagePart) -> dict[str, Any]:
    if part.data is not None:
        url = f"data:{part.media_type};base64,{part.data}"
    else:
        url = part.url
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_call_part_to_param(part: ToolCallPart) -> dict[str, Any]:
    return {
        "id": part.id,
        "type": "function",
        "function": {"name": part.name, "arguments": json.dumps(part.arguments)},
    }


def _tool_result_part_to_message(part: ToolResultPart) -> dict[str, Any]:
    # ChatCompletionToolMessageParam has no error flag -- fold is_error into
    # the text, same convention as the Responses adapter.
    content = part.content if not part.is_error else f"ERROR: {part.content}"
    return {"role": "tool", "tool_call_id": part.tool_call_id, "content": content}


def _tool_to_param(spec: ToolSpec) -> dict[str, Any]:
    # Chat Completions function tools nest under a "function" key (unlike
    # the Responses API, which is flat) -- verified against
    # ChatCompletionFunctionToolParam / FunctionDefinition. `spec.strict` is
    # always True (tools.py hardcodes it); downgrade to `strict: False` for
    # a schema that violates the strict-mode subset rather than sending one
    # a strict-enforcing server would reject -- see models/openai.py's
    # `_tool_to_param` for the full rationale. Never mutate
    # `spec.input_schema` itself.
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
            "strict": spec.strict and _is_strict_compatible(spec.input_schema),
        },
    }


def _schema_name(schema: dict[str, Any]) -> str:
    title = schema.get("title")
    if isinstance(title, str) and _SCHEMA_NAME_RE.match(title):
        return title
    return _DEFAULT_SCHEMA_NAME


_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def _append_schema_instruction(
    api_messages: list[dict[str, Any]], schema: dict[str, Any]
) -> None:
    instruction = (
        "Respond ONLY with a single JSON object that conforms to this JSON "
        "Schema -- no markdown fences, no commentary:\n" + json.dumps(schema)
    )
    part = {"type": "text", "text": instruction}
    for message in reversed(api_messages):
        if message["role"] == "user" and isinstance(message.get("content"), list):
            message["content"].append(part)
            return
    api_messages.append({"role": "user", "content": [part]})


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def _iter_balanced_objects(text: str) -> Iterator[str]:
    """Yield every balanced top-level ``{...}`` substring in ``text``, left to right.

    Tracks JSON string state so braces inside string values don't unbalance
    the scan. A candidate that never balances (an opening ``{`` with no
    matching close) simply ends the scan from that point. Callers that only
    want the first candidate can call ``next(iter, None)``; ``_lenient_json_parse``
    walks the whole sequence so a false lead (balanced but not valid JSON,
    e.g. ``{placeholder}``) doesn't stop it from finding a real object later
    in the same text.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break
        start = text.find("{", start + 1)


def _lenient_json_parse(text: str) -> dict[str, Any] | None:
    """Best-effort JSON recovery for ``schema_mode="prompt"`` responses."""
    candidate = _strip_fences(text)
    try:
        result = json.loads(candidate)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    # Try every balanced `{...}` substring in turn, not just the first -- a
    # balanced-but-invalid lead (e.g. prose containing a literal `{placeholder}`)
    # would otherwise mask a real JSON object appearing later in the text.
    for balanced in _iter_balanced_objects(candidate):
        try:
            result = json.loads(balanced)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    return None


# --- response-side mapping ----------------------------------------------------


def _message_to_parts(message: Any, model_id: str) -> tuple[list[ContentPart], bool]:
    """Return ``(parts, had_refusal)``.

    ``had_refusal`` is ``True`` when the message carried a non-empty
    ``message.refusal`` -- a genuine model refusal typically comes back with
    ``finish_reason`` still ``"stop"`` (Chat Completions has no dedicated
    "refusal" finish reason the way it has ``"content_filter"``), so
    ``_finalize`` must consult this alongside (and take priority over) the
    ``finish_reason`` -> ``StopReason`` table, mirroring how
    ``models/openai.py``'s ``_item_to_parts`` tracks ``is_refusal`` per
    content item for the Responses API's analogous ``refusal`` content type.
    """
    parts: list[ContentPart] = []
    had_refusal = False
    content = getattr(message, "content", None)
    if content:
        parts.append(TextPart(text=content))
    refusal = getattr(message, "refusal", None)
    if refusal:
        parts.append(TextPart(text=refusal))
        had_refusal = True
    for tool_call in getattr(message, "tool_calls", None) or []:
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            name = tool_call.function.name
            raise ProviderError(
                f"Server returned non-JSON arguments for tool call {name!r}: {exc}",
                provider=_PROVIDER,
                model=model_id,
            ) from exc
        parts.append(
            ToolCallPart(id=tool_call.id, name=tool_call.function.name, arguments=arguments)
        )
    return parts, had_refusal


def _map_usage(usage: Any, model_id: str) -> Usage:
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cache_read_tokens = (getattr(prompt_details, "cached_tokens", 0) or 0) if prompt_details else 0
    # `cache_write_tokens` genuinely exists on the wire -- verified against
    # the installed SDK's `openai.types.completion_usage.PromptTokensDetails`,
    # which documents it as "The unadjusted number of prompt tokens written
    # to cache" -- so it's mapped through to `cache_creation_tokens` for
    # observability (per the brief: map a cache-write field that genuinely
    # exists rather than leaving it at a guessed 0). It is NOT priced
    # separately below: this project's price table (prices.py) never sets an
    # `openai`/`openai-compatible` `cache_write_5m`/`cache_write_1h`
    # override, and `compute_cost`'s cache-write multipliers (1.25x/2x of
    # input) are shaped for Anthropic's actual TTL-tiered cache-write
    # pricing -- applying them here would be exactly the kind of guessed,
    # unverified price this project's "never fabricate cost" rule (see
    # prices.py's module docstring) forbids. Like `cache_write_tokens` on
    # the OpenAI side (see models/openai.py's `_map_usage`), it's already
    # included in `input_tokens` at the ordinary input rate, which is the
    # only verified-correct price available for it.
    cache_write_tokens = (
        (getattr(prompt_details, "cache_write_tokens", 0) or 0) if prompt_details else 0
    )

    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = (
        (getattr(completion_details, "reasoning_tokens", 0) or 0) if completion_details else 0
    )

    price = get_price(_PROVIDER, model_id)
    if price is None:
        cost_usd = None
        cost_complete = False
    else:
        # `prompt_tokens` INCLUDES `cached_tokens` -- verified against the
        # installed SDK's `PromptTokensDetails` docstring ("Breakdown of
        # tokens used in the prompt", i.e. cached_tokens is a subset, not
        # additive) and confirmed unambiguously by the structurally
        # identical `openai.types.realtime.
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
        cache_creation_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=cost_usd,
        cost_complete=cost_complete,
    )
