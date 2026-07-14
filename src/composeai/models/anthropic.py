"""Anthropic Messages API adapter.

The ``anthropic`` SDK is imported only from within this module, and only
lazily (inside :meth:`AnthropicModel.complete` / :meth:`AnthropicModel.stream`,
via :meth:`AnthropicModel._get_client`) -- so importing :mod:`composeai`,
importing this module, or constructing an :class:`AnthropicModel` never
requires the SDK to be installed. Passing your own ``client=`` skips
*constructing* the SDK's client (and so the API-key check in
:meth:`AnthropicModel._get_client`), but does not avoid the SDK dependency
entirely: :meth:`complete`/:meth:`stream` still ``import anthropic``
themselves (for ``anthropic.APIError`` exception mapping), so the package
must be installed before either is actually called, client injection or not.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from ..errors import ComposeError, ConfigError, ProviderError
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

_MAX_PAUSE_CONTINUATIONS = 5

_STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
    # "stop_sequence" and anything unrecognized fall through to OTHER via
    # the .get(..., StopReason.OTHER) lookup below.
}


class AnthropicModel:
    """``Model`` adapter for Anthropic's Messages API (``client.messages.create``)."""

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
        self._api_key = api_key
        self._max_retries = max_retries
        self._base_url = base_url
        self._timeout = timeout

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError(
                "Missing Anthropic API key: set the ANTHROPIC_API_KEY environment "
                "variable, or pass api_key=... / client=... to AnthropicModel."
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
        self._client = anthropic.Anthropic(**client_kwargs)
        return self._client

    def complete(self, request: ModelRequest) -> ModelResponse:
        import anthropic

        client = self._get_client()
        api_messages = _build_messages(request.messages, self.model_id)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": request.max_tokens,
            "messages": api_messages,
        }
        if request.system is not None:
            kwargs["system"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = [_tool_to_param(spec) for spec in request.tools]
        if request.output_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": request.output_schema}
            }

        accumulated_parts: list[ContentPart] = []
        total_usage = Usage()
        continuations = 0
        response: Any = None

        while True:
            try:
                response = client.messages.create(**kwargs)
            except anthropic.APIError as exc:
                raise ProviderError(str(exc), provider="anthropic", model=self.model_id) from exc

            turn_parts = [
                part
                for part in (_block_to_part(block, self.model_id) for block in response.content)
                if part is not None
            ]
            accumulated_parts.extend(turn_parts)
            total_usage = total_usage + _map_usage(response.usage, self.model_id)

            if response.stop_reason != "pause_turn":
                break

            continuations += 1
            if continuations > _MAX_PAUSE_CONTINUATIONS:
                raise ProviderError(
                    f"Anthropic paused {continuations} times in a row for "
                    f"model {self.model_id!r} (exceeded the "
                    f"{_MAX_PAUSE_CONTINUATIONS}-continuation limit)",
                    provider="anthropic",
                    model=self.model_id,
                )
            continuation_blocks = [
                block
                for block in (_part_to_block(part, self.model_id) for part in turn_parts)
                if block is not None
            ]
            kwargs["messages"].append({"role": "assistant", "content": continuation_blocks})

        return self._finalize(accumulated_parts, response.stop_reason, total_usage, request)

    def stream(self, request: ModelRequest) -> Iterator[RawStreamEvent]:
        """Stream a completion, yielding token/tool deltas then ``response_done``.

        Uses the SDK's ``client.messages.stream(...)`` helper. Applies the
        same ``pause_turn`` continuation policy as :meth:`complete` (keep
        streaming, up to :data:`_MAX_PAUSE_CONTINUATIONS` continuations) and
        the same error mapping to :class:`ProviderError`; the final
        ``response_done`` event carries a :class:`ModelResponse` built via
        the exact same finalization helper :meth:`complete` uses, so
        consuming only that event is equivalent to calling ``complete()``.
        """
        import anthropic

        client = self._get_client()
        api_messages = _build_messages(request.messages, self.model_id)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": request.max_tokens,
            "messages": api_messages,
        }
        if request.system is not None:
            kwargs["system"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = [_tool_to_param(spec) for spec in request.tools]
        if request.output_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": request.output_schema}
            }

        accumulated_parts: list[ContentPart] = []
        total_usage = Usage()
        continuations = 0
        final_message: Any = None

        while True:
            # index -> the content_block seen at content_block_start, so
            # later content_block_delta/content_block_stop events (which
            # don't repeat the block's id/name/type) can be correlated back
            # to the right tool call.
            block_by_index: dict[int, Any] = {}
            try:
                with client.messages.stream(**kwargs) as stream:
                    for raw_event in stream:
                        event_type = getattr(raw_event, "type", None)
                        if event_type == "content_block_start":
                            block = raw_event.content_block
                            block_by_index[raw_event.index] = block
                            if getattr(block, "type", None) == "tool_use":
                                yield RawStreamEvent(
                                    kind="tool_call_started",
                                    tool_call_id=block.id,
                                    tool_name=block.name,
                                )
                        elif event_type == "content_block_delta":
                            delta = raw_event.delta
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                yield RawStreamEvent(kind="text_delta", text=delta.text)
                            elif delta_type == "thinking_delta":
                                yield RawStreamEvent(kind="thinking_delta", text=delta.thinking)
                            elif delta_type == "input_json_delta":
                                block = block_by_index.get(raw_event.index)
                                yield RawStreamEvent(
                                    kind="tool_args_delta",
                                    text=delta.partial_json,
                                    tool_call_id=getattr(block, "id", None),
                                    tool_name=getattr(block, "name", None),
                                )
                            # citations_delta / signature_delta carry no
                            # RawStreamEvent kind of their own -- ignored.
                        elif event_type == "content_block_stop":
                            block = block_by_index.get(raw_event.index)
                            if block is not None and getattr(block, "type", None) == "tool_use":
                                yield RawStreamEvent(
                                    kind="tool_call_finished",
                                    tool_call_id=block.id,
                                    tool_name=block.name,
                                )
                        # message_start/message_delta, and any of the SDK's
                        # own derived per-kind events (TextEvent,
                        # ThinkingEvent, InputJsonEvent, ...) interleaved
                        # into the same iteration, carry nothing new here.
                    final_message = stream.get_final_message()
            except anthropic.APIError as exc:
                raise ProviderError(str(exc), provider="anthropic", model=self.model_id) from exc

            turn_parts = [
                part
                for part in (
                    _block_to_part(block, self.model_id) for block in final_message.content
                )
                if part is not None
            ]
            accumulated_parts.extend(turn_parts)
            total_usage = total_usage + _map_usage(final_message.usage, self.model_id)

            if final_message.stop_reason != "pause_turn":
                break

            continuations += 1
            if continuations > _MAX_PAUSE_CONTINUATIONS:
                raise ProviderError(
                    f"Anthropic paused {continuations} times in a row for "
                    f"model {self.model_id!r} (exceeded the "
                    f"{_MAX_PAUSE_CONTINUATIONS}-continuation limit)",
                    provider="anthropic",
                    model=self.model_id,
                )
            continuation_blocks = [
                block
                for block in (_part_to_block(part, self.model_id) for part in turn_parts)
                if block is not None
            ]
            kwargs["messages"].append({"role": "assistant", "content": continuation_blocks})

        response = self._finalize(
            accumulated_parts, final_message.stop_reason, total_usage, request
        )
        yield RawStreamEvent(kind="response_done", response=response)

    def _finalize(
        self,
        accumulated_parts: list[ContentPart],
        raw_stop_reason: str,
        total_usage: Usage,
        request: ModelRequest,
    ) -> ModelResponse:
        message = Message(role="assistant", parts=accumulated_parts)
        stop_reason = _STOP_REASON_MAP.get(raw_stop_reason, StopReason.OTHER)

        # Only attempt to decode structured output on a clean END_TURN: a
        # tool_use turn legitimately has no text at all (the model called a
        # tool instead of returning its final answer -- agentfn.py sends
        # output_schema on *every* turn of an @agent that combines
        # tools=[...] with a non-str output_type, so this is a normal
        # mid-conversation shape, not an error), and REFUSAL/MAX_TOKENS/OTHER
        # already have their own dedicated handling one level up in
        # agentfn.py's turn loop (ModelRefusalError / a max_tokens
        # ComposeError) that a spurious JSON-parse ProviderError would only
        # mask. agentfn.py's _extract_output (the only reader of `.parsed`)
        # is likewise only ever called on the END_TURN branch, so `None`
        # here for every other stop reason is exactly what it expects.
        parsed: dict[str, Any] | None = None
        if request.output_schema is not None and stop_reason == StopReason.END_TURN:
            try:
                parsed = json.loads(message.text)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    "Anthropic returned non-JSON text for a structured-output "
                    f"request: {exc}",
                    provider="anthropic",
                    model=self.model_id,
                ) from exc

        return ModelResponse(
            message=message,
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
            usage=total_usage,
            model_id=self.model_id,
            parsed=parsed,
        )


def create_model(model_id: str) -> AnthropicModel:
    """Factory used by :mod:`composeai.models.registry` for ``"anthropic/..."`` strings."""
    return AnthropicModel(model_id)


# --- request-side mapping ----------------------------------------------------


def _build_messages(messages: list[Message], model_id: str) -> list[dict[str, Any]]:
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        parts = msg.parts
        has_tool_result = any(isinstance(p, ToolResultPart) for p in parts)
        if has_tool_result and not all(isinstance(p, ToolResultPart) for p in parts):
            raise ComposeError(
                f"invalid message for Anthropic model {model_id!r}: a message "
                "containing tool_result parts must contain only tool_result "
                "parts; got mixed content in one message -- this is a caller "
                "bug, not a provider failure (batch all tool results for one "
                "turn into their own message)"
            )
        blocks = [
            block
            for block in (_part_to_block(part, model_id) for part in parts)
            if block is not None
        ]
        api_messages.append({"role": msg.role, "content": blocks})
    return api_messages


def _tool_to_param(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
        "strict": spec.strict,
    }


def _part_to_block(part: ContentPart, model_id: str) -> dict[str, Any] | None:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        if part.data is not None:
            source = {"type": "base64", "media_type": part.media_type, "data": part.data}
        else:
            source = {"type": "url", "url": part.url}
        return {"type": "image", "source": source}
    if isinstance(part, ToolCallPart):
        return {"type": "tool_use", "id": part.id, "name": part.name, "input": part.arguments}
    if isinstance(part, ToolResultPart):
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": part.tool_call_id,
            "content": part.content,
        }
        if part.is_error:
            block["is_error"] = True
        return block
    if isinstance(part, ThinkingPart):
        # Only echo a thinking block back to the same provider + model it
        # came from; other-model thinking is ignored by the API anyway, so
        # we drop it silently rather than sending garbage.
        if part.provider != "anthropic" or part.provider_token is None:
            return None
        try:
            payload = json.loads(part.provider_token)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"malformed thinking provider_token for Anthropic model {model_id!r}: {exc}",
                provider="anthropic",
                model=model_id,
            ) from exc
        if payload.get("model") != model_id:
            return None
        # Strip None-valued keys (e.g. a missing `signature`) before
        # re-sending -- an explicit `"signature": null` risks a live-API 400.
        return {k: v for k, v in payload["block"].items() if v is not None}
    raise AssertionError(f"unhandled ContentPart type: {type(part)!r}")  # pragma: no cover


# --- response-side mapping ----------------------------------------------------


def _block_to_part(block: Any, model_id: str) -> ContentPart | None:
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return TextPart(text=block.text)
    if block_type == "tool_use":
        return ToolCallPart(id=block.id, name=block.name, arguments=dict(block.input))
    if block_type == "thinking":
        raw_block = {
            "type": "thinking",
            "thinking": getattr(block, "thinking", "") or "",
            "signature": getattr(block, "signature", None),
        }
        token = json.dumps({"block": raw_block, "model": model_id})
        return ThinkingPart(text=raw_block["thinking"], provider="anthropic", provider_token=token)
    # Unknown/unsupported block types (citations, redacted_thinking, server
    # tool blocks, future additions, ...) are dropped rather than raising.
    return None


def _map_usage(usage: Any, model_id: str) -> Usage:
    # Unlike OpenAI (see models/openai.py's `_map_usage` for the double-
    # billing fix there), Anthropic's `input_tokens` does NOT include cache
    # reads/writes -- verified against the installed anthropic SDK's
    # `types/usage.py`: `input_tokens` ("The number of input tokens which
    # were used"), `cache_read_input_tokens` ("The number of input tokens
    # read from the cache"), and `cache_creation_input_tokens` ("The number
    # of input tokens used to create the cache entry") are documented as
    # separate, sibling fields, not one nested as a "breakdown of" another
    # the way OpenAI's `input_tokens_details.cached_tokens` is. The SDK's
    # own streaming accumulator (`anthropic/lib/streaming/_messages.py`)
    # confirms this in code: it copies `input_tokens`,
    # `cache_creation_input_tokens`, and `cache_read_input_tokens`
    # independently from each usage event, with no subtraction between
    # them. So billing `input_tokens * input_price +
    # cache_read_tokens * cache_read_price + cache_write * cache_write_price`
    # below (no subtraction of cache tokens out of `input_tokens`) is
    # already correct and must stay that way -- see
    # test_cache_read_tokens_are_billed_additively_not_subtracted_from_input
    # in test_anthropic.py for the pinning regression test.
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

    cache_creation = getattr(usage, "cache_creation", None)
    if cache_creation is not None:
        cache_write_5m = getattr(cache_creation, "ephemeral_5m_input_tokens", 0) or 0
        cache_write_1h = getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0
    else:
        cache_write_5m = cache_creation_tokens
        cache_write_1h = 0

    price = get_price("anthropic", model_id)
    if price is None:
        cost_usd = None
        cost_complete = False
    else:
        cost_usd = compute_cost(
            Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
            ),
            price,
            cache_write_5m,
            cache_write_1h,
        )
        cost_complete = True

    # reasoning_tokens is left at its Usage() default of 0 here. The real
    # API does report a `usage.output_tokens_details.thinking_tokens` field
    # (verified against the installed SDK's `types/output_tokens_details.py`
    # -- it exists), so this isn't a wire-format gap the way it might sound;
    # rather, extended thinking itself can't be turned on through
    # composeai's public API yet (there's no `thinking`/budget_tokens field
    # on ModelRequest -- see docs/design.md's "Out of scope v1" list), so a
    # real response never populates `output_tokens_details` in the first
    # place and mapping it here would be dead code serving an unreachable
    # feature. Revisit alongside adding the request-side `thinking` config
    # surface, not before.
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cost_usd=cost_usd,
        cost_complete=cost_complete,
    )
