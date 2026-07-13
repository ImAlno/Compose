"""Core message and usage types shared by adapters, the agent loop, and tracing.

Content is modeled as a discriminated union of small, frozen part types
(text, image, tool call/result, thinking) so adapters can normalize
provider-specific payloads into one shape. The system prompt is
deliberately *not* a message: providers treat it as a distinct top-level
request field, so composeai does the same rather than smuggling it into
``parts[0]``.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._encoding import register_serializable


class TextPart(BaseModel):
    """Plain text content."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """Image content, given as inline base64 ``data`` or a fetchable ``url``."""

    model_config = ConfigDict(frozen=True)

    type: Literal["image"] = "image"
    media_type: str
    data: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _check_exactly_one_source(self) -> ImagePart:
        if (self.data is None) == (self.url is None):
            raise ValueError(
                "ImagePart requires exactly one of `data` or `url`, not both/neither"
            )
        return self


class ToolCallPart(BaseModel):
    """A tool invocation requested by the model."""

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResultPart(BaseModel):
    """The result of executing a tool call, fed back to the model."""

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


class ThinkingPart(BaseModel):
    """Extended/reasoning content.

    ``provider_token`` is an opaque continuation token some providers
    require to resume reasoning across turns. Adapters must only echo it
    back to the same provider + model it came from, and drop it silently
    otherwise; this module just carries the fields.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["thinking"] = "thinking"
    text: str = ""
    provider_token: str | None = None
    provider: str | None = None


ContentPart = Annotated[
    TextPart | ImagePart | ToolCallPart | ToolResultPart | ThinkingPart,
    Field(discriminator="type"),
]


def _as_parts(text_or_parts: str | Sequence[ContentPart]) -> list[ContentPart]:
    if isinstance(text_or_parts, str):
        return [TextPart(text=text_or_parts)]
    return list(text_or_parts)


class Message(BaseModel):
    """A single turn in a conversation.

    The system prompt is never a ``Message``; it is a top-level request
    field on the adapter/agent API (a provider API constraint).
    """

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    parts: list[ContentPart]

    @classmethod
    def user(cls, text_or_parts: str | Sequence[ContentPart]) -> Message:
        """Build a user message from plain text or a list of parts."""
        return cls(role="user", parts=_as_parts(text_or_parts))

    @classmethod
    def assistant(cls, text_or_parts: str | Sequence[ContentPart]) -> Message:
        """Build an assistant message from plain text or a list of parts."""
        return cls(role="assistant", parts=_as_parts(text_or_parts))

    @property
    def text(self) -> str:
        """Concatenation of all ``TextPart`` texts in order, ignoring other parts."""
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))


class Usage(BaseModel):
    """Token counts and cost for one or more LLM calls.

    Cost is never fabricated: when an adapter can't price a call it sets
    ``cost_usd=None`` and ``cost_complete=False`` so totals downstream are
    flagged as partial (e.g. a renderer can print "≥$X"). The zero-usage
    default ``Usage()`` is the one case where ``cost_usd`` is ``None``
    while ``cost_complete`` stays ``True`` -- it represents "nothing
    happened yet", not "unknown price".
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    cost_complete: bool = True

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        if self.cost_usd is None and other.cost_usd is None:
            cost_usd = None
        else:
            cost_usd = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=cost_usd,
            cost_complete=self.cost_complete and other.cost_complete,
        )


class StopReason(str, enum.Enum):
    """Why a model turn stopped, normalized across providers.

    Adapters map provider-specific raw stop values to these and also keep
    the raw string separately (not modeled here).
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    REFUSAL = "refusal"
    ERROR = "error"
    OTHER = "other"


# Phase 8 (human-in-the-loop): `composeai.agentfn` persists an agent's
# conversation (`list[Message]`, including `ToolResultPart`s) to the
# `agent_state` table so a pause can be resumed mid-loop. `to_jsonable`
# auto-registers a type the first time it *encodes* an instance of it, but a
# fresh process resuming a paused run (see `tests/test_hitl_subprocess.py`'s
# flow-level equivalent) may need to *decode* one of these before ever
# encoding one -- so these are registered eagerly here, at import time,
# rather than relying on that lazy path.
#
# `Usage` joins this list for Phase 9: `composeai.cli` (a fresh process,
# never encoding a `Usage` before it needs to decode one) reads `spans.
# usage_json` back via `from_jsonable` -- the same "decode before any
# encode in this process" gap, just hit by a new caller.
register_serializable(TextPart)
register_serializable(ImagePart)
register_serializable(ToolCallPart)
register_serializable(ToolResultPart)
register_serializable(ThinkingPart)
register_serializable(Message)
register_serializable(Usage)
