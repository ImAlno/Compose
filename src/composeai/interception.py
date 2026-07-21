"""The tool-interceptor seam: a before/after callback around every tool call.

An interceptor (passed as ``tool_interceptor=`` to a run/chat/agent) can, per
tool call, PROCEED (optionally with modified arguments), DENY (producing an
error tool-result the model sees), or observe/modify the result afterward. It
fires on the live execution path only and is never journaled, so a resumed run
does not re-fire ``after`` for tools that already executed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from composeai.messages import ToolCallPart, ToolResultPart


class BeforeTool(BaseModel):
    """An interceptor's decision for a single tool call, before it runs.

    ``action="proceed"`` runs the tool (with ``arguments`` if given, else the
    original args). ``action="deny"`` skips execution AND the approver, feeding
    ``message`` (or a default) back as the denied tool's error result.
    """

    model_config = ConfigDict(frozen=True)

    action: Literal["proceed", "deny"] = "proceed"
    arguments: dict[str, Any] | None = None
    message: str | None = None


@runtime_checkable
class ToolInterceptor(Protocol):
    """Fires before and after each tool execution. Both methods may return
    ``None`` to mean 'no change'."""

    def before(self, call: ToolCallPart) -> BeforeTool | None: ...

    def after(self, call: ToolCallPart, result: ToolResultPart) -> ToolResultPart | None: ...


__all__ = ["BeforeTool", "ToolInterceptor"]
