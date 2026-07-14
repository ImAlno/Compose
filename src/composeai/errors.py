"""composeai's exception hierarchy.

Every error composeai raises is a :class:`ComposeError`. Design goal: the
hierarchy stays shallow and each type does little beyond carry a message
(and, for :class:`ProviderError`, a couple of attributes) -- exceptions
should never accumulate deep framework frames, so a user's traceback
stays focused on their own code rather than composeai's internals.
"""

from __future__ import annotations


class ComposeError(Exception):
    """Base class for every error raised by composeai."""


class ConfigError(ComposeError):
    """Bad or missing configuration, e.g. an absent API key or unknown model string."""


class ProviderError(ComposeError):
    """A provider/SDK call failed.

    Attributes:
        provider: The provider name (e.g. ``"anthropic"``), if known.
        model: The model string in use when the failure occurred, if known.
    """

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class SerializationError(ComposeError):
    """A value can't round-trip the journal encoder.

    The message names the full path to the offending value within the
    structure being encoded or decoded.
    """


class ResumeMismatchError(ComposeError):
    """A flow's source changed between the run being paused and resumed."""


class BudgetExceededError(ComposeError):
    """A run hit its configured :class:`Budget` cap."""


class CompositionTypeError(ComposeError, TypeError):
    """``pipe()`` wiring connects two stages with mismatched types.

    Also a :class:`TypeError`, so it's catchable either as a composeai
    error or as a plain Python type error.
    """


class MaxTurnsExceededError(ComposeError):
    """An ``@agent`` run used more LLM turns than its configured ``max_turns``."""


class ModelRefusalError(ComposeError):
    """The model refused to respond (``stop_reason`` was ``REFUSAL``).

    Attributes:
        raw: The provider's raw stop-reason string, if known.
    """

    def __init__(self, message: str = "", *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


class AgentTimeoutError(ComposeError):
    """An ``@agent`` run exceeded its configured ``timeout``.

    Checked only at turn boundaries (between LLM calls) -- a single
    in-flight model call is never interrupted mid-flight.
    """


class TaskTimeoutError(ComposeError):
    """A ``@task`` call exceeded its configured ``timeout``.

    There is no safe way to interrupt an arbitrary Python thread, so the
    task body keeps running to completion (or forever) in an abandoned
    daemon thread after this is raised; its eventual result or exception
    is discarded. Treat a timed-out task as failed and move on.
    """


class MCPToolError(ComposeError):
    """An MCP tool call failed: the server reported ``isError``, the
    transport failed mid-run, the per-call timeout expired, or the server
    connection was already closed.

    Inside an agent run this surfaces exactly like any tool-body
    exception: an ``is_error`` tool result the model can react to -- never
    a run abort (see ``composeai.agentfn._execute_one_tool``).
    """
