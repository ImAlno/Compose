"""Human-in-the-loop: ``approve``/``ask_human`` and the pause signal they raise.

One mechanism serves both ``@flow`` bodies and ``@agent`` runs (standalone or
nested in a flow): an interrupt is **named** (an ``id`` the caller picks,
never a positional/completion-order slot), its answer lives in the run's
journal under a reserved ``f"__interrupt__:{id}"`` key, and resuming a run
means re-running it with that answer already journaled -- ``approve``/
``ask_human`` simply look the answer up first and only pause (raise
:class:`_Pause`) on a miss.

:class:`_Pause` is a :class:`BaseException`, not an :class:`Exception`: a
user's ``except Exception`` (e.g. around a tool body, or inside a
``@task``'s retry loop) must never accidentally swallow a pause the way it
would a real failure. :mod:`composeai.tracing`'s ``span()`` recognizes it via
the ``_compose_pause`` class attribute (duck-typed there, to avoid a
tracing<->hitl import cycle -- this module imports ``tracing`` for
:func:`~composeai.tracing.span`) and marks the enclosing span ``"paused"``
with no :class:`~composeai.tracing.ErrorInfo`, rather than ``"error"`` --
pausing is control flow, never a failure, anywhere in a trace.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from . import runs, tracing


class Interrupt(BaseModel):
    """A named pause point raised by :func:`approve`/:func:`ask_human` (or an
    approval-gated ``@tool`` call, via :mod:`composeai.agentfn`) when its
    journaled answer isn't present yet.

    ``payload`` must be JSON-encodable via
    :func:`~composeai._encoding.to_jsonable` once persisted as a pending
    interrupt row -- an in-memory-only ``Interrupt`` that's immediately
    caught and inspected without ever reaching the store has no such
    restriction.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["approval", "question"]
    question: str | None = None
    payload: Any = None


class ApprovalReply(BaseModel):
    """A structured approver reply that can carry a denial message.

    Returned (instead of a bare ``bool``) from an ``approver=`` callable for
    an approval-gated ``@tool`` call. ``allow`` decides whether the call runs;
    when a denial (``allow=False``) supplies a ``message``, the agent sees that
    text as the denied tool's result instead of the default ``"denied by
    user"`` -- a live-approver-only channel for feedback (a resumed/journaled
    denial has no message; see ``_aprocess_tool_use``). A plain ``True``/
    ``False`` return stays fully supported and behaves identically. For an
    allow, ``message`` is currently ignored (only denials carry feedback).
    """

    model_config = ConfigDict(frozen=True)

    allow: bool
    message: str | None = None


class _Pause(BaseException):
    """Internal control-flow signal carrying the :class:`Interrupt` that caused it.

    Never meant to be caught by user code -- :class:`BaseException`, not
    :class:`Exception`, specifically so a stray ``except Exception`` can't
    swallow it. ``@flow.run()``/``resume()`` and the ``@agent`` loop catch it
    explicitly (by duck-typed ``_compose_pause`` attribute, in modules that
    can't import this one without cycling back to it) to turn it into a
    ``Run(status="paused", pending=interrupt)`` instead of letting it fail
    the run.
    """

    _compose_pause = True

    def __init__(self, interrupt: Interrupt) -> None:
        super().__init__(f"paused on interrupt {interrupt.id!r}")
        self.interrupt = interrupt


def _journal_key(id: str) -> str:
    return f"__interrupt__:{id}"


def approve(id: str, payload: Any = None) -> bool:
    """Ask for (or retrieve) approval under the named interrupt ``id``.

    Looks the answer up in the ambient run's journal first (an active
    ``@flow`` body, or a durable ``@agent`` run); a hit returns it coerced
    to ``bool``. A miss raises :class:`_Pause`, which ``@flow.run()``/
    ``resume()`` (or the agent loop) turns into a paused ``Run`` -- pausing
    is not an error, and the process may exit right after it.

    Raises :class:`~composeai.errors.ConfigError` if there is no active
    ``@flow`` body or durable ``@agent`` run to journal the answer against.
    """
    hit, value = runs.interrupt_lookup(_journal_key(id))
    if hit:
        return bool(value)
    with tracing.span("pause", id):
        raise _Pause(Interrupt(id=id, kind="approval", payload=payload))


def ask_human(id: str, question: str, payload: Any = None) -> Any:
    """Ask for (or retrieve) a free-form answer under the named interrupt ``id``.

    Same journal-first-then-pause mechanics as :func:`approve`, but returns
    whatever value the answer was (no ``bool`` coercion) and requires a
    ``question`` describing what's being asked (carried on the
    :class:`Interrupt` for whoever answers it). Also usable from inside a
    ``@tool`` body: a pause raised there propagates out of the tool call
    (it's a :class:`BaseException`) and the agent loop snapshots the
    in-flight conversation so resuming re-executes just that tool from the
    top -- tool bodies that call ``ask_human`` must be idempotent up to the
    point of the ask.
    """
    hit, value = runs.interrupt_lookup(_journal_key(id))
    if hit:
        return value
    with tracing.span("pause", id):
        raise _Pause(Interrupt(id=id, kind="question", question=question, payload=payload))


__all__ = ["ApprovalReply", "Interrupt", "approve", "ask_human"]
