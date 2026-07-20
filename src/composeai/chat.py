"""Persistent multi-turn chat sessions over @agent functions.

A Chat is the middle layer between one-shot agent calls and durable @flow
workflows: it keeps a conversation's history, sends each new user turn as a
normal durable agent run seeded with that history, and persists the updated
history to the run store after every completed send.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import runs
from ._encoding import from_jsonable, to_jsonable
from ._ids import new_ulid
from .agentfn import _AGENT_REGISTRY, AgentFunction, _run_agent, _stream_agent
from .errors import ConfigError
from .flow import resume as _resume
from .messages import Message

if TYPE_CHECKING:
    from .hitl import Interrupt
    from .models.base import Model
    from .runs import Budget, Run

__all__ = ["Chat", "ChatStream", "chat", "load_chat"]


class Chat:
    """A persistent conversation bound to one @agent function.

    Construct via compose.chat(agent_fn) or compose.load_chat(chat_id),
    never directly.
    """

    def __init__(
        self,
        agent_fn: AgentFunction,
        *,
        chat_id: str | None = None,
        messages: list[Message] | None = None,
        system: str | None = None,
        model: str | Model | None = None,
        approver: Callable[[Interrupt], bool] | None = None,
        context_manager: Callable[[list[Message], int], list[Message]] | None = None,
        created_at: float | None = None,
    ) -> None:
        self._agent = agent_fn
        self.id = chat_id or new_ulid()
        self._messages: list[Message] = list(messages or [])
        self._system = system
        self._model = model
        self._approver = approver
        self._context_manager = context_manager
        self._created_at = created_at if created_at is not None else time.time()
        self._pending: Interrupt | None = None
        self._last_run_id: str | None = None

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def pending(self) -> Interrupt | None:
        return self._pending

    def send(self, text: str, *, budget: Budget | None = None) -> Run[Any]:
        seed = [*self._messages, Message.user(text)]
        run = _run_agent(
            self._agent,
            (),
            {},
            budget=budget,
            system_override=self._system,
            model_override=self._model,
            approver=self._approver,
            context_manager=self._context_manager,
            seed_conversation=seed,
        )
        self._absorb(run)
        return run

    def resume(
        self,
        answers: dict[str, Any] | None = None,
        *,
        budget: Budget | None = None,
    ) -> Run[Any]:
        """Resume the last paused send and fold the outcome into history."""
        if self._pending is None or self._last_run_id is None:
            raise ConfigError("nothing to resume: this chat has no paused send")
        run = _resume(
            self._last_run_id,
            answers,
            budget=budget,
            approver=self._approver,
            context_manager=self._context_manager,
        )
        self._absorb(run)
        return run

    def stream(self, text: str, *, budget: Budget | None = None) -> ChatStream:
        seed = [*self._messages, Message.user(text)]
        inner = _stream_agent(
            self._agent,
            (),
            {},
            budget=budget,
            system_override=self._system,
            model_override=self._model,
            approver=self._approver,
            context_manager=self._context_manager,
            seed_conversation=seed,
        )
        return ChatStream(self, inner)

    def _absorb(self, run: Run[Any]) -> None:
        self._last_run_id = run.id
        if run.status == "completed":
            self._messages = list(run.messages)
            self._pending = None
            self._persist()
        elif run.status == "paused":
            self._pending = run.pending

    def _persist(self) -> None:
        store = runs.open_default()
        store.chat_put(
            chat_id=self.id,
            agent_name=self._agent.name,
            system=self._system,
            model=self._model if isinstance(self._model, str) else None,
            messages_json=json.dumps(to_jsonable(self._messages)),
            created_at=self._created_at,
            updated_at=time.time(),
        )


class ChatStream:
    """Event iterator for one chat send; folds the run into the chat on completion."""

    def __init__(self, chat_obj: Chat, inner: Any) -> None:
        self._chat = chat_obj
        self._inner = inner
        self._absorbed = False

    def __iter__(self) -> Any:
        yield from self._inner
        self._absorb()

    def close(self) -> None:
        self._inner.close()

    def cancel(self) -> None:
        """Cooperatively cancel the in-flight chat send (v0.9.0).

        Delegates to the underlying :class:`RunStream.cancel` -- no new
        turn/tool work starts, the in-flight LLM stream is aborted, and the
        run ends as ``Run(status="cancelled")``; iteration stops cleanly and
        :attr:`run` returns the cancelled ``Run`` without raising. A tool
        already executing runs to completion (cancellation is cooperative).
        Safe to call from another thread.
        """
        self._inner.cancel()

    @property
    def run(self) -> Run[Any]:
        run = self._inner.run
        self._absorb()
        return run

    def _absorb(self) -> None:
        if self._absorbed:
            return
        self._absorbed = True
        self._chat._absorb(self._inner.run)


def chat(
    agent_fn: AgentFunction,
    *,
    system: str | None = None,
    model: str | Model | None = None,
    approver: Callable[[Interrupt], bool] | None = None,
    context_manager: Callable[[list[Message], int], list[Message]] | None = None,
) -> Chat:
    """Start a new persistent chat over an @agent function."""
    return Chat(
        agent_fn,
        system=system,
        model=model,
        approver=approver,
        context_manager=context_manager,
    )


def load_chat(chat_id: str) -> Chat:
    """Restore a chat by id from the run store.

    The chatted agent must be imported (registered) in this process, same
    rule as resume().
    """
    store = runs.open_default()
    row = store.chat_get(chat_id)
    if row is None:
        raise ConfigError(f"no chat found with chat_id {chat_id!r}")
    agent_fn = _AGENT_REGISTRY.get(row["agent_name"])
    if agent_fn is None:
        raise ConfigError(
            f"chat {chat_id!r} belongs to agent {row['agent_name']!r}, which is not "
            "registered in this process -- import the module that defines it first"
        )
    messages = from_jsonable(json.loads(row["messages_json"])) if row["messages_json"] else []
    restored = Chat(
        agent_fn,
        chat_id=chat_id,
        messages=messages,
        system=row["system"],
        model=row["model"],
        created_at=row["created_at"],
    )
    return restored
