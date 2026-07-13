"""The event bus: one stream that carries both live output and trace telemetry.

Compose's headline feature is always-on tracing where streaming and
tracing are the same thing -- there is no separate "tracing mode" to turn
on. Every token, tool-call fragment, and span transition is an
:class:`Event` published to an :class:`EventBus`. A CLI or UI subscribes
to render live output; the tracing collector (see ``composeai.tracing``)
subscribes (or is simply the thing doing the emitting) to build the trace
tree. Both consume the exact same events, so there's nothing to keep in
sync.
"""

from __future__ import annotations

import contextvars
import queue
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Event:
    """One item on the bus.

    Constructed at high frequency during streaming, hence frozen +
    ``slots=True`` (cheap to allocate, immutable once published).
    """

    kind: Literal[
        "span_started",
        "text_delta",
        "thinking_delta",
        "tool_call_started",
        "tool_args_delta",
        "tool_call_finished",
        "span_finished",
        "paused",
        "run_finished",
    ]
    ts: float = field(default_factory=time.time)
    trace_id: str | None = None
    span_id: str | None = None
    name: str | None = None
    text: str | None = None
    """Payload for text_delta / thinking_delta."""
    data: dict[str, Any] | None = None
    """Kind-specific extras, e.g. ``{"kind": "agent", "status": "ok"}`` on
    span events, tool-args fragments, pause payloads."""


class _Closed:
    """Sentinel placed on a subscription's queue to end iteration."""


_CLOSE_SENTINEL = _Closed()


class Subscription:
    """A single subscriber's view onto an :class:`EventBus`.

    Iterable (blocks on its queue until an event arrives or the
    subscription is closed) and usable as a context manager (closes on
    exit). Closing is safe to call from another thread and is idempotent.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._queue: queue.Queue[Event | _Closed] = queue.Queue()
        self._drained = False

    def _deliver(self, event: Event) -> None:
        self._queue.put(event)

    def close(self) -> None:
        """Unsubscribe from the bus and end iteration.

        Safe to call more than once and from any thread; only the call
        that actually removes this subscription from the bus delivers the
        close sentinel.
        """
        if self._bus._unsubscribe(self):
            self._queue.put(_CLOSE_SENTINEL)

    def __iter__(self) -> Iterator[Event]:
        """Yield events until the subscription closes, then stop.

        A *second* iteration (after the close sentinel was already
        consumed once) returns immediately instead of blocking forever:
        once closed, the bus has already unsubscribed this queue and will
        never deliver another sentinel, so a second ``self._queue.get()``
        would otherwise hang with no timeout and no way to recover short of
        killing the process -- surprising for e.g. a caller that iterates a
        :class:`~composeai.runs.RunStream` twice, or breaks out of a ``for``
        loop and later re-enters one expecting to resume.
        """
        if self._drained:
            return
        while True:
            item = self._queue.get()
            if isinstance(item, _Closed):
                self._drained = True
                return
            yield item

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class EventBus:
    """Thread-safe fan-out from publishers to subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[Subscription] = []
        self._lock = threading.Lock()

    def subscribe(self) -> Subscription:
        sub = Subscription(self)
        with self._lock:
            self._subscribers.append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> bool:
        """Remove ``sub`` if present. Returns whether it was present."""
        with self._lock:
            try:
                self._subscribers.remove(sub)
                return True
            except ValueError:
                return False

    def emit(self, event: Event) -> None:
        """Deliver ``event`` to every active subscription. Non-blocking."""
        with self._lock:
            subs = list(self._subscribers)
        for sub in subs:
            sub._deliver(event)


# --- Ambient bus ---

_current_bus: contextvars.ContextVar[EventBus | None] = contextvars.ContextVar(
    "_current_bus", default=None
)


def current_bus() -> EventBus | None:
    """Return the ambient bus for the current context, or ``None``."""
    return _current_bus.get()


@contextmanager
def use_bus(bus: EventBus) -> Iterator[EventBus]:
    """Make ``bus`` the ambient bus for the duration of the ``with`` block."""
    token = _current_bus.set(bus)
    try:
        yield bus
    finally:
        _current_bus.reset(token)


def emit(event: Event) -> None:
    """Publish ``event`` to the ambient bus. No-op when there is none."""
    bus = _current_bus.get()
    if bus is not None:
        bus.emit(event)
