import threading
import time

import pytest

from composeai.events import Event, EventBus, current_bus, emit, use_bus

# --- Event dataclass ---


def test_event_defaults():
    before = time.time()
    e = Event(kind="text_delta")
    after = time.time()
    assert e.kind == "text_delta"
    assert before <= e.ts <= after
    assert e.trace_id is None
    assert e.span_id is None
    assert e.name is None
    assert e.text is None
    assert e.data is None


def test_event_is_frozen():
    e = Event(kind="text_delta")
    with pytest.raises(AttributeError):
        e.text = "x"  # type: ignore[misc]


def test_event_has_slots_no_dict():
    e = Event(kind="text_delta")
    assert not hasattr(e, "__dict__")


def test_event_carries_payload_fields():
    e = Event(
        kind="span_started",
        trace_id="t1",
        span_id="s1",
        name="agent",
        data={"kind": "agent", "status": "ok"},
    )
    assert e.trace_id == "t1"
    assert e.span_id == "s1"
    assert e.name == "agent"
    assert e.data == {"kind": "agent", "status": "ok"}


# --- EventBus fan-out ---


def test_bus_emits_to_two_subscribers():
    bus = EventBus()
    sub1 = bus.subscribe()
    sub2 = bus.subscribe()
    event = Event(kind="text_delta", text="hi")

    bus.emit(event)
    sub1.close()
    sub2.close()

    assert list(sub1) == [event]
    assert list(sub2) == [event]


def test_bus_emit_with_no_subscribers_does_not_raise():
    bus = EventBus()
    bus.emit(Event(kind="text_delta"))  # no-op, must not raise


def test_close_ends_iteration_with_no_events():
    bus = EventBus()
    sub = bus.subscribe()
    sub.close()
    assert list(sub) == []


def test_close_is_idempotent():
    bus = EventBus()
    sub = bus.subscribe()
    sub.close()
    sub.close()  # must not raise
    assert list(sub) == []


def test_close_from_another_thread_ends_iteration():
    bus = EventBus()
    sub = bus.subscribe()

    def closer():
        time.sleep(0.05)
        sub.close()

    thread = threading.Thread(target=closer)
    thread.start()

    received = list(sub)  # blocks until sentinel arrives from closer thread
    thread.join()

    assert received == []


def test_subscription_is_context_manager_and_closes_on_exit():
    bus = EventBus()
    with bus.subscribe() as sub:
        bus.emit(Event(kind="text_delta", text="in-context"))

    received = list(sub)
    assert [e.text for e in received] == ["in-context"]


def test_closed_subscription_stops_receiving_new_events():
    bus = EventBus()
    sub1 = bus.subscribe()
    sub2 = bus.subscribe()
    sub1.close()

    bus.emit(Event(kind="text_delta", text="after-close"))
    sub2.close()

    received2 = list(sub2)
    assert [e.text for e in received2] == ["after-close"]
    # sub1 only ever gets the close sentinel, no events emitted after close
    assert list(sub1) == []


def test_close_removes_subscription_from_bus_internal_list():
    bus = EventBus()
    sub = bus.subscribe()
    assert sub in bus._subscribers
    sub.close()
    assert sub not in bus._subscribers


def test_iterating_a_closed_subscription_a_second_time_returns_immediately():
    """Regression: iterating a Subscription to completion once (through the
    close sentinel), then iterating it again, used to block forever --
    the bus had already unsubscribed the queue and the sentinel was already
    consumed, so a second `self._queue.get()` had nothing left to wait for
    and no timeout. A second iteration now returns immediately instead."""
    bus = EventBus()
    sub = bus.subscribe()
    bus.emit(Event(kind="text_delta", text="hello"))
    sub.close()

    first = list(sub)
    assert [e.text for e in first] == ["hello"]

    second = list(sub)  # would hang forever before the fix
    assert second == []


def test_iterating_a_never_touched_but_already_closed_subscription_works():
    """A subscription closed before ever being iterated at all must still
    iterate cleanly the first time (the sentinel is genuinely in the queue
    then) -- the "already drained" fast path only kicks in *after* a full
    iteration has actually consumed the sentinel once."""
    bus = EventBus()
    sub = bus.subscribe()
    sub.close()
    assert list(sub) == []
    assert list(sub) == []  # second iteration, still fine


# --- Ambient bus ---


def test_current_bus_is_none_by_default():
    assert current_bus() is None


def test_ambient_emit_is_noop_without_bus():
    emit(Event(kind="text_delta"))  # must not raise


def test_use_bus_sets_and_resets_current_bus():
    bus = EventBus()
    assert current_bus() is None
    with use_bus(bus):
        assert current_bus() is bus
    assert current_bus() is None


def test_ambient_emit_delivers_within_use_bus():
    bus = EventBus()
    sub = bus.subscribe()
    with use_bus(bus):
        emit(Event(kind="text_delta", text="ambient"))
    sub.close()
    received = list(sub)
    assert [e.text for e in received] == ["ambient"]
