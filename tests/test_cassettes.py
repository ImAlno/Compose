"""Tests for composeai.testing's cassette record/replay kit (Phase 9).

Cassettes record a resolved model's real responses to a JSON file
(``record_cassette``) and later replay them offline, deterministically,
with no network call and -- crucially for replay -- no provider adapter
ever constructed (``replay_cassette``). The ``cassette`` pytest fixture
(re-exported by ``tests/conftest.py`` so it's requestable by name in any
test module -- see that file) picks the right mode automatically: record
when ``COMPOSE_RECORD=1``, replay when the file already exists, otherwise
pass through live.

Every test here drives real ``@agent`` runs against ``FakeModel`` (never a
real provider) -- the "no real adapter constructed" claim is instead
proven with a sentinel provider factory swapped in/out via
``monkeypatch.setitem`` on the registry's factory table: registered to
return a ``FakeModel`` while *recording*, then registered to raise while
*replaying* the exact same agent/request -- proving replay's ``resolve()``
never reaches it.
"""

from __future__ import annotations

import json
from typing import TypedDict

import pytest

from composeai import agent
from composeai._encoding import to_jsonable
from composeai.errors import ComposeError, ConfigError
from composeai.messages import Message, StopReason, Usage
from composeai.models import registry
from composeai.models.base import Model, ModelRequest, ModelResponse, ToolSpec
from composeai.testing import (
    FakeModel,
    ReplayModel,
    compute_full_hash,
    compute_message_hash,
    record_cassette,
    replay_cassette,
)


def _req(model: str = "fake/model", system: str | None = None, messages=None) -> ModelRequest:
    return ModelRequest(model=model, messages=messages or [Message.user("hi")], system=system)


def _encode_response(text: str) -> dict:
    response = ModelResponse(
        message=Message.assistant(text),
        stop_reason=StopReason.END_TURN,
        raw_stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
        model_id="fake/model",
    )
    return to_jsonable(response)


# --- record_cassette: file shape + hashes -----------------------------------


def test_record_cassette_writes_version_and_entries(tmp_path):
    path = tmp_path / "c1.json"
    fake = FakeModel(["hello"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    with record_cassette(path):
        greeter.run("world")

    data = json.loads(path.read_text())
    assert data["version"] == 2
    assert len(data["entries"]) == 1
    entry = data["entries"][0]
    assert set(entry) == {"full_hash", "message_hash", "request", "response"}
    assert isinstance(entry["full_hash"], str) and len(entry["full_hash"]) == 64
    assert isinstance(entry["message_hash"], str) and len(entry["message_hash"]) == 64


def test_record_cassette_records_every_call(tmp_path):
    path = tmp_path / "c1b.json"
    fake = FakeModel(["one", "two"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    with record_cassette(path):
        greeter.run("a")
        greeter.run("b")

    data = json.loads(path.read_text())
    assert len(data["entries"]) == 2


def test_record_cassette_response_round_trips_via_replay(tmp_path):
    path = tmp_path / "c2.json"
    fake = FakeModel(["scripted output"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    with record_cassette(path):
        run = greeter.run("world")
    assert run.output == "scripted output"

    with replay_cassette(path):
        replayed = greeter.run("world")
    assert replayed.output == "scripted output"


# --- replay: full_hash match, adapter never constructed ---------------------


def test_replay_never_constructs_real_adapter(tmp_path, monkeypatch):
    path = tmp_path / "c3.json"
    fake = FakeModel(["hi"])

    # While recording: the sentinel provider legitimately resolves to `fake`.
    monkeypatch.setitem(registry._PROVIDER_FACTORIES, "sentinel-1", lambda model_id: fake)

    @agent(model="sentinel-1/whatever")
    def greeter(name: str) -> str:
        """Greet."""
        return name

    with record_cassette(path):
        greeter.run("a")

    # While replaying the *same* agent/request: swap the factory for one that
    # raises if it's ever called -- proving replay's resolve() never reaches it.
    def _boom(model_id: str) -> Model:
        raise AssertionError("real adapter factory must never be called during replay")

    monkeypatch.setitem(registry._PROVIDER_FACTORIES, "sentinel-1", _boom)

    with replay_cassette(path):
        run = greeter.run("a")  # would raise AssertionError if _boom() were called
    assert run.output == "hi"


# --- full_hash -> message_hash fallback -------------------------------------


def test_replay_falls_back_to_message_hash_when_full_hash_differs():
    req = _req()
    entries = [
        {
            "full_hash": "no-such-full-hash",
            "message_hash": compute_message_hash(req.system, req.messages),
            "request": {},
            "response": _encode_response("fallback hit"),
        }
    ]
    replay = ReplayModel(entries)
    resp = replay.complete(req)
    assert resp.message.text == "fallback hit"


# --- ordered consumption of duplicate hashes --------------------------------


def test_replay_consumes_duplicate_full_hash_entries_in_order():
    req = _req()
    full = compute_full_hash(req)
    entries = [
        {
            "full_hash": full,
            "message_hash": "irrelevant",
            "request": {},
            "response": _encode_response("first"),
        },
        {
            "full_hash": full,
            "message_hash": "irrelevant",
            "request": {},
            "response": _encode_response("second"),
        },
    ]
    replay = ReplayModel(entries)
    assert replay.complete(req).message.text == "first"
    assert replay.complete(req).message.text == "second"


# --- miss -> ComposeError ----------------------------------------------------


def test_replay_miss_raises_compose_error_naming_the_request():
    replay = ReplayModel([])
    with pytest.raises(ComposeError, match="[Cc]assette"):
        replay.complete(_req())


def test_replay_miss_after_bucket_exhausted_raises():
    req = _req()
    full = compute_full_hash(req)
    entries = [
        {
            "full_hash": full,
            "message_hash": "x",
            "request": {},
            "response": _encode_response("only"),
        }
    ]
    replay = ReplayModel(entries)
    replay.complete(req)  # consumes the only entry
    with pytest.raises(ComposeError):
        replay.complete(req)


# --- streaming replay: word-level deltas ------------------------------------


def test_replay_stream_synthesizes_word_deltas():
    req = _req()
    entries = [
        {
            "full_hash": compute_full_hash(req),
            "message_hash": compute_message_hash(req.system, req.messages),
            "request": {},
            "response": _encode_response("hello there"),
        }
    ]
    replay = ReplayModel(entries)
    events = list(replay.stream(req))
    assert events[-1].kind == "response_done"
    assert events[-1].response is not None
    deltas = [e.text for e in events[:-1] if e.kind == "text_delta"]
    assert deltas == ["hello", " ", "there"]


# --- cassette fixture matrix --------------------------------------------------


def test_cassette_fixture_records_when_compose_record_env_set(tmp_path, monkeypatch, cassette):
    monkeypatch.setenv("COMPOSE_RECORD", "1")
    path = tmp_path / "fixture-record.json"
    fake = FakeModel(["recorded"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    with cassette(path):
        run = greeter.run("x")
    assert run.output == "recorded"
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data["entries"]) == 1


def test_cassette_fixture_replays_when_file_exists(tmp_path, monkeypatch, cassette):
    monkeypatch.delenv("COMPOSE_RECORD", raising=False)
    path = tmp_path / "fixture-replay.json"
    fake = FakeModel(["to be recorded"])

    monkeypatch.setitem(registry._PROVIDER_FACTORIES, "sentinel-2", lambda model_id: fake)

    @agent(model="sentinel-2/whatever")
    def greeter(name: str) -> str:
        """Greet."""
        return name

    with record_cassette(path):
        greeter.run("x")

    def _boom(model_id: str) -> Model:
        raise AssertionError("must not construct a real adapter")

    monkeypatch.setitem(registry._PROVIDER_FACTORIES, "sentinel-2", _boom)

    with cassette(path):
        run = greeter.run("x")
    assert run.output == "to be recorded"


def test_cassette_fixture_live_passthrough_when_no_file_and_no_record_env(
    tmp_path, monkeypatch, cassette
):
    monkeypatch.delenv("COMPOSE_RECORD", raising=False)
    path = tmp_path / "does-not-exist.json"
    fake = FakeModel(["live"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    with cassette(path):
        run = greeter.run("x")
    assert run.output == "live"
    assert not path.exists()


# --- full_hash: provider + full tool spec (capstone fix wave C) -------------


def test_compute_full_hash_differs_by_provider():
    """Regression: `full_hash` used to hash only `request.model` (already
    the *bare* id -- the registry strips the provider prefix before
    building a `ModelRequest`), so two providers sharing a bare model-id
    string hashed identically. `ModelRequest.provider` now carries the
    label alongside the bare id specifically so this can't happen."""
    req_a = ModelRequest(
        model="shared-id", messages=[Message.user("hi")], provider="anthropic"
    )
    req_b = ModelRequest(
        model="shared-id", messages=[Message.user("hi")], provider="openai-compatible"
    )
    assert compute_full_hash(req_a) != compute_full_hash(req_b)


def test_compute_full_hash_same_provider_and_request_matches():
    req_a = ModelRequest(model="shared-id", messages=[Message.user("hi")], provider="anthropic")
    req_b = ModelRequest(model="shared-id", messages=[Message.user("hi")], provider="anthropic")
    assert compute_full_hash(req_a) == compute_full_hash(req_b)


def test_compute_full_hash_differs_by_tool_schema_not_just_name():
    """Regression: `full_hash` used to hash only `sorted(t.name for t in
    request.tools)` -- two same-named tools with different input schemas
    hashed identically, which (given the process-wide cache/cassette) could
    let one agent's cached response poison another's differently-shaped
    tool call."""
    tool_a = ToolSpec(
        name="search",
        description="d",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    tool_b = ToolSpec(
        name="search",
        description="d",
        input_schema={"type": "object", "properties": {"q": {"type": "integer"}}},
    )
    req_a = ModelRequest(model="m", messages=[Message.user("hi")], tools=[tool_a])
    req_b = ModelRequest(model="m", messages=[Message.user("hi")], tools=[tool_b])
    assert compute_full_hash(req_a) != compute_full_hash(req_b)


def test_compute_full_hash_differs_by_requires_approval():
    tool_a = ToolSpec(name="search", description="d", input_schema={}, requires_approval=False)
    tool_b = ToolSpec(name="search", description="d", input_schema={}, requires_approval=True)
    req_a = ModelRequest(model="m", messages=[Message.user("hi")], tools=[tool_a])
    req_b = ModelRequest(model="m", messages=[Message.user("hi")], tools=[tool_b])
    assert compute_full_hash(req_a) != compute_full_hash(req_b)


# --- cassette version bump + refusal (capstone fix wave C) ------------------


def test_replay_cassette_refuses_v1_file_with_clear_message(tmp_path):
    path = tmp_path / "old-v1.json"
    path.write_text(json.dumps({"version": 1, "entries": []}))
    with pytest.raises(ConfigError, match="version"):
        with replay_cassette(path):
            pass  # pragma: no cover -- must never get here


def test_replay_cassette_refuses_file_with_no_version_field(tmp_path):
    path = tmp_path / "no-version.json"
    path.write_text(json.dumps({"entries": []}))
    with pytest.raises(ConfigError):
        with replay_cassette(path):
            pass  # pragma: no cover


# --- record/replay cassette: nesting is rejected, not silently corrupted ----


def test_record_cassette_nested_raises_config_error(tmp_path):
    with record_cassette(tmp_path / "outer.json"):
        with pytest.raises(ConfigError):
            with record_cassette(tmp_path / "inner.json"):
                pass  # pragma: no cover
    # Outer context still cleans up normally afterward -- resolve is
    # restored to a plain, unintercepted callable, not left stuck wrapped.
    assert not getattr(registry.resolve, "_compose_cassette_active", False)


def test_replay_cassette_nested_inside_record_cassette_raises_config_error(tmp_path):
    outer = tmp_path / "outer2.json"
    with record_cassette(outer):
        pass
    with record_cassette(outer):
        with pytest.raises(ConfigError):
            with replay_cassette(outer):
                pass  # pragma: no cover


def test_cassette_context_reusable_sequentially_after_nesting_attempt_failed(tmp_path):
    """A failed (rejected) nesting attempt must not corrupt the outer
    context's ability to clean up, nor leave registry.resolve intercepted
    afterward."""
    path = tmp_path / "seq.json"
    fake = FakeModel(["ok"])

    @agent(model=fake)
    def greeter(name: str) -> str:
        """Greet."""
        return name

    with record_cassette(path):
        with pytest.raises(ConfigError):
            with record_cassette(tmp_path / "other.json"):
                pass
        greeter.run("x")  # outer context still functions after the failed nesting attempt

    # registry.resolve is restored to a plain, unwrapped callable afterward.
    assert not getattr(registry.resolve, "_compose_cassette_active", False)

    # And a fresh cassette context afterward works normally (not "stuck open").
    with replay_cassette(path):
        run = greeter.run("x")
    assert run.output == "ok"


# --- 0.6.0 request-config fields vs hashing ------------------------------

_GOLDEN_050_HASH = "ae33af6a7ee6f8ddc4462301410e89375f6937d61714ac46ad7c6a18302548fc"


class _ReqKwargs(TypedDict):
    model: str
    messages: list[Message]
    system: str
    provider: str


def _req_kwargs() -> _ReqKwargs:
    return {
        "model": "claude-sonnet-5",
        "messages": [Message.user("hi")],
        "system": "be terse",
        "provider": "anthropic",
    }


def test_full_hash_unchanged_from_050_at_default_fields():
    assert compute_full_hash(ModelRequest(**_req_kwargs())) == _GOLDEN_050_HASH


def test_prompt_cache_never_affects_hash():
    base = ModelRequest(**_req_kwargs())
    cached = ModelRequest(**_req_kwargs(), prompt_cache=True)
    assert compute_full_hash(base) == compute_full_hash(cached)


def test_thinking_and_effort_change_hash_only_when_set():
    base = compute_full_hash(ModelRequest(**_req_kwargs()))
    thinking = compute_full_hash(ModelRequest(**_req_kwargs(), thinking=True))
    thinking_off = compute_full_hash(ModelRequest(**_req_kwargs(), thinking=False))
    effort = compute_full_hash(ModelRequest(**_req_kwargs(), effort="high"))
    assert thinking != base
    assert thinking_off != base
    assert thinking_off != thinking
    assert effort != base
    assert effort != thinking
