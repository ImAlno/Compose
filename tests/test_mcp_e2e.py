"""E2E tests against the stdio fixture server (tests/fixtures/mcp_fixture_server.py).

Two sections: direct bridge tests for ``composeai.mcp._MCPServer`` (exercising
start_stdio, list_tools_sync, call_tool_sync, close without going through
mcp_tools()/@agent), and tests for the public ``composeai.mcp_tools()`` on
top of it. All gated on the ``mcp`` SDK being installed (composeai[mcp]).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from composeai.errors import ConfigError, MCPToolError  # noqa: E402
from composeai.mcp import _MCPServer  # noqa: E402
from composeai.messages import ToolResultPart  # noqa: E402

_FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_fixture_server.py")


@pytest.fixture(autouse=True)
def _close_all_mcp_servers_after_each_test():
    """Make this file's tests order-independent regardless of run order.

    Servers are per-test (each test builds its own via the ``server``
    fixture or ``_fixture_tools()``), so calling close_all() again in
    teardown is cheap and idempotent -- it's a no-op for servers a test
    already closed itself (e.g. test_close_all_then_call_raises_and_is_idempotent).
    """
    yield
    import composeai.mcp as compose_mcp

    compose_mcp.close_all()


@pytest.fixture
def server():
    srv = _MCPServer("fixture", connect_timeout=10.0)
    srv.start_stdio(sys.executable, [_FIXTURE])
    yield srv
    srv.close()


def test_list_tools_sync_returns_fixture_tools(server):
    tools = server.list_tools_sync()
    names = {t.name for t in tools}
    assert names == {"echo", "add", "boom", "slow"}


def test_call_tool_sync_echo(server):
    # FastMCP infers an outputSchema from `echo`'s `-> str` return
    # annotation and auto-populates structuredContent (wrapping the
    # scalar in {"result": ...}) even though the tool body never builds
    # a dict itself. Per _render_result's text-first contract (AMENDED
    # 2026-07-14 during Task 5) the text block wins over that wrapper
    # noise -- verified against the live fixture server, not merely the
    # unit-level rendering helper.
    assert server.call_tool_sync("echo", {"text": "hi"}, timeout=10.0) == "echo: hi"


def test_call_tool_sync_boom_raises_mcp_tool_error(server):
    with pytest.raises(MCPToolError, match="kaboom"):
        server.call_tool_sync("boom", {}, timeout=10.0)


def test_call_tool_sync_after_close_raises():
    srv = _MCPServer("fixture", connect_timeout=10.0)
    srv.start_stdio(sys.executable, [_FIXTURE])
    srv.close()
    with pytest.raises(MCPToolError, match="closed"):
        srv.call_tool_sync("echo", {"text": "hi"}, timeout=10.0)
    srv.close()  # closing twice is fine


def test_call_tool_timeout_raises_and_cancels(server):
    # The `slow` tool sleeps for 10s server-side; a 0.3s client timeout
    # must raise promptly (proving future.cancel() actually unblocks the
    # waiting thread rather than the call quietly running to completion).
    with pytest.raises(MCPToolError, match="exceeded"):
        server.call_tool_sync("slow", {"seconds": 10}, 0.3)
    # The bridge (loop/thread/session) stays usable after a timed-out call.
    assert "still alive" in server.call_tool_sync("echo", {"text": "still alive"}, None)


def test_start_stdio_twice_raises_config_error(server):
    with pytest.raises(ConfigError, match="already started"):
        server.start_stdio(sys.executable, [_FIXTURE])


def test_connect_failure_raises_config_error():
    server = _MCPServer("bogus", connect_timeout=5.0)
    with pytest.raises(ConfigError):
        server.start_stdio(command="definitely-not-a-real-executable-xyz", args=[], env=None)


def test_connect_hang_raises_config_error_with_message():
    # A "server" that spawns fine but never speaks the MCP protocol hangs
    # inside session.initialize() -- future.result() raises a bare
    # concurrent.futures.TimeoutError with an empty str(), which used to
    # surface to the user as "Failed to connect to MCP server 'x': ".
    # Catching _FutureTimeoutError first must produce a real message.
    server = _MCPServer("hang", connect_timeout=1.0)
    with pytest.raises(ConfigError, match="did not become ready"):
        server.start_stdio(sys.executable, ["-c", "import time; time.sleep(60)"])


# --- E2E for mcp_tools() against the same fixture server ---


def _fixture_tools(**kwargs):
    from composeai import mcp_tools

    return mcp_tools(command=sys.executable, args=[_FIXTURE], **kwargs)


def test_mcp_tools_lists_and_calls():
    tools = _fixture_tools()
    names = sorted(t.name for t in tools)
    assert names == ["add", "boom", "echo", "slow"]
    echo = next(t for t in tools if t.name == "echo")
    assert echo.execute({"text": "hi"}) == "echo: hi"


def test_mcp_structured_result_renders_as_json():
    tools = _fixture_tools()
    add = next(t for t in tools if t.name == "add")
    result = add.execute({"a": 1, "b": 2})
    import json as _json

    assert _json.loads(result) == {"total": 3}


def test_mcp_is_error_raises_mcp_tool_error():
    tools = _fixture_tools()
    boom = next(t for t in tools if t.name == "boom")
    with pytest.raises(MCPToolError, match="kaboom"):
        boom.execute({})


def test_mcp_include_exclude_prefix():
    tools = _fixture_tools(include=["echo", "add"], prefix="fx")
    assert sorted(t.name for t in tools) == ["fx_add", "fx_echo"]
    fx_echo = next(t for t in tools if t.name == "fx_echo")
    assert fx_echo.execute({"text": "y"}) == "echo: y"  # original name on the wire

    fewer = _fixture_tools(exclude=["boom", "slow"])
    assert sorted(t.name for t in fewer) == ["add", "echo"]


def test_mcp_unknown_include_name_raises():
    with pytest.raises(ConfigError, match="echoo"):
        _fixture_tools(include=["echoo"])


def test_mcp_requires_approval_list_sets_spec():
    tools = _fixture_tools(requires_approval=["echo"])
    by_name = {t.name: t for t in tools}
    assert by_name["echo"].spec.requires_approval is True
    assert by_name["add"].spec.requires_approval is False


def test_mcp_tools_validation_errors():
    from composeai import mcp_tools

    with pytest.raises(ConfigError):
        mcp_tools()  # neither transport
    with pytest.raises(ConfigError):
        mcp_tools(command="x", url="http://y")  # both
    with pytest.raises(ConfigError):
        mcp_tools(url="http://y", args=["--x"])  # stdio arg with url


def test_failed_validation_closes_the_server():
    # A post-connect failure (unknown include name, duplicate tool name,
    # etc.) must still close the server -- otherwise the subprocess and its
    # bridge thread (named "composeai-mcp-<label>", see
    # _MCPServer._start_loop_thread) leak with no caller-accessible handle
    # to close them.
    import composeai.mcp as compose_mcp

    live_before = len(compose_mcp._LIVE_SERVERS)
    threads_before = {t.name for t in threading.enumerate()}
    with pytest.raises(ConfigError):
        _fixture_tools(include=["not_a_real_tool"])
    assert len(compose_mcp._LIVE_SERVERS) == live_before
    lingering = {t.name for t in threading.enumerate()} - threads_before
    assert not {n for n in lingering if "composeai-mcp" in n}, lingering


# --- agent integration: approval, timeout, close_all -----------------------


def test_mcp_tool_inside_agent_run():
    from composeai import agent, prompt
    from composeai.testing import FakeModel

    tools = _fixture_tools(include=["echo"])
    model = FakeModel(
        [
            {"tool_calls": [{"name": "echo", "arguments": {"text": "ping"}}]},
            "final answer",
        ]
    )

    @agent(model=model, tools=tools, name="mcp_agent_e2e")
    def mcp_user(q: str) -> str:
        """Use tools."""
        return prompt(q)

    assert mcp_user("go") == "final answer"
    result_part = model.requests[1].messages[-1].parts[0]
    assert isinstance(result_part, ToolResultPart)
    assert result_part.is_error is False
    assert "echo: ping" in result_part.content


def test_mcp_timeout_surfaces_as_error_result():
    from composeai import agent, prompt
    from composeai.testing import FakeModel

    tools = _fixture_tools(include=["slow"], timeout=0.3)
    model = FakeModel(
        [
            {"tool_calls": [{"name": "slow", "arguments": {"seconds": 10}}]},
            "recovered",
        ]
    )

    @agent(model=model, tools=tools, name="mcp_slow_agent")
    def slow_user(q: str) -> str:
        """Use tools."""
        return prompt(q)

    assert slow_user("go") == "recovered"
    result_part = model.requests[1].messages[-1].parts[0]
    assert isinstance(result_part, ToolResultPart)
    assert result_part.is_error is True
    assert "MCPToolError" in result_part.content


def test_mcp_requires_approval_pauses_and_resumes():
    from composeai import agent, prompt, resume
    from composeai.testing import FakeModel

    tools = _fixture_tools(include=["echo"], requires_approval=True)
    model = FakeModel(
        [
            {"tool_calls": [{"name": "echo", "arguments": {"text": "guarded"}}]},
            "approved and done",
        ]
    )

    @agent(model=model, tools=tools, name="mcp_approval_agent")
    def guarded(q: str) -> str:
        """Use tools."""
        return prompt(q)

    run = guarded.run("go")
    assert run.status == "paused"
    resumed = resume(run.id, {"echo": True})
    assert resumed.output == "approved and done"


def test_mcp_tool_inside_arun():
    """Async twin of ``test_mcp_tool_inside_agent_run`` (v0.4.0 Plan B, Task
    8): the same fixture-server tool, called via ``await agent.arun(...)``
    under ``asyncio.run`` -- the MCP bridge's blocking call happens on its
    own dedicated worker thread (see ``composeai.mcp``'s own bridge thread,
    and ``_dispatch.run_stage``'s sync-tool-on-its-own-thread dispatch),
    never the asyncio loop driving this test, so this must complete
    normally with no extra bridging."""
    import asyncio

    from composeai import agent, prompt
    from composeai.testing import FakeModel

    tools = _fixture_tools(include=["echo"])
    model = FakeModel(
        [
            {"tool_calls": [{"name": "echo", "arguments": {"text": "ping"}}]},
            "final answer",
        ]
    )

    @agent(model=model, tools=tools, name="mcp_agent_e2e_arun")
    def mcp_user(q: str) -> str:
        """Use tools."""
        return prompt(q)

    async def drive():
        return await mcp_user.arun("go")

    run = asyncio.run(drive())
    assert run.status == "completed"
    assert run.output == "final answer"
    result_part = model.requests[1].messages[-1].parts[0]
    assert isinstance(result_part, ToolResultPart)
    assert result_part.is_error is False
    assert "echo: ping" in result_part.content


def test_close_all_then_call_raises_and_is_idempotent():
    import composeai.mcp as compose_mcp
    from composeai.errors import MCPToolError

    tools = _fixture_tools(include=["echo"])
    echo = tools[0]
    assert echo.execute({"text": "before"}) == "echo: before"
    compose_mcp.close_all()
    compose_mcp.close_all()  # idempotent
    with pytest.raises(MCPToolError, match="closed"):
        echo.execute({"text": "after"})
