"""MCP (Model Context Protocol) client support: ``mcp_tools()``.

The ``mcp`` SDK is imported only lazily inside this module (install via
``pip install 'composeai[mcp]'``), keeping composeai's core pydantic-only.
The SDK is async-only; each connected server gets a dedicated daemon
thread running its own event loop that owns the transport and session --
per-call ``asyncio.run`` would tear down a stdio session between calls,
and a shared loop would couple unrelated servers' lifetimes. Sync callers
block on ``run_coroutine_threadsafe`` futures.

Pinned ``mcp>=1.27,<2``: SDK v2 is a from-scratch breaking redesign; every
SDK touchpoint lives in this one module so migrating is a contained change.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import threading
from collections.abc import Sequence
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import TYPE_CHECKING, Any

from .errors import ConfigError, MCPToolError
from .models.base import ToolSpec
from .tools import Tool

if TYPE_CHECKING:
    from contextlib import AsyncExitStack

    from mcp import ClientSession

_NO_DESCRIPTION = "(no description provided by the MCP server)"

_LIVE_SERVERS: list[_MCPServer] = []
_ATEXIT_REGISTERED = False


def _require_mcp() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise ConfigError("Install the MCP SDK: pip install 'composeai[mcp]'") from exc


def _render_result(content_blocks: list[Any], structured: dict[str, Any] | None) -> str:
    """Render an MCP tool result to a single string.

    Text-first (AMENDED 2026-07-14 during Task 5, from live SDK behavior):
    text blocks win when any exist -- joined with newlines; any other
    block type (image, audio, resource, ...) is rendered as an explicit
    placeholder rather than silently dropped. Only when there are NO text
    blocks does structured content fall back in (JSON-encoded). Rationale:
    FastMCP auto-wraps a scalar return in ``structuredContent={"result":
    ...}`` for every typed-return tool, so structured-first fed the model
    that wrapper noise instead of the server's intended textual
    rendering -- text blocks are what a conversation-fed LLM should see.
    """
    has_text = any(getattr(block, "type", None) == "text" for block in content_blocks or [])
    if not has_text and structured is not None:
        return json.dumps(structured)
    parts: list[str] = []
    for block in content_blocks or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
        else:
            parts.append(f"[unsupported MCP content block: {block_type}]")
    return "\n".join(parts)


def _spec_from_mcp_tool(
    name: str,
    description: str | None,
    input_schema: dict[str, Any],
    *,
    requires_approval: bool,
) -> ToolSpec:
    """Map one MCP ``Tool`` (already duck-typed apart by the caller) to a :class:`ToolSpec`.

    ``strict=False`` always: an MCP server's JSON schema is out of
    composeai's control and generally won't satisfy provider strict-mode
    constraints (e.g. ``additionalProperties: false`` on every nested
    object), so it's passed through unmodified rather than sealed.
    """
    return ToolSpec(
        name=name,
        description=description or _NO_DESCRIPTION,
        input_schema=input_schema,
        requires_approval=requires_approval,
        strict=False,
    )


def close_all() -> None:
    """Close every currently-live MCP server bridge. Idempotent; never raises.

    A public-ish escape hatch -- ``mcp_tools()`` (Task 5) registers this
    once with :mod:`atexit` so servers spawned via stdio don't outlive the
    interpreter, but callers may also invoke it directly (e.g. between
    test cases).
    """
    for server in list(_LIVE_SERVERS):
        server.close()


class _MCPServer:
    """Sync bridge to one MCP server, owning a dedicated background event loop.

    The loop and its thread are started by :meth:`start_stdio` or
    :meth:`start_http` (not ``__init__``) since connecting is the part
    that can fail, and a partially-constructed server should never end up
    in :data:`_LIVE_SERVERS`. All SDK objects (the transport context, the
    :class:`mcp.ClientSession`) live entirely on the bridge thread; every
    public method below schedules a coroutine onto that loop via
    ``asyncio.run_coroutine_threadsafe`` and blocks the calling thread on
    the resulting future.
    """

    def __init__(self, label: str, connect_timeout: float) -> None:
        self._label = label
        self._connect_timeout = connect_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._closed = False

    # -- connecting -----------------------------------------------------

    def _start_loop_thread(self) -> None:
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_run, name=f"composeai-mcp-{self._label}", daemon=True
        )
        thread.start()
        ready.wait()
        self._loop = loop
        self._thread = thread

    def _connect(self, coro_factory: Any) -> None:
        """Run ``coro_factory()`` (an async connect coroutine) on the bridge loop.

        On any failure, tears down whatever was opened and joins the
        thread before raising :class:`ConfigError` -- a server that fails
        to connect never lands in :data:`_LIVE_SERVERS` and leaves no
        thread behind.

        Guards against a second ``start_stdio``/``start_http`` call on the
        same instance: without this, re-starting an already-started (or
        already-closed) server would orphan the first loop/thread/session
        rather than reusing or rejecting it.
        """
        if self._thread is not None or self._session is not None or self._closed:
            raise ConfigError(f"MCP server {self._label!r} is already started")
        self._start_loop_thread()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro_factory(), self._loop)
        try:
            future.result(timeout=self._connect_timeout)
        except _FutureTimeoutError:
            future.cancel()
            self._teardown_after_failed_connect()
            raise ConfigError(
                f"MCP server {self._label!r} did not become ready within "
                f"connect_timeout={self._connect_timeout}s"
            ) from None
        except Exception as exc:
            future.cancel()
            self._teardown_after_failed_connect()
            raise ConfigError(
                f"Failed to connect to MCP server {self._label!r}: {exc}"
            ) from exc
        _LIVE_SERVERS.append(self)

    def _aclose_stop_and_join(self, *, suppress_errors: bool) -> None:
        """Close the transport stack, stop the bridge loop, and join its thread.

        Shared by :meth:`_teardown_after_failed_connect` and :meth:`close`,
        which differ only in whether the loop-stop/join steps swallow
        exceptions (``close()``'s "idempotent; never raises" contract) and
        in what they do with ``_LIVE_SERVERS`` afterwards.
        """
        loop = self._loop
        stack = self._stack
        if loop is not None and stack is not None:

            async def _aclose() -> None:
                await stack.aclose()

            try:
                fut = asyncio.run_coroutine_threadsafe(_aclose(), loop)
                fut.result(timeout=5)
            except Exception:
                pass
        if loop is not None:
            if suppress_errors:
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except Exception:
                    pass
            else:
                loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            if suppress_errors:
                try:
                    self._thread.join(timeout=5)
                except Exception:
                    pass
            else:
                self._thread.join(timeout=5)

    def _teardown_after_failed_connect(self) -> None:
        self._aclose_stop_and_join(suppress_errors=False)
        self._closed = True

    def start_stdio(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> _MCPServer:
        """Connect over stdio, spawning ``command`` as a subprocess."""
        _require_mcp()

        async def _do_connect() -> None:
            from contextlib import AsyncExitStack

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            stack = AsyncExitStack()
            self._stack = stack
            params = StdioServerParameters(command=command, args=args or [], env=env)
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session

        self._connect(_do_connect)
        return self

    def start_http(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> _MCPServer:
        """Connect over streamable HTTP."""
        _require_mcp()

        async def _do_connect() -> None:
            from contextlib import AsyncExitStack

            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            stack = AsyncExitStack()
            self._stack = stack
            # streamablehttp_client yields a 3-tuple: (read, write,
            # get_session_id_callback) -- unlike stdio_client's 2-tuple.
            read, write, _get_session_id = await stack.enter_async_context(
                streamablehttp_client(url, headers=headers)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session

        self._connect(_do_connect)
        return self

    # -- listing / calling ------------------------------------------------

    async def _list_tools(self) -> list[Any]:
        assert self._session is not None
        tools: list[Any] = []
        cursor: str | None = None
        while True:
            # `ClientSession.list_tools(cursor=...)` is confirmed present on
            # the installed SDK (mcp 1.28.1); this loop follows `nextCursor`
            # until the server reports no further page.
            result = await self._session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if not cursor:
                break
        return tools

    def list_tools_sync(self) -> list[Any]:
        if self._closed:
            raise MCPToolError(
                f"MCP server {self._label!r} is closed (close_all() or interpreter "
                "shutdown already ran); rebuild tools with mcp_tools()"
            )
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(self._list_tools(), self._loop)
        try:
            return future.result(timeout=self._connect_timeout)
        except _FutureTimeoutError:
            future.cancel()
            raise MCPToolError(
                f"Listing tools on MCP server {self._label!r} exceeded "
                f"timeout={self._connect_timeout}s"
            ) from None
        except Exception as exc:
            raise MCPToolError(
                f"Listing tools on MCP server {self._label!r} failed: {exc}"
            ) from exc

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        assert self._session is not None
        result = await self._session.call_tool(name, arguments)
        structured = getattr(result, "structuredContent", None)
        rendered = _render_result(result.content, structured)
        if result.isError:
            # The server already formats its own message into a text block
            # (FastMCP: "Error executing tool {name}: {exc}") -- surface it
            # verbatim rather than re-wrapping it a second time.
            raise MCPToolError(rendered)
        return rendered

    def call_tool_sync(
        self, name: str, arguments: dict[str, Any], timeout: float | None
    ) -> str:
        if self._closed:
            raise MCPToolError(
                f"MCP server {self._label!r} is closed (close_all() or interpreter "
                "shutdown already ran); rebuild tools with mcp_tools()"
            )
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(self._call_tool(name, arguments), self._loop)
        try:
            result = future.result(
                timeout=timeout if timeout is not None else self._connect_timeout
            )
        except _FutureTimeoutError:
            future.cancel()
            raise MCPToolError(
                f"MCP tool {name!r} on server {self._label!r} exceeded "
                f"timeout={timeout if timeout is not None else self._connect_timeout}s"
            ) from None
        except MCPToolError:
            raise
        except Exception as exc:
            raise MCPToolError(f"MCP tool {name!r} failed: {exc}") from exc
        return result

    # -- closing ----------------------------------------------------------

    def close(self) -> None:
        """Tear down the connection and its bridge thread. Idempotent; never raises."""
        if self._closed:
            return
        self._closed = True
        self._aclose_stop_and_join(suppress_errors=True)
        if self in _LIVE_SERVERS:
            try:
                _LIVE_SERVERS.remove(self)
            except ValueError:
                pass


def mcp_tools(
    *,
    command: str | None = None,
    args: Sequence[str] = (),
    env: dict[str, str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    prefix: str | None = None,
    requires_approval: bool | Sequence[str] = False,
    timeout: float | None = None,
    connect_timeout: float = 30.0,
) -> list[Tool]:
    """Connect to one MCP server and expose its tools as composeai :class:`Tool` objects.

    Exactly one transport: ``command`` (stdio -- spawns ``command`` with
    ``args``/``env``) or ``url`` (streamable HTTP, with optional static
    ``headers`` such as an auth token). Mixing transport-specific kwargs
    across transports (``headers`` with ``command``; ``env``/``args`` with
    ``url``) is a :class:`~composeai.errors.ConfigError`, same as omitting
    or supplying both transports.

    Eager: connects, calls ``initialize()``, and lists every tool the
    server offers (following pagination) before returning -- called at
    module level this is an import-time connect, the trade-off of the
    module-level ``@agent`` idiom the same way model clients are. A
    server that returns two tools with the same name is a
    :class:`ConfigError` (fail loudly rather than silently shadow one).

    ``include``/``exclude`` filter by the server's original tool names
    (before ``prefix`` is applied); an unknown name in either -- or in
    ``requires_approval`` when given as a list -- is a ``ConfigError``
    naming the server's actual tool names, so a typo fails at
    construction instead of silently matching nothing. ``prefix`` renames
    each selected tool's spec to ``f"{prefix}_{name}"`` for the model, but
    execution still calls the server with the tool's original name --
    the prefix only exists to avoid collisions across multiple servers'
    tools in one ``@agent(tools=[...])`` list.

    ``requires_approval`` is either ``True`` (every tool from this server
    pauses for approval), ``False`` (default, none do), or a list of the
    server's original tool names that should. It flows straight into
    :attr:`~composeai.models.base.ToolSpec.requires_approval` -- the
    existing HITL pause/resume machinery, no MCP-specific approval
    mechanism.

    ``timeout`` bounds each tool call (``None`` falls back to
    ``connect_timeout``); ``connect_timeout`` bounds spawning/connecting/
    ``initialize()``/``list_tools()``. The connection is process-lifetime:
    it is registered so :func:`close_all` (also wired into ``atexit``,
    once, on this function's first successful call) tears it down at
    interpreter shutdown, but a resumed flow in a fresh process must call
    ``mcp_tools()`` again -- connections don't survive across processes.
    """
    global _ATEXIT_REGISTERED

    if (command is None) == (url is None):
        raise ConfigError(
            "mcp_tools() requires exactly one of command= (stdio) or url= "
            "(streamable HTTP)"
        )
    if command is not None and headers is not None:
        raise ConfigError(
            "mcp_tools(headers=...) is only valid with url= (streamable HTTP), "
            "not command= (stdio)"
        )
    if url is not None and (env is not None or args):
        raise ConfigError(
            "mcp_tools(env=..., args=...) are only valid with command= (stdio), "
            "not url= (streamable HTTP)"
        )

    _require_mcp()

    if command is not None:
        label = " ".join([command, *args])
        server = _MCPServer(label, connect_timeout)
        server.start_stdio(command, list(args), env)
    else:
        assert url is not None
        label = url
        server = _MCPServer(label, connect_timeout)
        server.start_http(url, headers)

    try:
        mcp_tool_list = server.list_tools_sync()

        actual_names = [mcp_tool.name for mcp_tool in mcp_tool_list]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in actual_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ConfigError(
                f"MCP server {label!r} returned duplicate tool name(s): "
                f"{sorted(duplicates)!r}"
            )
        actual_set = set(actual_names)

        if include is not None:
            unknown_include = sorted(set(include) - actual_set)
            if unknown_include:
                raise ConfigError(
                    f"mcp_tools(include=...) name(s) not found on server {label!r}: "
                    f"{unknown_include!r} (actual tool names: {sorted(actual_set)!r})"
                )
        unknown_exclude = sorted(set(exclude) - actual_set)
        if unknown_exclude:
            raise ConfigError(
                f"mcp_tools(exclude=...) name(s) not found on server {label!r}: "
                f"{unknown_exclude!r} (actual tool names: {sorted(actual_set)!r})"
            )
        if not isinstance(requires_approval, bool):
            unknown_approval = sorted(set(requires_approval) - actual_set)
            if unknown_approval:
                raise ConfigError(
                    f"mcp_tools(requires_approval=...) name(s) not found on server "
                    f"{label!r}: {unknown_approval!r} "
                    f"(actual tool names: {sorted(actual_set)!r})"
                )

        include_set = set(include) if include is not None else None
        exclude_set = set(exclude)
        selected = [
            mcp_tool
            for mcp_tool in mcp_tool_list
            if (include_set is None or mcp_tool.name in include_set)
            and mcp_tool.name not in exclude_set
        ]

        built: list[Tool] = []
        for mcp_tool in selected:
            original = mcp_tool.name
            display = f"{prefix}_{original}" if prefix else original
            approval = (
                requires_approval is True
                or (not isinstance(requires_approval, bool) and original in requires_approval)
            )
            spec = _spec_from_mcp_tool(
                display,
                getattr(mcp_tool, "description", None),
                getattr(mcp_tool, "inputSchema", {}) or {},
                requires_approval=approval,
            )

            def _executor(arguments: dict[str, Any], _name: str = original) -> str:
                return server.call_tool_sync(_name, arguments, timeout)

            built.append(Tool(spec=spec, executor=_executor))
    except BaseException:
        server.close()
        raise

    if not _ATEXIT_REGISTERED:
        atexit.register(close_all)
        _ATEXIT_REGISTERED = True

    return built
