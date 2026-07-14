# MCP

`compose.mcp_tools(...)` connects to one [Model Context Protocol](https://modelcontextprotocol.io) server and turns every tool it offers into ordinary composeai `Tool` objects — pass the result straight into `@agent(tools=...)` alongside (or instead of) `@compose.tool`-decorated functions.

## Connecting

Two transports, mutually exclusive: `command=` (stdio — spawns a subprocess) or `url=` (streamable HTTP). A filesystem server over stdio:

```python
import composeai as compose

notes_tools = compose.mcp_tools(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/notes"],
)


@compose.agent(model="anthropic/claude-sonnet-5", tools=notes_tools)
def notes_assistant(question: str) -> str:
    """You help answer questions about the files in the notes directory."""
    return compose.prompt(question)
```

`mcp_tools()` is eager: it connects, calls `initialize()`, and lists every tool the server offers (following pagination) before returning — there's no lazy handle. Called at module level, as above, that's an import-time connect, the same trade-off the module-level `@agent` idiom already makes for model clients.

For streamable HTTP, `url=` replaces `command=`/`args=`/`env=`, and `headers=` carries static headers such as an auth token:

```python
weather_tools = compose.mcp_tools(
    url="https://mcp.example.com/mcp",
    headers={"Authorization": "Bearer sk-..."},
)
```

Mixing transport-specific kwargs across transports is a `ConfigError`: `headers=` with `command=`, or `env=`/`args=` with `url=` — as is omitting or supplying both `command=` and `url=`.

## Using the tools

An MCP tool is indistinguishable from a local `@compose.tool` once built — same `Tool` type, same place in the `tools=` list, same loop:

```python
@compose.tool
def count_words(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text whose words should be counted.
    """
    return len(text.split())


@compose.agent(model="anthropic/claude-sonnet-5", tools=[count_words, *notes_tools])
def researcher(question: str) -> str:
    """Answer questions, using local and MCP tools as needed."""
    return compose.prompt(question)
```

Each tool's description and JSON Schema come straight from the server's own `list_tools()` response (a missing description is replaced with a placeholder noting the server didn't provide one); the schema is never sealed to strict mode the way `@compose.tool`'s is, since a server's schema is outside composeai's control.

## Selecting and naming

`include=`/`exclude=` filter by the server's original tool names, evaluated before `prefix=` is applied:

```python
tools = compose.mcp_tools(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/notes"],
    include=["read_file", "list_directory"],
    prefix="fs",
)
```

An unknown name in `include=`/`exclude=` (or in `requires_approval=` when given as a list, below) raises `ConfigError` naming the server's actual tool names, so a typo fails at construction instead of silently matching nothing. A server that returns two tools with the same name is also a `ConfigError` — fail loudly rather than silently shadow one.

`prefix=` renames each selected tool's spec to `f"{prefix}_{name}"` for the model — above, `read_file` becomes `fs_read_file` — but execution still calls the server with the tool's original name; the prefix exists purely to avoid name collisions when combining multiple servers' tools in one `tools=[...]` list:

```python
fs_tools = compose.mcp_tools(command="npx", args=[...], prefix="fs")
db_tools = compose.mcp_tools(command="npx", args=[...], prefix="db")

@compose.agent(model="anthropic/claude-sonnet-5", tools=[*fs_tools, *db_tools])
def assistant(question: str) -> str: ...
```

## Approval

`requires_approval=` is `True` (every tool from this server pauses for approval), `False` (the default), or a list of the server's original tool names that should. It flows straight into `ToolSpec.requires_approval` — the same human-in-the-loop pause/resume machinery `@compose.tool(requires_approval=True)` uses, with no MCP-specific approval mechanism; see [flows](flows.md)'s human-in-the-loop section for the full story.

```python
tools = compose.mcp_tools(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/notes"],
    requires_approval=["write_file"],
)


@compose.agent(model="anthropic/claude-sonnet-5", tools=tools, name="notes_editor")
def notes_editor(instruction: str) -> str:
    """Edit notes as instructed."""
    return compose.prompt(instruction)


run = notes_editor.run("Update the meeting notes with today's action items")
# run.status == "paused" if the model called write_file
resumed = compose.resume(run.id, answers={"write_file": True})
```

Same shorthand as a local tool's approval gate: the bare tool name (`"write_file"`) resolves to the reserved interrupt id `tool:write_file:{call_id}` when exactly one pending interrupt matches. `answers={"write_file": False}` denies the call instead — the model sees `"denied by user"` and carries on.

## Timeouts

`timeout=` bounds each individual tool call (`None`, the default, falls back to `connect_timeout=`); `connect_timeout=` (default `30.0` seconds) bounds each phase of setup separately, not their sum — one full `connect_timeout=` budget for spawning the subprocess or opening the HTTP connection plus `initialize()`, and a second, separately-timed `connect_timeout=` budget for the initial `list_tools()` pagination. A server that's slow in both phases can take up to roughly `2 × connect_timeout` end-to-end:

```python
tools = compose.mcp_tools(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/notes"],
    timeout=10.0,
    connect_timeout=15.0,
)
```

A tool call that exceeds `timeout=` (or otherwise fails — the server reported `isError`, the transport broke mid-call) raises `MCPToolError` internally, which surfaces to the model exactly like any tool-body exception: an `is_error` tool result the model can react to, never a run abort. The agent keeps running.

## Lifecycle & caveats

- **Process-lifetime connections.** Each `mcp_tools()` call opens one connection that lives for the process, on a dedicated background thread — there's no per-call reconnect. Nothing closes it for you mid-run.
- **`composeai.mcp.close_all()`** closes every currently-live MCP server connection; idempotent and never raises. `mcp_tools()` registers it with `atexit` on its first successful call, so stdio subprocesses don't outlive the interpreter, but call it directly too (e.g. between test cases, or to release servers early).
- **Import-time connect.** A module-level `mcp_tools()` call (the natural home for the module-level `@agent` idiom) connects at import time, the same trade-off model-string resolution already makes — importing the module requires the server to be reachable.
- **A resumed flow needs the server reachable again.** The connection doesn't survive across processes: a paused run resumed in a fresh process re-executes the module top level, so `mcp_tools()` reconnects from scratch — if the server isn't reachable at resume time, that reconnect fails.
- **The tool list is a snapshot.** `mcp_tools()` lists tools once, at call time. If the server adds, removes, or changes tools afterward, the `Tool` objects already built don't see it — call `mcp_tools()` again to pick up changes.

## Install

`pip install 'composeai[mcp]'` (pinned `mcp>=1.27,<2`, since SDK v2 is a from-scratch breaking redesign still to land).

## See also

[agents](agents.md) covers the `tools=` list and the `@compose.tool` decorator MCP tools sit alongside; [flows](flows.md) covers `resume()` and the human-in-the-loop pause/resume mechanism `requires_approval=` reuses; [testing](testing.md) covers `FakeModel` for testing an agent's tool-calling behavior with no network, MCP or otherwise.
