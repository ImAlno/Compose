"""Scripted MCP server for composeai's test suite (stdio transport).

Run: python tests/fixtures/mcp_fixture_server.py
Tools: echo (text), add (structured result), boom (isError), slow (timeout tests).
Requires the `mcp` package (a composeai dev dependency).
"""

from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("composeai-fixture")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@mcp.tool()
def add(a: int, b: int) -> dict[str, int]:
    """Add two integers and return a structured total."""
    return {"total": a + b}


@mcp.tool()
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


@mcp.tool()
async def slow(seconds: float) -> str:
    """Sleep, then answer.

    ``asyncio.sleep`` (not blocking ``time.sleep``) so the server's event
    loop stays free to handle other in-flight requests concurrently --
    this is what lets a client-side timeout on ``slow`` actually free up
    the bridge for a subsequent call rather than queuing behind it.
    """
    await asyncio.sleep(seconds)
    return "done"


if __name__ == "__main__":
    mcp.run()
