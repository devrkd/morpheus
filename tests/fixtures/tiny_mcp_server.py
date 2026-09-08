"""A minimal MCP server over stdio, so the harness is tested against a real one.

Deliberately real rather than mocked: the things that break in MCP wiring are
transport, tool-name prefixing and lifecycle, none of which a mock exercises.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tiny")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
