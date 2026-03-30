import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Obsidian MCP")


def main() -> None:
    """Run the Obsidian MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()