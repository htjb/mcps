import os

import ollama
from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(name="Ollama MCP")


def main() -> None:
    """Run the Ollama MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()