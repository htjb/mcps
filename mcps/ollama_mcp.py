import os

import ollama
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from ollama import ChatResponse, Client, chat
from pydantic import Field

load_dotenv(os.path.expanduser("~/.env"))

ollama_key = os.getenv("OLLAMA_API_KEY", "")

assert ollama_key, "Error: OLLAMA_API_KEY not found. Update .env"

client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + ollama_key},
)

mcp = FastMCP(name="Ollama MCP")


@mcp.tool(name="list_models", description="List available Ollama models")
def list_models() -> str:
    """List available Ollama models.

    Returns:
        str: A string listing the available models.
    """
    local_models = [m.model for m in ollama.list().models]
    cloud_models = [m.model + "-cloud" for m in client.list().models]
    return (
        "Available models: "
        + ", ".join(local_models)
        + "; Cloud models: "
        + ", ".join(cloud_models)
    )


@mcp.tool(
    name="query_model",
    description="Send a query to a specific local or "
    + "cloud model and return its response. Use this"
    + "to delegate a question to a more capable model.",
)
def ollama_chat(
    query: str = Field(description="The input query to send to the model."),
    model: str = Field(
        description="The model to use.",
        default="qwen3-coder:480b-cloud",
    ),
) -> str:
    """Chat with Ollama.

    Args:
        query (str): The query to send to Ollama.
        model (str): The model to use for the chat.

    Returns:
        str: The response from Ollama.
    """
    local_models = [m.model for m in ollama.list().models]
    cloud_models = [m.model + "-cloud" for m in client.list().models]
    if model not in [*local_models, *cloud_models]:
        return (
            "Model '{model}' not found."
            + f"Available models: {', '.join(local_models)}; "
            + f"Cloud models: {', '.join(cloud_models)}"
        )

    if "cloud" in model:
        chatter = client.chat
    else:
        chatter = chat
    response: ChatResponse = chatter(
        model=model,
        messages=[{"role": "user", "content": query}],
    )
    if response.message.content is None:
        return "No response from Ollama."
    else:
        return response.message.content


def main() -> None:
    """Run the Ollama MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()
