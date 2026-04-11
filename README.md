# MCP Tools

Install with 

```bash
uv tool install .
```

Then run

```bash
claude mcp add ollama-mcp -s user -- ollama-mcp
```

substituting `ollama-mcp` for the appropriate tool.

To install for opencode run
```bash
opencode mcp add
```
and follow the hints.

## Ollama MCP

Allows task to be offloaded to different ollama models. Need an Ollama API key
in `~/.env` under `OLLAMA_API_KEY=...` to use cloud models.

Tool is installed as `ollama-mcp`.

## Obsidian MCP

Perform RAG on obsidian notes and load obsidian notes in to context. Embeddings
are stored in the obsidian vault in an SQLite DB called `notes.db`. Need to add
`OBSIDIAN_DIR=/path/to/obsidian/vault/"` in `~/.env`.

Installed as `obsidian-mcp`.

Needs the `nomic-embed-text` model installed via ollama to embed the text. Run

```bash
ollama pull nomic-embed-text
```
to pull it.