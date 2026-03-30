import glob
import hashlib
import os
import sqlite3

import markdown
import numpy as np
import ollama
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

load_dotenv(os.path.expanduser("~/.env"))

obsidian_location = os.getenv("OBSIDIAN_DIR", "")

assert obsidian_location, "Obsidian notes not found."

conn = sqlite3.connect(obsidian_location + "/notes.db")

conn.execute("""
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY,
        text TEXT,
        text_hash TEXT UNIQUE,
        embedding BLOB
    )
""")

# mcp = FastMCP(name="Obsidian MCP")


def embedding(prompt: str = Field(description="The prompt to embed.")) -> dict:
    """A wrapper for nomic-embed-text model.

    Args:
        prompt (str): The prompt to embed.
    """
    return np.array(ollama.embeddings(
        model="nomic-embed-text",
        prompt=prompt,
    ).embedding)


# Store
def save_note(text, text_hash, embedding):
    conn.execute(
        """
        INSERT OR IGNORE INTO notes (text, text_hash, embedding)
        VALUES (?, ?, ?)
    """,
        (text, text_hash, embedding.tobytes()),
    )
    conn.commit()


# Search
def search(query_embedding, top_k=5):
    rows = conn.execute("SELECT text, embedding FROM notes").fetchall()
    texts = [r[0] for r in rows]
    embeddings = np.stack(
        [np.frombuffer(r[1], dtype=np.float32) for r in rows]
    )

    scores = embeddings @ query_embedding  # cosine if normalised
    top = np.argsort(scores)[::-1][:top_k]
    return [texts[i] for i in top]


def main() -> None:
    """Run the Obsidian MCP server."""
    files = glob.glob(obsidian_location + "**/*.md", recursive=True)

    n = 5000
    hashes, chunks = [], []
    for i in range(len(files)):
        with open(files[i]) as f:
            contents = markdown.markdown(f.read())
        split = [contents[i : i + n] for i in range(0, len(contents), n)]
        chunk_hash = [hashlib.md5(c.encode()).hexdigest() for c in split]
        chunks.extend(split)
        hashes.extend(chunk_hash)

    for chunk, text_hash in zip(chunks, hashes):
        exists = (
            conn.execute(
                "SELECT 1 FROM notes WHERE text_hash = ?", (text_hash,)
            ).fetchone()
            is not None
        )
        if not exists:
            save_note(chunk, text_hash, embedding(chunk))
    # mcp.run()


if __name__ == "__main__":
    main()
