import glob
import hashlib
import os
import sqlite3

import markdown
import numpy as np
import ollama
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from tqdm import tqdm

load_dotenv(os.path.expanduser("~/.env"))

obsidian_location = os.getenv("OBSIDIAN_DIR", "")

assert obsidian_location, "Obsidian notes not found."

conn = sqlite3.connect(obsidian_location + "/notes.db")

conn.execute("""
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY,
        text TEXT,
        text_hash TEXT UNIQUE,
        embedding BLOB,
        file_name TEXT
    )
""")

mcp = FastMCP(name="Obsidian MCP")


def embedding(prompt: str = Field(description="The prompt to embed.")) -> dict:
    """A wrapper for nomic-embed-text model.

    Args:
        prompt (str): The prompt to embed.
    """
    return np.array(
        ollama.embeddings(
            model="nomic-embed-text",
            prompt=prompt,
        ).embedding,
        dtype=np.float32,
    )


def save_note(
    text: str, text_hash: str, file_name: str, embedding: np.ndarray
) -> None:
    """Save a note to the database.

    Args:
        text (str): The text of the note.
        text_hash (str): The hash of the note's text.
        file_name (str): The name of the file containing the note.
        embedding (np.ndarray): The embedding of the note's text.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO notes (text, text_hash, embedding, file_name)
        VALUES (?, ?, ?, ?)
    """,
        (text, text_hash, embedding.tobytes(), file_name),
    )
    conn.commit()


@mcp.tool(
    name="obsidian_search",
    description="Search for notes with similar content to the prompt using "
    + "cosine similarity between nomic-embed-text embeddings.",
)
def search(prompt: str, top_k: int = 5) -> list:
    """Search for notes with similar content to the prompt.

    Args:
        prompt (str): The prompt to search for.
        top_k (int): The number of top results to return.
    """
    query_embedding = embedding(prompt)
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
    # find all the md files
    files = glob.glob(obsidian_location + "**/*.md", recursive=True)

    # chunk all the text into 5000 character blocks and create a hash
    # hashes are used to check if the text has already been embedded later
    n = 500
    hashes, chunks, file_names = [], [], []
    for i in tqdm(range(len(files)), desc="Processing files"):
        with open(files[i]) as f:
            contents = markdown.markdown(f.read())
        contents = BeautifulSoup(contents, "html.parser").get_text()
        split = [contents[i : i + n] for i in range(0, len(contents), n)]
        chunk_hash = [hashlib.md5(c.encode()).hexdigest() for c in split]
        chunks.extend(split)
        hashes.extend(chunk_hash)
        file_names.extend([files[i]] * len(split))

    for i in tqdm(range(len(chunks)), desc="Embedding notes"):
        chunk = chunks[i]
        text_hash = hashes[i]
        file_name = file_names[i]
        # if the text exists and pass if it does
        exists = (
            conn.execute(
                "SELECT 1 FROM notes WHERE text_hash = ?", (text_hash,)
            ).fetchone()
            is not None
        )
        if not exists:
            # add the note to the database
            save_note(chunk, text_hash, file_name, embedding(chunk))
    mcp.run()


if __name__ == "__main__":
    main()
    #print(search("star formation"))
