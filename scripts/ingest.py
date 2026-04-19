"""
scripts/ingest.py

Chunks knowledge-base.md using markdown headers, embeds each chunk with
text-embedding-3-small, and upserts into Qdrant.

Run once after starting the stack:
    docker compose exec app python -m scripts.ingest
"""

import os
import asyncio
from openai import AsyncOpenAI
from qdrant_client.models import PointStruct
from app.db.qdrant_client import create_collection_if_not_exists, upsert_points

KB_PATH    = os.path.join(os.path.dirname(__file__), "../data/knowledge-base.md")
EMBED_MODEL = "text-embedding-3-small"



def chunk_markdown(path: str) -> list[dict]:
    """
    Splits markdown into chunks based on H2 (##) and H3 (###) headers.
    Each chunk carries its section/subsection as metadata.
    """
    chunks = []
    current_h2 = ""
    current_h3 = ""
    buffer     = []

    def flush(h2, h3, buf):
        text = "\n".join(buf).strip()
        if text:
            chunks.append({
                "content": text,
                "metadata": {
                    "Header 2": h2,
                    "Header 3": h3,
                }
            })

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()

            if line.startswith("## "):
                flush(current_h2, current_h3, buffer)
                current_h2 = line.lstrip("# ").strip()
                current_h3 = ""
                buffer = []

            elif line.startswith("### "):
                flush(current_h2, current_h3, buffer)
                current_h3 = line.lstrip("# ").strip()
                buffer = []

            else:
                buffer.append(line)

    flush(current_h2, current_h3, buffer)
    return chunks

async def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Adds an 'embedding' field to each chunk."""
    client = AsyncOpenAI()
    texts  = [c["content"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    response = await client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = response.data[i].embedding

    return chunks


async def main():
    print("── ShopWave KB Ingestion ──")


    print("Creating Qdrant collection if needed...")
    create_collection_if_not_exists()


    print(f"Chunking {KB_PATH}...")
    chunks = chunk_markdown(KB_PATH)
    print(f"  → {len(chunks)} chunks created")

    for i, c in enumerate(chunks):
        section = c["metadata"]["Header 2"]
        subsection = c["metadata"]["Header 3"]
        label = f"{section} > {subsection}" if subsection else section
        print(f"  [{i+1}] {label} ({len(c['content'])} chars)")

    chunks = await embed_chunks(chunks)


    print("Upserting into Qdrant...")
    points = [
        PointStruct(
            id=i,
            vector=chunk["embedding"],
            payload={
                "content":  chunk["content"],
                "metadata": chunk["metadata"],
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    upsert_points(points)

    print(f"Done! {len(points)} vectors stored in Qdrant.")

if __name__ == "__main__":
    asyncio.run(main())