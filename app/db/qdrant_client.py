from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, QueryRequest
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 1536  # text-embedding-3-small dimension

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_collection_if_not_exists():
    """Create the knowledge base collection in Qdrant if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


def upsert_points(points: list[PointStruct]):
    """Insert vectors into Qdrant."""
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


def search(query_vector: list[float], top_k: int = 3):
    """Search the knowledge base for similar chunks."""
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points
    return results