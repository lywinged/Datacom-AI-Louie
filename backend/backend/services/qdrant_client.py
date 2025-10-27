"""
Utilities for working with Qdrant vector store.
"""
from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.config.settings import settings

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Instantiate a Qdrant client with settings from configuration."""
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )


def ensure_collection(vector_size: int, collection: Optional[str] = None) -> None:
    """Ensure the target collection exists with the expected schema."""
    client = get_qdrant_client()
    collection_name = collection or settings.QDRANT_COLLECTION

    try:
        info = client.get_collection(collection_name=collection_name)
        existing_size = info.config.params.vectors.size  # type: ignore[attr-defined]
        if existing_size != vector_size:
            raise ValueError(
                f"Qdrant collection '{collection_name}' vector size {existing_size} "
                f"does not match expected {vector_size}"
            )
        return
    except UnexpectedResponse as exc:
        if getattr(exc, "status_code", None) != 404:
            raise
    except Exception:
        # Other errors (e.g., connection issues) should be raised upstream
        raise

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
    except UnexpectedResponse as exc:
        if getattr(exc, "status_code", None) != 409:
            raise
