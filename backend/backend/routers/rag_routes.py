"""
Task 3.2: High-Performance RAG endpoints backed by Qdrant.
"""
import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from backend.config.settings import settings
from backend.models.rag_schemas import (
    DocumentResponse,
    DocumentUpload,
    RAGRequest,
    RAGResponse,
)
from backend.services.onnx_inference import (
    get_current_reranker_path,
    get_current_embed_path,
    switch_to_fallback_mode,
    switch_to_primary_mode,
)
from backend.services.qdrant_client import get_qdrant_client, ensure_collection
from backend.services.qdrant_seed import get_seed_status
from backend.services.rag_pipeline import (
    answer_question,
    ingest_document,
    VECTOR_LIMIT_MIN,
    VECTOR_LIMIT_MAX,
    CONTENT_CHAR_MIN,
    CONTENT_CHAR_MAX,
    DEFAULT_CONTENT_CHAR_LIMIT,
)
from backend.utils.file_loader import load_document_from_path
from backend.config.knowledge_config.inference_config import inference_config

logger = logging.getLogger(__name__)
router = APIRouter()
COLLECTION_NAME = settings.QDRANT_COLLECTION


def _ensure_vector_collection() -> None:
    """Ensure the target Qdrant collection exists with the expected vector size."""
    if inference_config.ENABLE_REMOTE_INFERENCE:
        vector_size = settings.RAG_VECTOR_SIZE
    else:
        from backend.services.onnx_inference import get_embedding_model  # Local import to avoid loading ONNX when remote inference is used

        vector_size = get_embedding_model().vector_size

    ensure_collection(vector_size)


@router.get("/seed-status")
async def seed_status() -> Dict[str, Any]:
    """Expose current Qdrant seed progress."""
    return get_seed_status()


@router.post("/ask", response_model=RAGResponse)
async def ask_question(request: RAGRequest) -> RAGResponse:
    """Answer a question using vector retrieval, reranking and citations."""
    try:
        logger.info("📝 RAG query received")
        return await answer_question(
            request.question,
            top_k=request.top_k or 5,
            include_timings=request.include_timings,
            reranker_override=request.reranker,
            vector_limit=request.vector_limit,
            content_char_limit=request.content_char_limit,
        )
    except Exception as exc:
        logger.exception("❌ RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(payload: DocumentUpload) -> DocumentResponse:
    """Ingest a document into the knowledge base."""
    try:
        logger.info("📚 Ingesting document: %s", payload.title)
        return await ingest_document(
            title=payload.title,
            content=payload.content,
            source=payload.source or payload.title,
            metadata=payload.metadata or {},
        )
    except Exception as exc:
        logger.exception("❌ Document ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest/sample", response_model=Dict[str, Any])
async def ingest_sample_corpus() -> Dict[str, Any]:
    """
    Convenience endpoint to ingest sample documents from the data/ folder.
    """
    try:
        docs = load_document_from_path("data/The-Prop-Building-Guidebook.txt")
        docs += load_document_from_path("data/Revenge-Of-The-Sith-pdf.pdf")

        responses = []
        for doc in docs:
            responses.append(
                await ingest_document(
                    title=doc.title,
                    content=doc.content,
                    source=doc.source,
                    metadata=doc.metadata,
                )
            )

        return {
            "ingested_documents": [resp.title for resp in responses],
            "total_chunks": sum(resp.num_chunks for resp in responses),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("❌ Sample ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats")
async def stats() -> Dict[str, Any]:
    """Return basic statistics about the Qdrant collection."""
    _ensure_vector_collection()
    client = get_qdrant_client()
    info = client.get_collection(collection_name=COLLECTION_NAME)
    return {
        "vectors_count": info.vectors_count,  # type: ignore[attr-defined]
        "segments_count": info.segments_count,  # type: ignore[attr-defined]
        "status": info.status,  # type: ignore[attr-defined]
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check connectivity with Qdrant."""
    try:
        _ensure_vector_collection()
        return {"status": "healthy"}
    except Exception as exc:
        logger.exception("RAG health check failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/config")
async def rag_config() -> Dict[str, Any]:
    """Expose RAG model metadata and tunable parameter ranges."""
    from backend.services.onnx_inference import get_embedding_model  # Local import keeps lazy loading behavior

    embedding_model = get_embedding_model()
    embedding_path = getattr(
        embedding_model,
        "resolved_model_path",
        getattr(embedding_model, "configured_path", settings.ONNX_EMBED_MODEL_PATH),
    )
    current_embed = get_current_embed_path()
    current_reranker = get_current_reranker_path()

    # Determine current mode
    is_primary_mode = (
        current_embed == settings.ONNX_EMBED_MODEL_PATH
        and current_reranker == settings.ONNX_RERANK_MODEL_PATH
    )
    is_fallback_mode = (
        current_embed == settings.EMBED_FALLBACK_MODEL_PATH
        and current_reranker == settings.RERANK_FALLBACK_MODEL_PATH
    )

    # Mode options
    mode_options = ["primary", "fallback"]

    return {
        "models": {
            "embedding_current": current_embed,
            "embedding_primary": settings.ONNX_EMBED_MODEL_PATH,
            "embedding_fallback": settings.EMBED_FALLBACK_MODEL_PATH,
            "reranker_current": current_reranker,
            "reranker_primary": settings.ONNX_RERANK_MODEL_PATH,
            "reranker_fallback": settings.RERANK_FALLBACK_MODEL_PATH,
            "llm_default": settings.OPENAI_MODEL,
        },
        "current_mode": "primary" if is_primary_mode else "fallback" if is_fallback_mode else "mixed",
        "mode_options": mode_options,
        "limits": {
            "vector_min": VECTOR_LIMIT_MIN,
            "vector_max": VECTOR_LIMIT_MAX,
            "content_char_min": CONTENT_CHAR_MIN,
            "content_char_max": CONTENT_CHAR_MAX,
            "content_char_default": DEFAULT_CONTENT_CHAR_LIMIT,
        },
    }


@router.post("/switch-mode")
async def switch_mode(mode: str) -> Dict[str, Any]:
    """
    Switch between primary (BGE) and fallback (MiniLM) mode.

    Modes:
    - primary: BGE-M3 embed + BGE reranker (high quality, slower)
    - fallback: MiniLM embed + MiniLM reranker (fast, good quality)
    """
    mode = mode.lower()

    if mode not in ["primary", "fallback"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be 'primary' or 'fallback'."
        )

    try:
        if mode == "primary":
            switched = switch_to_primary_mode()
            new_embed = settings.ONNX_EMBED_MODEL_PATH
            new_reranker = settings.ONNX_RERANK_MODEL_PATH
        else:  # fallback
            switched = switch_to_fallback_mode()
            new_embed = settings.EMBED_FALLBACK_MODEL_PATH
            new_reranker = settings.RERANK_FALLBACK_MODEL_PATH

        return {
            "success": True,
            "mode": mode,
            "switched": switched,
            "models": {
                "embedding": new_embed,
                "reranker": new_reranker,
            },
            "message": f"Switched to {mode} mode (embed + reranker)" if switched else f"Already in {mode} mode"
        }
    except Exception as e:
        logger.error(f"Failed to switch mode to {mode}: {e}")
        raise HTTPException(status_code=500, detail=f"Mode switch failed: {str(e)}")
