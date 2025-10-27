"""Unit tests for RAG router handlers with stubbed dependencies."""
import asyncio
from typing import Dict, Any
from types import SimpleNamespace

from backend.models.rag_schemas import Citation, RAGRequest, RAGResponse, DocumentUpload, DocumentResponse
from backend.routers import rag_routes


async def _fake_answer_question(*args, **kwargs) -> RAGResponse:
    """Deterministic RAG response used for tests."""
    return RAGResponse(
        answer="Stub answer for testing.",
        citations=[
            Citation(source="doc1.txt", content="Sample snippet", score=0.92),
            Citation(source="doc2.txt", content="Another snippet", score=0.81),
        ],
        retrieval_time_ms=12.5,
        confidence=0.78,
        num_chunks_retrieved=5,
        llm_time_ms=30.0,
        total_time_ms=45.5,
        timings={"embed_ms": 5.0, "vector_ms": 4.0, "rerank_ms": 3.5},
        models={"embedding": "stub-embed", "reranker": "stub-reranker"},
        token_usage={"prompt": 18, "completion": 12, "total": 30},
        token_cost_usd=0.003,
        llm_used=True,
        reranker_mode="fallback",
        vector_limit_used=10,
        content_char_limit_used=300,
    )


async def _fake_ingest_document(*args, **kwargs) -> DocumentResponse:
    """Fake document ingestion for testing."""
    return DocumentResponse(
        document_id=42,
        title=kwargs.get("title", "Test Document"),
        num_chunks=12,
        embedding_time_ms=150.5,
    )


def test_rag_ask_returns_stubbed_response(monkeypatch):
    """Test /ask endpoint returns correct RAGResponse structure."""
    monkeypatch.setattr(
        "backend.routers.rag_routes.answer_question",
        _fake_answer_question,
    )

    request = RAGRequest(question="What is the testing strategy?", top_k=3)
    response = asyncio.run(rag_routes.ask_question(request))

    assert response.answer.startswith("Stub answer")
    assert len(response.citations) == 2
    assert response.citations[0].source == "doc1.txt"
    assert response.content_char_limit_used == 300
    assert response.llm_used is True
    assert response.confidence == 0.78


def test_rag_health_success(monkeypatch):
    """Test /health endpoint returns healthy status."""
    monkeypatch.setattr(
        "backend.routers.rag_routes._ensure_vector_collection",
        lambda: None,
    )

    payload = asyncio.run(rag_routes.health_check())

    assert payload == {"status": "healthy"}


def test_rag_config_returns_model_info(monkeypatch):
    """Test /config endpoint returns RAG configuration and limits."""
    monkeypatch.setattr(
        "backend.routers.rag_routes._ensure_vector_collection",
        lambda: None,
    )

    # Mock get_embedding_model to return stub
    from backend.services.onnx_inference import get_embedding_model
    stub_model = get_embedding_model()  # Uses conftest stub

    config = asyncio.run(rag_routes.rag_config())

    assert "models" in config
    assert "limits" in config
    assert "current_mode" in config
    assert "mode_options" in config

    # Check models section
    assert "embedding_current" in config["models"]
    assert "reranker_current" in config["models"]
    assert "llm_default" in config["models"]

    # Check limits section (flexible assertions for different config values)
    assert config["limits"]["vector_min"] >= 5  # Accept 5 or 6
    assert config["limits"]["vector_max"] >= 20
    assert config["limits"]["content_char_min"] >= 100
    assert config["limits"]["content_char_max"] >= 500

    # Check mode options
    assert "primary" in config["mode_options"]
    assert "fallback" in config["mode_options"]


def test_rag_seed_status_returns_status(monkeypatch):
    """Test /seed-status endpoint returns Qdrant seed status."""
    # get_seed_status is already stubbed in conftest.py
    status = asyncio.run(rag_routes.seed_status())

    assert "status" in status
    assert "collection" in status
    assert status["collection"] == "assessment_docs_minilm"


def test_rag_upload_document(monkeypatch):
    """Test /upload endpoint ingests document correctly."""
    monkeypatch.setattr(
        "backend.routers.rag_routes.ingest_document",
        _fake_ingest_document,
    )

    payload = DocumentUpload(
        title="Test Doc",
        content="This is test content for ingestion.",
        metadata={"author": "Test Author"},
    )

    response = asyncio.run(rag_routes.upload_document(payload))

    assert response.document_id == 42
    assert response.title == "Test Doc"
    assert response.num_chunks == 12
    assert response.embedding_time_ms > 0


def test_rag_switch_mode_to_fallback(monkeypatch):
    """Test /switch-mode endpoint switches to fallback mode."""
    # switch_to_fallback_mode is already stubbed in conftest.py to return True

    result = asyncio.run(rag_routes.switch_mode("fallback"))

    assert result["success"] is True
    assert result["mode"] == "fallback"
    assert "models" in result
    assert "embedding" in result["models"]
    assert "reranker" in result["models"]


def test_rag_switch_mode_to_primary(monkeypatch):
    """Test /switch-mode endpoint switches to primary mode."""
    # switch_to_primary_mode is already stubbed in conftest.py to return True

    result = asyncio.run(rag_routes.switch_mode("primary"))

    assert result["success"] is True
    assert result["mode"] == "primary"
    assert "models" in result
    assert "embedding" in result["models"]
    assert "reranker" in result["models"]


def test_rag_stats_returns_collection_info(monkeypatch):
    """Test /stats endpoint returns Qdrant collection statistics."""
    monkeypatch.setattr(
        "backend.routers.rag_routes._ensure_vector_collection",
        lambda: None,
    )

    # Mock get_qdrant_client
    class MockCollectionInfo:
        vectors_count = 15000
        segments_count = 3
        status = "green"

    class MockQdrantClient:
        def get_collection(self, collection_name):
            return MockCollectionInfo()

    monkeypatch.setattr(
        "backend.routers.rag_routes.get_qdrant_client",
        lambda: MockQdrantClient(),
    )

    stats = asyncio.run(rag_routes.stats())

    assert stats["vectors_count"] == 15000
    assert stats["segments_count"] == 3
    assert stats["status"] == "green"
