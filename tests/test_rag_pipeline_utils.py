"""Unit tests for helper logic inside the RAG pipeline service."""
from backend.services import rag_pipeline


def test_should_not_switch_when_remote_inference_enabled(monkeypatch):
    monkeypatch.setattr(rag_pipeline.inference_config, "ENABLE_REMOTE_INFERENCE", True)
    rag_pipeline._reranker_switch_locked = False

    assert rag_pipeline._should_switch_reranker(500.0) is False

    # Restore default for other tests
    monkeypatch.setattr(rag_pipeline.inference_config, "ENABLE_REMOTE_INFERENCE", False)
    rag_pipeline._reranker_switch_locked = False


def test_should_switch_when_latency_exceeds_threshold(monkeypatch):
    rag_pipeline._reranker_switch_locked = False
    monkeypatch.setattr(rag_pipeline.inference_config, "ENABLE_REMOTE_INFERENCE", False)
    monkeypatch.setattr(rag_pipeline.settings, "RERANK_FALLBACK_MODEL_PATH", "./models/minilm-reranker-onnx")
    monkeypatch.setattr(rag_pipeline.settings, "RERANK_CPU_SWITCH_THRESHOLD_MS", 300.0)
    monkeypatch.setattr(rag_pipeline, "reranker_is_cpu_only", lambda: True)

    assert rag_pipeline._should_switch_reranker(350.0) is True
    rag_pipeline._reranker_switch_locked = False


def test_should_not_switch_when_gpu_available(monkeypatch):
    rag_pipeline._reranker_switch_locked = False
    monkeypatch.setattr(rag_pipeline.inference_config, "ENABLE_REMOTE_INFERENCE", False)
    monkeypatch.setattr(rag_pipeline.settings, "RERANK_FALLBACK_MODEL_PATH", "./models/minilm-reranker-onnx")
    monkeypatch.setattr(rag_pipeline, "reranker_is_cpu_only", lambda: False)

    assert rag_pipeline._should_switch_reranker(600.0) is False
    assert rag_pipeline._reranker_switch_locked is True
    rag_pipeline._reranker_switch_locked = False
