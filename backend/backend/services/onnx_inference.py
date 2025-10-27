"""
ONNX-based embedding and reranker utilities.
"""
from __future__ import annotations

import os
import logging
from functools import lru_cache
from threading import Lock
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from backend.config.settings import settings

logger = logging.getLogger(__name__)


def _has_cuda_available() -> bool:
    """Check if CUDA is available for ONNX Runtime."""
    try:
        providers = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in providers
        if has_cuda:
            logger.info("✅ CUDA detected - GPU acceleration available")
        else:
            logger.info("ℹ️  CUDA not available - using CPU")
        return has_cuda
    except Exception as e:
        logger.warning(f"Failed to check CUDA availability: {e}")
        return False


def _resolve_model_file(model_path: str, prefer_int8: bool = True) -> str:
    """Return concrete ONNX file path from directory or file.

    Auto-detects GPU:
    - If GPU available: Use full precision model.onnx (better quality)
    - If CPU only: Use int8 quantized model_int8.onnx (faster on CPU)
    """
    # Auto-detect: GPU prefers FP32, CPU prefers INT8
    has_gpu = _has_cuda_available()
    if has_gpu:
        prefer_int8 = False  # GPU: use full precision
        logger.info("🚀 GPU detected - using full precision ONNX model")
    else:
        prefer_int8 = True  # CPU: use quantized
        logger.info("🔧 CPU mode - using INT8 quantized ONNX model")

    if os.path.isdir(model_path):
        candidates = []
        if prefer_int8:
            # CPU: prefer quantized models
            candidates.extend([
                os.path.join(model_path, "model_int8.onnx"),
                os.path.join(model_path, "onnx/model_int8.onnx"),
            ])
        # Always try full precision as fallback
        candidates.extend([
            os.path.join(model_path, "model.onnx"),
            os.path.join(model_path, "onnx/model.onnx"),
        ])
        for candidate in candidates:
            if os.path.isfile(candidate):
                logger.info(f"✅ Using ONNX model: {candidate}")
                return candidate
    if os.path.isfile(model_path):
        return model_path
    raise FileNotFoundError(f"ONNX model file not found for path: {model_path}")


class ONNXEmbeddingModel:
    """Wrapper around ONNX Runtime for embedding generation."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        resolved = _resolve_model_file(model_path, prefer_int8=settings.USE_INT8_QUANTIZATION)
        self.configured_path = model_path
        self.resolved_model_path = resolved

        # Tokenizer files are always at the root model directory, not in subdirs like onnx/
        # So always use model_path directly for tokenizer if it's a directory
        if os.path.isdir(model_path):
            tokenizer_path = model_path
        else:
            # If model_path is a file, look for tokenizer in the same directory
            tokenizer_path = os.path.dirname(model_path)

        # Use slow tokenizer to avoid fast tokenizer version incompatibility
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

        providers: list[str] = []
        if device.startswith("cuda"):
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        num_threads = int(os.getenv("OMP_NUM_THREADS", "8"))
        # num_threads=int(os.cpu_count())
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads

        self.session = ort.InferenceSession(resolved, sess_options=sess_options, providers=providers)
        self.vector_size = self.session.get_outputs()[0].shape[2]

    def encode(self, texts: List[str], *, batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            ort_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in encoded:
                ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

            outputs = self.session.run(None, ort_inputs)
            last_hidden = outputs[0]  # (batch, seq, hidden)
            mask = encoded["attention_mask"]
            pooled = self._mean_pool(last_hidden, mask)
            normalized = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            embeddings.append(normalized)
        return np.vstack(embeddings)

    @staticmethod
    def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask


class ONNXRerankerModel:
    """Wrapper around ONNX Runtime for reranking scores."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        resolved = _resolve_model_file(model_path, prefer_int8=settings.USE_INT8_QUANTIZATION)
        self.model_path = model_path
        self.configured_path = model_path
        self.resolved_model_path = resolved

        # Tokenizer files are always at the root model directory, not in subdirs like onnx/
        # So always use model_path directly for tokenizer if it's a directory
        if os.path.isdir(model_path):
            tokenizer_path = model_path
        else:
            # If model_path is a file, look for tokenizer in the same directory
            tokenizer_path = os.path.dirname(model_path)

        # Use slow tokenizer to avoid fast tokenizer version incompatibility
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

        providers: list[str] = []
        if device.startswith("cuda"):
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        num_threads = int(os.getenv("OMP_NUM_THREADS", "8"))
        # num_threads = int(os.cpu_count())
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads

        self.session = ort.InferenceSession(resolved, sess_options=sess_options, providers=providers)
        self.providers = self.session.get_providers()

    def score(self, query: str, documents: List[str], *, batch_size: int = 64) -> np.ndarray:
        """Compute relevance scores between query and documents."""
        scores: list[np.ndarray] = []
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            pairs = [[query, doc] for doc in batch_docs]
            encoded = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            ort_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in encoded:
                ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
            outputs = self.session.run(None, ort_inputs)
            logits = outputs[0].astype(np.float32).squeeze(-1)
            scores.append(logits)
        return np.concatenate(scores)

    def is_cpu_only(self) -> bool:
        """Return True if model runs solely on CPU."""
        normalized = [provider.lower() for provider in self.providers]
        return not any("cuda" in provider or "tensorrt" in provider or "rocm" in provider for provider in normalized)


_embed_lock = Lock()
_embed_model: Optional[ONNXEmbeddingModel] = None
_embed_model_path: Optional[str] = settings.ONNX_EMBED_MODEL_PATH


def get_embedding_model() -> ONNXEmbeddingModel:
    """Return the active embedding model instance."""
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                if not _embed_model_path:
                    raise RuntimeError("ONNX_EMBED_MODEL_PATH is not configured.")
                _embed_model = ONNXEmbeddingModel(_embed_model_path)
    return _embed_model


_reranker_lock = Lock()
_reranker_model: Optional[ONNXRerankerModel] = None
_reranker_model_path: Optional[str] = settings.ONNX_RERANK_MODEL_PATH


def get_reranker_model() -> ONNXRerankerModel:
    """Return the active reranker model instance."""
    global _reranker_model
    if _reranker_model is None:
        with _reranker_lock:
            if _reranker_model is None:
                if not _reranker_model_path:
                    raise RuntimeError("ONNX_RERANK_MODEL_PATH is not configured.")
                _reranker_model = ONNXRerankerModel(_reranker_model_path)
    return _reranker_model


def reranker_is_cpu_only() -> bool:
    return get_reranker_model().is_cpu_only()


def switch_to_fallback_reranker() -> bool:
    """Switch to fallback reranker model if configured."""
    fallback_path = settings.RERANK_FALLBACK_MODEL_PATH
    if not fallback_path:
        return False

    global _reranker_model_path, _reranker_model
    if _reranker_model_path == fallback_path:
        return False

    with _reranker_lock:
        _reranker_model = ONNXRerankerModel(fallback_path)
        _reranker_model_path = fallback_path
    return True


def set_reranker_model_path(model_path: str) -> None:
    """Force reload of reranker model from a specific path."""
    global _reranker_model_path, _reranker_model
    with _reranker_lock:
        _reranker_model = ONNXRerankerModel(model_path)
        _reranker_model_path = model_path


def get_current_reranker_path() -> Optional[str]:
    """Return the configured reranker model path."""
    return _reranker_model_path


def switch_to_fallback_embed() -> bool:
    """Switch to fallback embedding model if configured."""
    fallback_path = settings.EMBED_FALLBACK_MODEL_PATH
    if not fallback_path:
        return False

    global _embed_model_path, _embed_model
    if _embed_model_path == fallback_path:
        return False

    with _embed_lock:
        _embed_model = ONNXEmbeddingModel(fallback_path)
        _embed_model_path = fallback_path
    return True


def switch_to_primary_embed() -> bool:
    """Switch back to primary embedding model."""
    primary_path = settings.ONNX_EMBED_MODEL_PATH
    if not primary_path:
        return False

    global _embed_model_path, _embed_model
    if _embed_model_path == primary_path:
        return False

    with _embed_lock:
        _embed_model = ONNXEmbeddingModel(primary_path)
        _embed_model_path = primary_path
    return True


def get_current_embed_path() -> Optional[str]:
    """Return the configured embedding model path."""
    return _embed_model_path


def switch_to_fallback_mode() -> bool:
    """Switch both embedding and reranker to fallback models."""
    embed_switched = switch_to_fallback_embed()
    reranker_switched = switch_to_fallback_reranker()
    return embed_switched or reranker_switched


def switch_to_primary_mode() -> bool:
    """Switch both embedding and reranker to primary models."""
    embed_switched = switch_to_primary_embed()
    reranker_switched = (
        set_reranker_model_path(settings.ONNX_RERANK_MODEL_PATH)
        if settings.ONNX_RERANK_MODEL_PATH
        else False
    )
    return embed_switched or (reranker_switched is not False)
