"""
Inference Service Configuration
"""
import os
from typing import Literal


class InferenceServiceConfig:
    """Configuration holder for the inference service."""

    # === Model configuration ===
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
    RERANK_MODEL_NAME: str = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")

    # === Device configuration ===
    DEVICE: str = os.getenv("DEVICE", "cpu")  # cpu, cuda, cuda:0, mps
    USE_FP16: bool = os.getenv("USE_FP16", "false").lower() == "true"

    # === Performance tuning ===
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "32"))
    DYNAMIC_BATCHING: bool = os.getenv("DYNAMIC_BATCHING", "false").lower() == "true"
    MAX_WAIT_MS: int = int(os.getenv("MAX_WAIT_MS", "10"))

    # === Service options ===
    HOST: str = os.getenv("INFERENCE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("INFERENCE_PORT", "8001"))
    WORKERS: int = int(os.getenv("INFERENCE_WORKERS", "1"))

    # === Logging ===
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    @classmethod
    def get_config_summary(cls) -> dict:
        """Return a concise configuration summary."""
        return {
            "embed_model": cls.EMBED_MODEL_NAME,
            "rerank_model": cls.RERANK_MODEL_NAME,
            "device": cls.DEVICE,
            "max_batch_size": cls.MAX_BATCH_SIZE,
            "dynamic_batching": cls.DYNAMIC_BATCHING,
        }


config = InferenceServiceConfig()
