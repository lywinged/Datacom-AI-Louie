"""
Inference service configuration module.

Client access to the remote inference service is disabled by default for backward compatibility.
"""
import os


class InferenceConfig:
    """Configuration values for the inference clients."""

    # === Feature flags (disabled by default) ===
    ENABLE_REMOTE_INFERENCE: bool = os.getenv("ENABLE_REMOTE_INFERENCE", "false").lower() == "true"

    # === Service URLs ===
    EMBEDDING_SERVICE_URL: str = os.getenv(
        "EMBEDDING_SERVICE_URL",
        "http://localhost:8001"
    )
    RERANK_SERVICE_URL: str = os.getenv(
        "RERANK_SERVICE_URL",
        "http://localhost:8001"
    )

    # === Performance parameters ===
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))
    RERANK_BATCH_SIZE: int = int(os.getenv("RERANK_BATCH_SIZE", "64"))
    MAX_PARALLEL_INFERENCE: int = int(os.getenv("MAX_PARALLEL_INFERENCE", "4"))

    # === Timeout settings ===
    EMBED_TIMEOUT_SEC: float = float(os.getenv("EMBED_TIMEOUT_SEC", "10.0"))
    RERANK_TIMEOUT_SEC: float = float(os.getenv("RERANK_TIMEOUT_SEC", "5.0"))

    # === Retry strategy ===
    MAX_RETRIES: int = int(os.getenv("INFERENCE_MAX_RETRIES", "2"))
    RETRY_DELAY_SEC: float = float(os.getenv("INFERENCE_RETRY_DELAY", "0.5"))

    # === Fallback policy ===
    ENABLE_LOCAL_FALLBACK: bool = os.getenv("ENABLE_LOCAL_FALLBACK", "true").lower() == "true"
    FALLBACK_AFTER_FAILURES: int = int(os.getenv("FALLBACK_AFTER_FAILURES", "3"))

    # === Circuit breaker ===
    CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "10"))
    CIRCUIT_BREAKER_TIMEOUT_SEC: int = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))

    @classmethod
    def is_enabled(cls) -> bool:
        """Return True when remote inference is enabled."""
        return cls.ENABLE_REMOTE_INFERENCE

    @classmethod
    def get_config_summary(cls) -> dict:
        """Produce a concise summary for logging."""
        return {
            "enabled": cls.ENABLE_REMOTE_INFERENCE,
            "embedding_url": cls.EMBEDDING_SERVICE_URL if cls.ENABLE_REMOTE_INFERENCE else "local",
            "rerank_url": cls.RERANK_SERVICE_URL if cls.ENABLE_REMOTE_INFERENCE else "local",
        }


inference_config = InferenceConfig()
