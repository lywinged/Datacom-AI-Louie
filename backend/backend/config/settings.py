"""
Configuration settings for AI Assessment Project
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_VERSION: str = os.getenv("AZURE_OPENAI_VERSION", "2024-08-01-preview")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Gpt4o")

    # Generic OpenAI-compatible settings (e.g. custom gateways, OpenAI SaaS)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
    # Use Gpt4o as default model
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "Gpt4o")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/aidb"
    )

    # Vector DB (Qdrant)
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "assessment_docs_minilm")

    # Chat Configuration (Task 3.1)
    CHAT_HISTORY_LIMIT: int = 10
    STREAM_BUFFER_SIZE: int = 1024
    CHAT_TIMEOUT: int = 30

    # Cost Configuration (USD per 1K tokens)
    # Gpt4o pricing
    PROMPT_COST_PER_1K: float = 0.005  # $0.005 per 1K prompt tokens
    COMPLETION_COST_PER_1K: float = 0.015  # $0.015 per 1K completion tokens

    # RAG Configuration (Task 3.2)
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 20  # Candidates for reranking
    RERANK_TOP_N: int = 5
    TARGET_RETRIEVAL_MS: int = 300  # Target <300ms
    METADATA_TITLE_MATCH_LIMIT: int = int(os.getenv("METADATA_TITLE_MATCH_LIMIT", "5"))
    RAG_VECTOR_SIZE: int = int(os.getenv("RAG_VECTOR_SIZE", "1024"))

    # ONNX Inference
    USE_ONNX_INFERENCE: bool = os.getenv("USE_ONNX_INFERENCE", "true").lower() == "true"
    USE_INT8_QUANTIZATION: bool = os.getenv("USE_INT8_QUANTIZATION", "true").lower() == "true"
    ONNX_EMBED_MODEL_PATH: str = os.getenv("ONNX_EMBED_MODEL_PATH", "./models/bge-m3-embed-int8")
    ONNX_RERANK_MODEL_PATH: str = os.getenv("ONNX_RERANK_MODEL_PATH", "./models/bge-reranker-int8")
    INFERENCE_SERVICE_URL: str = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8001")
    EMBED_FALLBACK_MODEL_PATH: Optional[str] = os.getenv("EMBED_FALLBACK_MODEL_PATH", "./models/minilm-embed-int8")
    RERANK_FALLBACK_MODEL_PATH: Optional[str] = os.getenv("RERANK_FALLBACK_MODEL_PATH", "./models/minilm-reranker-onnx")
    RERANK_CPU_SWITCH_THRESHOLD_MS: float = float(os.getenv("RERANK_CPU_SWITCH_THRESHOLD_MS", "450.0"))

    # Agent Configuration (Task 3.3)
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_TOOL_TIMEOUT: int = 5
    AGENT_RETRY_ATTEMPTS: int = 3

    # Code Assistant Configuration (Task 3.4)
    CODE_MAX_RETRIES: int = 3
    CODE_OUTPUT_DIR: str = "./generated_code"
    SUPPORTED_LANGUAGES: list = ["python", "rust", "javascript", "typescript", "go", "bash", "java", "c", "cpp", "csharp"]

    # Performance & Monitoring
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_METRICS: bool = True
    METRICS_EXPORT_PATH: str = "./metrics"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Allow unused env vars for flexible deployments


# Global settings instance
settings = Settings()


# Chat configuration
CHAT_CONFIG = {
    "max_history": settings.CHAT_HISTORY_LIMIT,
    "stream_buffer_size": settings.STREAM_BUFFER_SIZE,
    "timeout_seconds": settings.CHAT_TIMEOUT,
    "cost_per_1k_prompt": settings.PROMPT_COST_PER_1K,
    "cost_per_1k_completion": settings.COMPLETION_COST_PER_1K,
}

# RAG configuration
RAG_CONFIG = {
    "chunk_size": settings.CHUNK_SIZE,
    "chunk_overlap": settings.CHUNK_OVERLAP,
    "top_k": settings.RETRIEVAL_TOP_K,
    "rerank_top_n": settings.RERANK_TOP_N,
    "target_ms": settings.TARGET_RETRIEVAL_MS,
    "use_onnx": settings.USE_ONNX_INFERENCE,
    "use_int8": settings.USE_INT8_QUANTIZATION,
    "inference_url": settings.INFERENCE_SERVICE_URL,
    "rerank_fallback_model_path": settings.RERANK_FALLBACK_MODEL_PATH,
    "rerank_cpu_switch_threshold_ms": settings.RERANK_CPU_SWITCH_THRESHOLD_MS,
    "metadata_title_match_limit": settings.METADATA_TITLE_MATCH_LIMIT,
}

# Agent configuration
AGENT_CONFIG = {
    "max_iterations": settings.AGENT_MAX_ITERATIONS,
    "tool_timeout": settings.AGENT_TOOL_TIMEOUT,
    "retry_attempts": settings.AGENT_RETRY_ATTEMPTS,
}

# Code assistant configuration
CODE_CONFIG = {
    "max_retries": settings.CODE_MAX_RETRIES,
    "output_dir": settings.CODE_OUTPUT_DIR,
    "supported_languages": settings.SUPPORTED_LANGUAGES,
}

# Generic OpenAI configuration
OPENAI_CONFIG = {
    "api_key": settings.OPENAI_API_KEY,
    "base_url": settings.OPENAI_BASE_URL,
    "model": settings.OPENAI_MODEL,
}
