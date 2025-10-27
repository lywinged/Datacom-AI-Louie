"""
Knowledge Base Configuration

Centralized configuration for knowledge management features including
summarization, deduplication, and cold/hot tier management.
"""

import os
from typing import Literal


class KnowledgeConfig:
    """Configuration for knowledge base features."""

    # ========== Summarization Settings ==========

    # Enable automatic summarization during document ingestion
    KNOWLEDGE_SUMMARY_ENABLED: bool = os.getenv("KNOWLEDGE_SUMMARY_ENABLED", "false").lower() == "true"

    # Maximum token count for summaries
    SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", "200"))

    # Token threshold - texts longer than this may be summarized
    SUMMARY_TOKEN_THRESHOLD: int = int(os.getenv("SUMMARY_TOKEN_THRESHOLD", "500"))

    # Summarization method: "extractive", "llm", or "hybrid"
    SUMMARY_METHOD: Literal["extractive", "llm", "hybrid"] = os.getenv("SUMMARY_METHOD", "hybrid")

    # Model to use for LLM-based summarization
    SUMMARY_LLM_MODEL: str = os.getenv("SUMMARY_LLM_MODEL", "deepseek-chat")

    # Summarization style: "concise", "keywords", or "abstract"
    SUMMARY_STYLE: Literal["concise", "keywords", "abstract"] = os.getenv("SUMMARY_STYLE", "concise")

    # Use LLM for long documents in hybrid mode
    SUMMARY_USE_LLM_FOR_LONG: bool = os.getenv("SUMMARY_USE_LLM_FOR_LONG", "true").lower() == "true"

    # ========== Deduplication Settings ==========

    # Enable automatic deduplication during ingestion
    KNOWLEDGE_DEDUPE_ENABLED: bool = os.getenv("KNOWLEDGE_DEDUPE_ENABLED", "true").lower() == "true"

    # Deduplication scope: "document", "role", or "global"
    DEDUPE_SCOPE: Literal["document", "role", "global"] = os.getenv("DEDUPE_SCOPE", "document")

    # ========== Cold/Hot Tier Settings ==========

    # Days of inactivity before marking document as cold
    COLD_TIER_DAYS: int = int(os.getenv("COLD_TIER_DAYS", "30"))

    # Minimum access count to prevent cold tier migration
    COLD_TIER_MIN_ACCESS: int = int(os.getenv("COLD_TIER_MIN_ACCESS", "5"))

    # Enable automatic cold tier migration
    COLD_TIER_AUTO_MIGRATE: bool = os.getenv("COLD_TIER_AUTO_MIGRATE", "true").lower() == "true"

    # ========== Chunk Settings ==========

    # Maximum chunk size in tokens
    MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "512"))

    # Overlap between chunks in tokens
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

    # ========== Batch Processing Settings ==========

    # Batch size for maintenance operations
    MAINTENANCE_BATCH_SIZE: int = int(os.getenv("MAINTENANCE_BATCH_SIZE", "100"))

    # Enable scheduled maintenance tasks
    MAINTENANCE_ENABLED: bool = os.getenv("MAINTENANCE_ENABLED", "false").lower() == "true"

    # Cron schedule for maintenance (default: daily at 2 AM)
    MAINTENANCE_SCHEDULE: str = os.getenv("MAINTENANCE_SCHEDULE", "0 2 * * *")

    @classmethod
    def get_summary_config(cls) -> dict:
        """Get summarization configuration as a dictionary."""
        return {
            "enabled": cls.KNOWLEDGE_SUMMARY_ENABLED,
            "max_tokens": cls.SUMMARY_MAX_TOKENS,
            "token_threshold": cls.SUMMARY_TOKEN_THRESHOLD,
            "method": cls.SUMMARY_METHOD,
            "llm_model": cls.SUMMARY_LLM_MODEL,
            "style": cls.SUMMARY_STYLE,
            "use_llm_for_long": cls.SUMMARY_USE_LLM_FOR_LONG,
        }

    @classmethod
    def get_dedupe_config(cls) -> dict:
        """Get deduplication configuration as a dictionary."""
        return {
            "enabled": cls.KNOWLEDGE_DEDUPE_ENABLED,
            "scope": cls.DEDUPE_SCOPE,
        }

    @classmethod
    def get_cold_tier_config(cls) -> dict:
        """Get cold tier configuration as a dictionary."""
        return {
            "days": cls.COLD_TIER_DAYS,
            "min_access": cls.COLD_TIER_MIN_ACCESS,
            "auto_migrate": cls.COLD_TIER_AUTO_MIGRATE,
        }


# Create singleton instance
knowledge_config = KnowledgeConfig()
