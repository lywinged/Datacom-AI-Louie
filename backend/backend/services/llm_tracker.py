"""
LLM call tracking helpers.

Automatically records token usage, cost, and latency metrics for OpenAI-style APIs.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from functools import wraps
from contextlib import asynccontextmanager

from backend.services.token_counter import get_token_counter, TokenUsage
from backend.services.metrics import (
    llm_token_usage_counter,
    llm_request_counter,
    llm_cost_counter,
    llm_request_duration_histogram,
)

logger = logging.getLogger(__name__)


class LLMTracker:
    """Utility class that records metrics for LLM API calls."""

    def __init__(self):
        self.token_counter = get_token_counter()

    async def track_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        completion: str,
        duration: float,
        endpoint: str = "chat",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> TokenUsage:
        """
        Track a single ChatCompletion call.

        Args:
            model: Model identifier
            messages: Input message payload
            completion: Completed text output
            duration: Request duration in seconds
            endpoint: API endpoint label
            extra_metadata: Optional metadata for downstream logging

        Returns:
            TokenUsage dataclass
        """
        try:
            # Count prompt and completion tokens
            usage = self.token_counter.count_chat_completion(messages, completion, model)

            # Emit Prometheus counters and histograms
            llm_token_usage_counter.labels(model=model, token_type="prompt").inc(
                usage.prompt_tokens
            )
            llm_token_usage_counter.labels(model=model, token_type="completion").inc(
                usage.completion_tokens
            )

            llm_request_counter.labels(model=model, endpoint=endpoint, status="success").inc()

            # Estimate usage cost
            cost = self.token_counter.estimate_cost(usage)
            llm_cost_counter.labels(model=model).inc(cost)

            # Record end-to-end latency
            llm_request_duration_histogram.labels(model=model, endpoint=endpoint).observe(
                duration
            )

            # Use print for debugging to ensure visibility
            print(f"✅ LLM call tracked - Model: {model}, Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}, Cost: ${cost:.6f}, Duration: {duration:.2f}s")
            logger.info(
                f"LLM call tracked - Model: {model}, "
                f"Prompt tokens: {usage.prompt_tokens}, "
                f"Completion tokens: {usage.completion_tokens}, "
                f"Total: {usage.total_tokens}, "
                f"Cost: ${cost:.6f}, "
                f"Duration: {duration:.2f}s"
            )

            return usage

        except Exception as e:
            logger.error(f"Error tracking LLM call: {e}", exc_info=True)
            # Record failure while keeping the pipeline alive
            llm_request_counter.labels(model=model, endpoint=endpoint, status="error").inc()
            # Provide an empty usage snapshot to callers
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model=model,
                timestamp=usage.timestamp if 'usage' in locals() else None,
            )

    async def track_streaming_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        collected_chunks: List[str],
        duration: float,
        endpoint: str = "chat",
    ) -> TokenUsage:
        """
        Track a streaming ChatCompletion call.

        Args:
            model: Model identifier
            messages: Input message payload
            collected_chunks: All streaming chunks concatenated later
            duration: Total duration in seconds
            endpoint: API endpoint label

        Returns:
            TokenUsage dataclass
        """
        completion = "".join(collected_chunks)
        return await self.track_chat_completion(
            model=model,
            messages=messages,
            completion=completion,
            duration=duration,
            endpoint=endpoint,
        )

    async def track_embedding(
        self,
        model: str,
        texts: List[str],
        duration: float,
    ) -> TokenUsage:
        """
        Track an embedding API call.

        Args:
            model: Model identifier
            texts: Input text batch
            duration: Request duration in seconds

        Returns:
            TokenUsage dataclass
        """
        try:
            # Embeddings only consume prompt tokens
            total_tokens = sum(
                self.token_counter.count_tokens(text, model) for text in texts
            )

            usage = TokenUsage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
                model=model,
                timestamp=None,
            )

            # Update Prometheus metrics
            llm_token_usage_counter.labels(model=model, token_type="prompt").inc(
                total_tokens
            )
            llm_request_counter.labels(model=model, endpoint="embedding", status="success").inc()

            # Estimate usage cost
            cost = self.token_counter.estimate_cost(usage)
            llm_cost_counter.labels(model=model).inc(cost)

            llm_request_duration_histogram.labels(model=model, endpoint="embedding").observe(
                duration
            )

            logger.info(
                f"Embedding call tracked - Model: {model}, "
                f"Tokens: {total_tokens}, "
                f"Texts: {len(texts)}, "
                f"Cost: ${cost:.6f}, "
                f"Duration: {duration:.2f}s"
            )

            return usage

        except Exception as e:
            logger.error(f"Error tracking embedding call: {e}", exc_info=True)
            llm_request_counter.labels(model=model, endpoint="embedding", status="error").inc()
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model=model,
                timestamp=None,
            )

    @asynccontextmanager
    async def track_request(
        self,
        model: str,
        endpoint: str = "chat",
    ):
        """
        Context manager that automatically tracks request duration.

        Usage::

            async with tracker.track_request("gpt-4", "chat") as ctx:
                response = await client.chat.completions.create(...)
                ctx["response"] = response

        Yields:
            A mutable context dictionary for passing metadata (e.g., duration)
        """
        start_time = time.time()
        context = {}

        try:
            yield context
            duration = time.time() - start_time

            # Allow callers to override duration manually via the context dict
            if "duration" in context:
                duration = context["duration"]

            context["duration"] = duration

        except Exception as e:
            duration = time.time() - start_time
            llm_request_counter.labels(model=model, endpoint=endpoint, status="error").inc()
            llm_request_duration_histogram.labels(model=model, endpoint=endpoint).observe(
                duration
            )
            logger.error(
                f"LLM request failed - Model: {model}, "
                f"Endpoint: {endpoint}, "
                f"Duration: {duration:.2f}s, "
                f"Error: {e}"
            )
            raise


# Global singleton
_tracker: Optional[LLMTracker] = None


def get_llm_tracker() -> LLMTracker:
    """Return the shared LLMTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = LLMTracker()
    return _tracker


# Convenience helpers
async def track_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    completion: str,
    duration: float,
    endpoint: str = "chat",
) -> TokenUsage:
    """Track a standard chat completion request."""
    return await get_llm_tracker().track_chat_completion(
        model, messages, completion, duration, endpoint
    )


async def track_streaming_completion(
    model: str,
    messages: List[Dict[str, str]],
    collected_chunks: List[str],
    duration: float,
    endpoint: str = "chat",
) -> TokenUsage:
    """Track a streaming chat completion request."""
    return await get_llm_tracker().track_streaming_completion(
        model, messages, collected_chunks, duration, endpoint
    )


async def track_embedding(
    model: str,
    texts: List[str],
    duration: float,
) -> TokenUsage:
    """Track an embedding request."""
    return await get_llm_tracker().track_embedding(model, texts, duration)
