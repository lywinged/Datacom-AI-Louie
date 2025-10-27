"""
Task 3.1: Conversational Chat Core

Features:
- Streaming chat with GPT-4o
- Persistent conversation history (last N messages)
- Token counting and cost tracking
- Latency monitoring
"""

import os
import time
import logging
from typing import List, Dict, AsyncGenerator
from openai import AsyncOpenAI

from backend.models.chat_schemas import ChatMessage, ChatResponse
from backend.services.llm_tracker import get_llm_tracker
from backend.utils.openai import sanitize_messages

logger = logging.getLogger(__name__)


class ChatService:
    """Chat service with streaming support and telemetry"""

    def __init__(self):
        """Initialize with standard OpenAI API"""
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        # Initialize conversation history
        self.conversation_history: List[ChatMessage] = []

        # Initialize tracker
        self.tracker = get_llm_tracker()

        logger.info(f"✅ ChatService initialized with model: {self.model_name}")

    def _get_history_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history as OpenAI message format"""
        recent_messages = self.conversation_history[-max_messages:] if self.conversation_history else []
        return [{"role": msg.role, "content": msg.content} for msg in recent_messages]

    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        message = ChatMessage(role=role, content=content)
        self.conversation_history.append(message)

        # Keep only last N messages to prevent memory overflow
        max_history = 20  # Keep more for context
        if len(self.conversation_history) > max_history * 2:
            self.conversation_history = self.conversation_history[-max_history:]

    def get_history(self) -> List[ChatMessage]:
        """Get full conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("🗑️  Conversation history cleared")

    async def chat_completion(
        self,
        user_message: str,
        max_history: int = 10
    ) -> ChatResponse:
        """
        Non-streaming chat completion with telemetry

        Args:
            user_message: User's input message
            max_history: Maximum number of historical messages to include

        Returns:
            ChatResponse with assistant message and metrics
        """
        start_time = time.time()

        # Add user message to history
        self.add_message("user", user_message)

        # Build messages for API call
        messages = sanitize_messages(self._get_history_context(max_history))

        try:
            # Call OpenAI
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )

            # Extract response
            assistant_message = response.choices[0].message.content
            self.add_message("assistant", assistant_message)

            # Calculate metrics
            duration = time.time() - start_time
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Calculate cost (GPT-4o pricing)
            # Input: $2.50 / 1M tokens, Output: $10.00 / 1M tokens
            cost_usd = (
                (prompt_tokens / 1_000_000) * 2.50 +
                (completion_tokens / 1_000_000) * 10.00
            )

            # Track usage
            await self.tracker.track_chat_completion(
                model=self.model_name,
                messages=messages,
                completion=assistant_message,
                duration=duration,
                endpoint="chat",
            )

            logger.info(
                f"✅ Chat completion - Tokens: {total_tokens}, "
                f"Cost: ${cost_usd:.6f}, Latency: {duration*1000:.2f}ms"
            )

            return ChatResponse(
                message=assistant_message,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                latency_ms=duration * 1000,
            )

        except Exception as e:
            logger.error(f"❌ Chat completion failed: {e}", exc_info=True)
            raise

    async def chat_completion_stream(
        self,
        user_message: str,
        max_history: int = 10
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat completion

        Args:
            user_message: User's input message
            max_history: Maximum number of historical messages to include

        Yields:
            Chunks of assistant response as they arrive
        """
        start_time = time.time()

        # Add user message to history
        self.add_message("user", user_message)

        # Build messages for API call
        messages = sanitize_messages(self._get_history_context(max_history))

        try:
            # Call OpenAI with streaming
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True,
            )

            # Collect chunks for history and tracking
            collected_chunks = []

            # Stream response
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_chunks.append(content)
                    yield content

            # Assemble full response
            full_response = "".join(collected_chunks)
            self.add_message("assistant", full_response)

            # Track usage (streaming doesn't return token counts, so we estimate)
            duration = time.time() - start_time
            await self.tracker.track_streaming_completion(
                model=self.model_name,
                messages=messages,
                collected_chunks=collected_chunks,
                duration=duration,
                endpoint="chat",
            )

            logger.info(
                f"✅ Streaming chat completed - "
                f"Chunks: {len(collected_chunks)}, Latency: {duration*1000:.2f}ms"
            )

        except Exception as e:
            logger.error(f"❌ Streaming chat failed: {e}", exc_info=True)
            raise


# Global chat service instance
_chat_service: ChatService = None


def get_chat_service() -> ChatService:
    """Get or create global ChatService instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
