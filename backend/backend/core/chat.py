"""
Conversational core service with optional streaming support.
Prefers generic OpenAI credentials when provided, otherwise falls back to Azure OpenAI.
"""
import logging
import time
from typing import Generator, Optional, Tuple

from openai import OpenAI, AzureOpenAI

from backend.config.settings import settings, OPENAI_CONFIG
from backend.models.chat import ChatSession, MessageRole, MessageMetrics
from backend.utils.openai import sanitize_messages

logger = logging.getLogger(__name__)


class ChatService:
    """Handles chat completions and per-turn telemetry."""

    def __init__(self) -> None:
        self.provider = "azure"
        self.model = settings.MODEL_NAME

        # Gpt4o pricing defaults. When using other models adjust via env.
        self.prompt_cost_per_1k = settings.PROMPT_COST_PER_1K
        self.completion_cost_per_1k = settings.COMPLETION_COST_PER_1K

        openai_api_key = OPENAI_CONFIG["api_key"]
        openai_base_url = OPENAI_CONFIG["base_url"]
        openai_model = OPENAI_CONFIG["model"] or self.model

        if openai_api_key:
            client_kwargs = {"api_key": openai_api_key}
            if openai_base_url:
                client_kwargs["base_url"] = openai_base_url
            self.client = OpenAI(**client_kwargs)
            self.provider = "openai"
            self.model = openai_model
            logger.info(
                "ChatService using OpenAI-compatible endpoint "
                f"(base_url={openai_base_url or 'https://api.openai.com/v1'}, model={self.model})"
            )
        else:
            if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_KEY:
                raise RuntimeError(
                    "No OpenAI credentials configured. Provide OPENAI_API_KEY or Azure OpenAI settings."
                )
            self.client = AzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_KEY,
                api_version=settings.AZURE_OPENAI_VERSION,
            )
            logger.info(
                "ChatService using Azure OpenAI endpoint "
                f"(endpoint={settings.AZURE_OPENAI_ENDPOINT}, deployment={self.model})"
            )

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_cost = (prompt_tokens / 1000) * self.prompt_cost_per_1k
        completion_cost = (completion_tokens / 1000) * self.completion_cost_per_1k
        return prompt_cost + completion_cost

    def chat_streaming(
        self,
        session: ChatSession,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Generator[str, None, MessageMetrics]:
        """
        Streaming chat response. Yields assistant content chunks and returns metrics.
        """
        start_time = time.perf_counter()

        session.add_message(MessageRole.USER, user_message)
        messages = sanitize_messages(session.get_messages_for_api())

        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if self.provider == "openai":
            request_kwargs["stream_options"] = {"include_usage": True}

        stream = self.client.chat.completions.create(**request_kwargs)

        response_parts: list[str] = []
        prompt_tokens = completion_tokens = total_tokens = 0

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice and getattr(choice, "delta", None) and choice.delta.content:
                content = choice.delta.content
                response_parts.append(content)
                yield content

            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                total_tokens = usage.total_tokens or (prompt_tokens + completion_tokens)

        response_text = "".join(response_parts)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        cost_usd = self._calculate_cost(prompt_tokens, completion_tokens)

        tokens_per_second = 0.0
        if latency_ms > 0:
            tokens_per_second = total_tokens / (latency_ms / 1000)

        metrics = MessageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
        )

        session.add_message(MessageRole.ASSISTANT, response_text)
        session.metrics.add_message(metrics)

        return metrics

    def chat_non_streaming(
        self,
        session: ChatSession,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Tuple[str, MessageMetrics]:
        """
        Synchronous chat completion that returns the full assistant message and metrics.
        """
        start_time = time.perf_counter()

        session.add_message(MessageRole.USER, user_message)
        messages = sanitize_messages(session.get_messages_for_api())

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        assistant_message = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens or 0
        completion_tokens = response.usage.completion_tokens or 0
        total_tokens = response.usage.total_tokens or (prompt_tokens + completion_tokens)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        cost_usd = self._calculate_cost(prompt_tokens, completion_tokens)

        tokens_per_second = 0.0
        if latency_ms > 0:
            tokens_per_second = total_tokens / (latency_ms / 1000)

        metrics = MessageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
        )

        session.add_message(MessageRole.ASSISTANT, assistant_message)
        session.metrics.add_message(metrics)

        return assistant_message, metrics


_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
