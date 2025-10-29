"""
Token accounting utilities.

Provides precise tracking of token usage for LLM calls, including:
- Exact counting via tiktoken
- Model-specific token estimators
- Hooks for Prometheus metrics
- Optional database persistence
"""

import tiktoken
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Mapping between model identifiers and tiktoken encodings
MODEL_ENCODINGS = {
    # OpenAI models
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "Gpt4o": "o200k_base",
    "Gpt4o-mini": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",

    # Anthropic models (approximate with Claude tokenizer)
    "claude-3-opus": "cl100k_base",  # approximate
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "claude-3-5-sonnet": "cl100k_base",

    # DeepSeek models
    "deepseek-chat": "cl100k_base",
    "deepseek-v3": "cl100k_base",
    "deepseek-v3-250324": "cl100k_base",
    "deepseek-v3-1-terminus": "cl100k_base",

    # Default encoding fallback
    "default": "cl100k_base",
}


@dataclass
class TokenUsage:
    """Token usage statistics for a single API call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
        }


class TokenCounter:
    """Utility for counting tokens across LLM interactions."""

    def __init__(self):
        self._encoders: Dict[str, tiktoken.Encoding] = {}

    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Return the encoding for the given model name."""
        encoding_name = MODEL_ENCODINGS.get(model, MODEL_ENCODINGS["default"])

        if encoding_name not in self._encoders:
            try:
                self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to get encoding {encoding_name}: {e}, using cl100k_base")
                self._encoders[encoding_name] = tiktoken.get_encoding("cl100k_base")

        return self._encoders[encoding_name]

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Compute the number of tokens in a plain text string.

        Args:
            text: Text to evaluate
            model: Model identifier

        Returns:
            Token count estimate
        """
        if not text:
            return 0

        try:
            encoder = self._get_encoder(model)
            return len(encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Simple fallback heuristic: 1 token ≈ 4 characters
            return len(text) // 4

    def count_messages_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4"
    ) -> int:
        """
        Compute token usage for a ChatCompletion-style payload.

        Mirrors OpenAI's guidance:
        - Each message has a fixed overhead (role, separators)
        - Every field contributes tokens

        Args:
            messages: Sequence of role/content dictionaries
            model: Model identifier

        Returns:
            Total token count
        """
        try:
            encoder = self._get_encoder(model)

            # Model-specific overhead
            if model.startswith("gpt-4") or model.startswith("gpt-3.5"):
                tokens_per_message = 3  # <|start|>role<|end|>content
                tokens_per_name = 1
            else:
                tokens_per_message = 3
                tokens_per_name = 1

            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    if value:
                        num_tokens += len(encoder.encode(str(value)))
                        if key == "name":
                            num_tokens += tokens_per_name

            num_tokens += 3  # Fixed overhead per reply

            return num_tokens

        except Exception as e:
            logger.error(f"Error counting message tokens: {e}")
            # Estimate tokens via character count when encoder lookup fails
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            return total_chars // 4

    def create_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> TokenUsage:
        """Instantiate a TokenUsage dataclass."""
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=model,
            timestamp=datetime.utcnow(),
        )

    def count_chat_completion(
        self,
        messages: List[Dict[str, str]],
        completion: str,
        model: str = "gpt-4"
    ) -> TokenUsage:
        """
        Calculate token usage for a single ChatCompletion call.

        Args:
            messages: Input message list
            completion: Model completion text
            model: Model identifier

        Returns:
            TokenUsage dataclass with prompt/completion counts
        """
        prompt_tokens = self.count_messages_tokens(messages, model)
        completion_tokens = self.count_tokens(completion, model)

        return self.create_usage(prompt_tokens, completion_tokens, model)

    def estimate_cost(self, usage: TokenUsage) -> float:
        """
        Estimate USD cost for a given token usage snapshot.

        Pricing reference (2024):
        - GPT-4: $0.03/1K prompt, $0.06/1K completion
        - GPT-4 Turbo: $0.01/1K prompt, $0.03/1K completion
        - Gpt4o: $0.005/1K prompt, $0.015/1K completion
        - GPT-3.5 Turbo: $0.0005/1K prompt, $0.0015/1K completion

        Args:
            usage: TokenUsage dataclass

        Returns:
            Cost in USD
        """
        # Pricing table (per 1,000 tokens)
        pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "Gpt4o": {"prompt": 0.005, "completion": 0.015},
            "Gpt4o-mini": {"prompt": 0.00015, "completion": 0.0006},
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
            "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
            "claude-3-5-sonnet": {"prompt": 0.003, "completion": 0.015},
            "deepseek-chat": {"prompt": 0.0001, "completion": 0.0002},
            "deepseek-v3": {"prompt": 0.00027, "completion": 0.0011},
            "deepseek-v3-250324": {"prompt": 0.00027, "completion": 0.0011},
            "deepseek-v3-1-terminus": {"prompt": 0.00027, "completion": 0.0011},
        }

        # Match the appropriate pricing tier
        model_pricing = None
        for model_key, price in pricing.items():
            if usage.model.startswith(model_key):
                model_pricing = price
                break

        if not model_pricing:
            logger.warning(f"No pricing data for model {usage.model}")
            return 0.0

        prompt_cost = (usage.prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1000) * model_pricing["completion"]

        return prompt_cost + completion_cost


# Global singleton instance
_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Return the global TokenCounter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


# Convenience wrappers
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in raw text."""
    return get_token_counter().count_tokens(text, model)


def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """Count tokens for a list of chat messages."""
    return get_token_counter().count_messages_tokens(messages, model)


def count_chat_completion(
    messages: List[Dict[str, str]],
    completion: str,
    model: str = "gpt-4"
) -> TokenUsage:
    """Compute token usage for a ChatCompletion call."""
    return get_token_counter().count_chat_completion(messages, completion, model)


def estimate_cost(usage: TokenUsage) -> float:
    """Estimate token cost in USD."""
    return get_token_counter().estimate_cost(usage)
