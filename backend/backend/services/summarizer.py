"""
Document Summarization Service

Provides lightweight summarization for long knowledge chunks to reduce token count
while preserving semantic meaning for RAG retrieval.

Strategies:
1. Extractive summarization (fast, no LLM needed) - using TextRank or similar
2. LLM-based summarization (higher quality, costs tokens)
3. Hybrid approach
"""

import re
from typing import Literal
from services.token_counter import TokenCounter
from services.llm_tracker import LLMTracker


class DocumentSummarizer:
    """
    Handles document and chunk summarization for knowledge base optimization.
    """

    def __init__(self):
        self.token_counter = TokenCounter()
        self.llm_tracker = LLMTracker()

    def extractive_summary(
        self,
        text: str,
        max_sentences: int = 3,
        max_tokens: int | None = None
    ) -> str:
        """
        Simple extractive summarization using sentence ranking.
        No LLM needed - very fast and free.

        Algorithm:
        1. Split into sentences
        2. Score sentences by position, length, and keyword density
        3. Select top-N sentences
        4. Return in original order

        Args:
            text: Input text to summarize
            max_sentences: Maximum number of sentences to keep
            max_tokens: Optional token limit for summary

        Returns:
            Summarized text
        """
        # Split into sentences (simple regex-based)
        sentences = re.split(r'[。！？\.\!\?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            return text

        # Score each sentence
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            score = 0.0

            # Position bonus (first and last sentences often important)
            if idx == 0:
                score += 1.0
            elif idx == len(sentences) - 1:
                score += 0.5

            # Length bonus (moderate length preferred)
            length = len(sentence)
            if 20 < length < 200:
                score += 0.3
            elif length >= 200:
                score += 0.1

            # Keyword density (simple word count)
            words = sentence.split()
            if len(words) > 5:
                score += min(len(words) / 50, 0.5)

            scored_sentences.append((idx, sentence, score))

        # Sort by score and take top N
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        top_sentences = scored_sentences[:max_sentences]

        # Sort back to original order
        top_sentences.sort(key=lambda x: x[0])

        # Build summary
        summary = "。".join(s[1] for s in top_sentences)
        if not summary.endswith(('。', '.', '!', '?', '！', '？')):
            summary += "。"

        # Check token limit if specified
        if max_tokens:
            tokens = self.token_counter.count_tokens(summary)
            if tokens > max_tokens:
                # Recursively reduce sentence count
                return self.extractive_summary(
                    text,
                    max_sentences=max(1, max_sentences - 1),
                    max_tokens=max_tokens
                )

        return summary

    async def llm_summary(
        self,
        text: str,
        model: str = "deepseek-chat",
        max_tokens: int = 200,
        style: Literal["concise", "keywords", "abstract"] = "concise"
    ) -> tuple[str, float]:
        """
        Use LLM to generate high-quality summary.

        Args:
            text: Input text to summarize
            model: LLM model to use
            max_tokens: Maximum tokens for summary
            style: Summarization style
                - concise: Brief summary preserving key points
                - keywords: Extract key phrases and concepts
                - abstract: Academic-style abstract

        Returns:
            Tuple of (summary, cost_in_usd)
        """
        import httpx
        import os
        import time

        # Choose prompt based on style
        if style == "keywords":
            instruction = "Extract the key concepts, names, and important phrases from the following text. List them concisely:"
        elif style == "abstract":
            instruction = "Write a brief academic-style abstract summarizing the main points of this text:"
        else:  # concise
            instruction = f"Summarize the following text in {max_tokens} tokens or less, preserving the most important information:"

        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries."},
            {"role": "user", "content": f"{instruction}\n\n{text}"}
        ]

        # Call LLM (reuse existing LLM endpoint configuration)
        api_base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        start_time = time.perf_counter()

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,  # Lower temperature for more focused summaries
                }
            )
            response.raise_for_status()
            data = response.json()

        summary = data["choices"][0]["message"]["content"].strip()
        duration = time.perf_counter() - start_time

        # Track the LLM call
        usage = await self.llm_tracker.track_chat_completion(
            model=model,
            messages=messages,
            completion=summary,
            duration=duration,
            endpoint="summarization"
        )

        cost = self.llm_tracker.token_counter.estimate_cost(usage)

        return summary, cost

    async def hybrid_summary(
        self,
        text: str,
        token_threshold: int = 500,
        target_tokens: int = 200,
        use_llm_for_long: bool = True
    ) -> tuple[str, float, str]:
        """
        Hybrid approach: Use extractive for short texts, LLM for long texts.

        Args:
            text: Input text
            token_threshold: If text exceeds this, use LLM (if enabled)
            target_tokens: Target token count for summary
            use_llm_for_long: Whether to use LLM for texts exceeding threshold

        Returns:
            Tuple of (summary, cost_in_usd, method_used)
        """
        current_tokens = self.token_counter.count_tokens(text)

        # If already short enough, return as-is
        if current_tokens <= target_tokens:
            return text, 0.0, "none"

        # If under threshold or LLM disabled, use extractive
        if current_tokens < token_threshold or not use_llm_for_long:
            summary = self.extractive_summary(text, max_tokens=target_tokens)
            return summary, 0.0, "extractive"

        # Use LLM for long documents
        summary, cost = await self.llm_summary(text, max_tokens=target_tokens)
        return summary, cost, "llm"

    def should_summarize(
        self,
        text: str,
        min_tokens: int = 300,
        min_reduction_ratio: float = 0.3
    ) -> bool:
        """
        Determine if a text chunk would benefit from summarization.

        Args:
            text: Text to check
            min_tokens: Minimum token count to consider summarization
            min_reduction_ratio: Minimum reduction ratio to make it worthwhile

        Returns:
            True if summarization is recommended
        """
        tokens = self.token_counter.count_tokens(text)

        # Too short to summarize
        if tokens < min_tokens:
            return False

        # Estimate potential reduction (rough heuristic)
        # Extractive typically gets 40-60% reduction
        # LLM can get 60-80% reduction
        estimated_summary_tokens = tokens * (1 - min_reduction_ratio)

        # Worth summarizing if we can save significant tokens
        return (tokens - estimated_summary_tokens) > 100


# Singleton instance
_summarizer_instance: DocumentSummarizer | None = None


def get_summarizer() -> DocumentSummarizer:
    """Get or create the summarizer singleton."""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = DocumentSummarizer()
    return _summarizer_instance
