"""
Chat data models for Task 3.1
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum


class MessageRole(str, Enum):
    """Message role types"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """Single chat message"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MessageMetrics:
    """Metrics for a single message (per-turn statistics)"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    tokens_per_second: float
    timestamp: datetime = field(default_factory=datetime.now)

    def format_stats(self) -> str:
        """Format as required: [stats] prompt=8  completion=23  cost=$0.000146  latency=623 ms"""
        return (
            f"[stats] prompt={self.prompt_tokens}  "
            f"completion={self.completion_tokens}  "
            f"cost=${self.cost_usd:.6f}  "
            f"latency={self.latency_ms}ms  "
            f"throughput={self.tokens_per_second:.1f} tok/s"
        )

    def to_dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SessionMetrics:
    """Cumulative session metrics (for dashboard/analytics)"""
    total_messages: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    average_latency_ms: float = 0.0
    min_latency_ms: Optional[int] = None
    max_latency_ms: Optional[int] = None
    average_tokens_per_second: float = 0.0
    message_history: List[MessageMetrics] = field(default_factory=list)

    def add_message(self, metrics: MessageMetrics):
        """Add a message's metrics to session totals"""
        self.total_messages += 1
        self.total_prompt_tokens += metrics.prompt_tokens
        self.total_completion_tokens += metrics.completion_tokens
        self.total_tokens += metrics.total_tokens
        self.total_cost_usd += metrics.cost_usd

        # Update average latency
        if self.total_messages == 1:
            self.average_latency_ms = float(metrics.latency_ms)
            self.average_tokens_per_second = metrics.tokens_per_second
        else:
            self.average_latency_ms = (
                (self.average_latency_ms * (self.total_messages - 1) + metrics.latency_ms)
                / self.total_messages
            )
            self.average_tokens_per_second = (
                (self.average_tokens_per_second * (self.total_messages - 1) + metrics.tokens_per_second)
                / self.total_messages
            )

        # Update min/max latency
        if self.min_latency_ms is None or metrics.latency_ms < self.min_latency_ms:
            self.min_latency_ms = metrics.latency_ms
        if self.max_latency_ms is None or metrics.latency_ms > self.max_latency_ms:
            self.max_latency_ms = metrics.latency_ms

        # Keep history for analytics
        self.message_history.append(metrics)

    def to_dict(self):
        return {
            "total_messages": self.total_messages,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "average_latency_ms": self.average_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "average_tokens_per_second": self.average_tokens_per_second,
        }

    def format_summary(self) -> str:
        """Format session summary"""
        return (
            f"\n{'='*60}\n"
            f"Session Summary:\n"
            f"  Messages: {self.total_messages}\n"
            f"  Total Tokens: {self.total_tokens} "
            f"(prompt: {self.total_prompt_tokens}, completion: {self.total_completion_tokens})\n"
            f"  Total Cost: ${self.total_cost_usd:.6f}\n"
            f"  Avg Latency: {self.average_latency_ms:.0f}ms "
            f"(min: {self.min_latency_ms}ms, max: {self.max_latency_ms}ms)\n"
            f"{'='*60}"
        )


@dataclass
class ChatSession:
    """Complete chat session with history and metrics"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    max_history: int = 10  # Keep last N messages
    created_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: MessageRole, content: str) -> Message:
        """Add a message to history"""
        msg = Message(role=role, content=content)
        self.messages.append(msg)

        # Keep only last N messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

        return msg

    def get_messages_for_api(self) -> List[dict]:
        """Get messages in format for OpenAI API"""
        return [{"role": msg.role.value, "content": msg.content} for msg in self.messages]

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at.isoformat()
        }
