"""
Data models for Chat API (Task 3.1)
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Message timestamp")


class ChatRequest(BaseModel):
    """Request for chat completion"""
    message: str = Field(..., description="User message", min_length=1)
    stream: bool = Field(default=True, description="Whether to stream the response")
    max_history: Optional[int] = Field(default=10, description="Max conversation history to include")


class ChatResponse(BaseModel):
    """Response from chat completion"""
    message: str = Field(..., description="Assistant response")
    prompt_tokens: int = Field(..., description="Number of prompt tokens used")
    completion_tokens: int = Field(..., description="Number of completion tokens used")
    total_tokens: int = Field(..., description="Total tokens used")
    cost_usd: float = Field(..., description="Estimated cost in USD")
    latency_ms: float = Field(..., description="Response latency in milliseconds")


class ChatHistory(BaseModel):
    """Chat conversation history"""
    messages: List[ChatMessage] = Field(default_factory=list, description="List of messages")
    total_messages: int = Field(default=0, description="Total number of messages")


class ChatMetrics(BaseModel):
    """Chat metrics and telemetry"""
    total_requests: int = Field(default=0, description="Total number of chat requests")
    total_tokens: int = Field(default=0, description="Total tokens consumed")
    total_cost_usd: float = Field(default=0.0, description="Total estimated cost in USD")
    average_latency_ms: float = Field(default=0.0, description="Average latency in milliseconds")
    last_request: Optional[datetime] = Field(default=None, description="Last request timestamp")
