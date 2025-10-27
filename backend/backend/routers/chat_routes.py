"""
Task 3.1: Chat API Endpoints

Endpoints:
- POST /api/chat/message - Non-streaming chat
- POST /api/chat/stream - Streaming chat (SSE)
- GET /api/chat/history - Get conversation history
- DELETE /api/chat/history - Clear history
- GET /api/chat/metrics - Get chat metrics
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from backend.models.chat_schemas import ChatRequest, ChatResponse, ChatMessage, ChatHistory
from backend.services.chat_service import get_chat_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """
    Non-streaming chat completion

    Returns complete response with token counts and cost
    """
    try:
        service = get_chat_service()
        response = await service.chat_completion(
            user_message=request.message,
            max_history=request.max_history or 10
        )
        return response
    except Exception as e:
        logger.error(f"Chat message failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat completion using Server-Sent Events

    Returns chunked response as it's generated
    """
    try:
        service = get_chat_service()

        async def generate():
            """Generate SSE events"""
            try:
                async for chunk in service.chat_completion_stream(
                    user_message=request.message,
                    max_history=request.max_history or 10
                ):
                    yield {
                        "event": "message",
                        "data": chunk,
                    }

                # Send completion event
                yield {
                    "event": "done",
                    "data": "[DONE]",
                }
            except Exception as e:
                logger.error(f"Stream generation failed: {e}")
                yield {
                    "event": "error",
                    "data": str(e),
                }

        return EventSourceResponse(generate())

    except Exception as e:
        logger.error(f"Chat stream failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=ChatHistory)
async def get_history():
    """Get conversation history"""
    try:
        service = get_chat_service()
        messages = service.get_history()
        return ChatHistory(
            messages=messages,
            total_messages=len(messages)
        )
    except Exception as e:
        logger.error(f"Get history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history")
async def clear_history():
    """Clear conversation history"""
    try:
        service = get_chat_service()
        service.clear_history()
        return {"message": "History cleared successfully"}
    except Exception as e:
        logger.error(f"Clear history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """
    Get chat metrics and telemetry

    Returns aggregated metrics from Prometheus counters
    """
    try:
        from backend.services.metrics import (
            llm_token_usage_counter,
            llm_request_counter,
            llm_cost_counter,
        )

        # Get metrics from Prometheus
        # Note: This is a simplified version - in production you'd query Prometheus API
        metrics = {
            "message": "Metrics are available at /metrics endpoint (Prometheus format)",
            "endpoint": "/metrics",
            "note": "Use Prometheus/Grafana for visualization"
        }

        return metrics
    except Exception as e:
        logger.error(f"Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
