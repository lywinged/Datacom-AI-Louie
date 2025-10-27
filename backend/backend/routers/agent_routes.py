"""
API routes for Planning Agent (Task 3.3).
"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.models.agent_schemas import PlanRequest, PlanResponse, AgentMetrics
from backend.services.planning_agent import get_planning_agent
from backend.services.agent_metrics_store import planning_metrics_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/plan", response_model=PlanResponse)
async def create_trip_plan(request: PlanRequest):
    """
    Create a trip plan using the autonomous planning agent.

    The agent will:
    1. Search for flights
    2. Check weather forecast
    3. Find tourist attractions
    4. Compile a complete itinerary

    Args:
        request: Plan request with user prompt and constraints

    Returns:
        Complete trip plan with itinerary, reasoning trace, and tool calls
    """
    try:
        logger.info(f"📝 Planning request: {request.prompt[:100]}...")

        agent = get_planning_agent()
        response = await agent.create_plan(request)

        logger.info(
            f"✅ Plan created - {len(response.tool_calls)} tools used, "
            f"constraints satisfied: {response.constraints_satisfied}"
        )

        return response

    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"❌ Planning failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for Planning Agent service.

    Returns:
        Service health status
    """
    try:
        agent = get_planning_agent()

        return JSONResponse(content={
            "status": "healthy",
            "service": "planning_agent",
            "model": agent.model_name
        })

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.get("/metrics", response_model=AgentMetrics)
async def get_metrics():
    """
    Get Planning Agent performance metrics.

    Returns:
        Agent performance metrics
    """
    summary = planning_metrics_store.get_summary()
    return AgentMetrics(**summary)


@router.post("/metrics/reset")
async def reset_metrics():
    """
    Reset the planning metrics store (aggregated across sessions).
    """
    planning_metrics_store.reset()
    return {"status": "reset"}
