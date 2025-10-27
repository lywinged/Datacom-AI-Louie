"""
API routes for Code Assistant (Task 3.4).
"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.models.code_schemas import CodeRequest, CodeResponse, CodeMetrics
from backend.services.code_assistant import get_code_assistant

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """
    Generate code with automated testing and self-healing.

    The assistant will:
    1. Generate initial code from natural language description
    2. Run automated tests
    3. If tests fail, analyze errors and fix the code
    4. Retry up to max_retries times
    5. Return final code with test results

    Args:
        request: Code generation request

    Returns:
        Generated code with test results and retry history
    """
    try:
        logger.info(f"💻 Code generation request: {request.task[:100]}... ({request.language})")

        assistant = get_code_assistant()
        response = await assistant.generate_code(request)

        logger.info(
            f"✅ Code generated - tests passed: {response.test_passed}, "
            f"retries: {response.total_retries}"
        )

        return response

    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"❌ Code generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for Code Assistant service.

    Returns:
        Service health status
    """
    try:
        assistant = get_code_assistant()

        return JSONResponse(content={
            "status": "healthy",
            "service": "code_assistant",
            "model": assistant.model_name
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


@router.get("/metrics", response_model=CodeMetrics)
async def get_metrics():
    """
    Get Code Assistant performance metrics.

    Returns:
        Code assistant performance metrics
    """
    # Placeholder - in production, track these in a database
    return CodeMetrics(
        total_requests=0,
        success_rate=0.0,
        avg_retries=0.0,
        avg_generation_time_ms=0.0,
        languages_used={}
    )
