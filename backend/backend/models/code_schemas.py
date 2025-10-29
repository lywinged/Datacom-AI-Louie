"""
Data models for Code Assistant (Task 3.4).
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class Language(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    RUST = "rust"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    BASH = "bash"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"


class TestFramework(str, Enum):
    """Supported test frameworks"""
    PYTEST = "pytest"
    CARGO = "cargo"
    JEST = "jest"
    GO_TEST = "go test"
    JUNIT = "junit"


class CodeRequest(BaseModel):
    """Request to generate code"""
    task: str = Field(..., description="Natural language task description", min_length=1)
    language: Language = Field(..., description="Programming language")
    test_framework: Optional[TestFramework] = Field(None, description="Test framework to use")
    max_retries: int = Field(default=3, description="Maximum retry attempts on test failure")
    stream_progress: bool = Field(default=True, description="Stream progress updates")
    include_samples: Optional[bool] = Field(
        default=False,
        description="If True, evaluate sample assertions and return outputs (may slow execution)",
    )


class TestResult(BaseModel):
    """Result from running tests"""
    passed: bool = Field(..., description="Whether tests passed")
    stdout: str = Field(default="", description="Standard output from tests")
    stderr: str = Field(default="", description="Standard error from tests")
    exit_code: int = Field(..., description="Exit code from test command")
    execution_time_ms: float = Field(..., description="Test execution time in milliseconds")
    samples: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Sample evaluations captured from test assertions",
    )


class RetryAttempt(BaseModel):
    """A single retry attempt"""
    attempt_number: int = Field(..., description="Attempt number (1-indexed)")
    code: str = Field(..., description="Code generated in this attempt")
    test_result: TestResult = Field(..., description="Test result")
    error_analysis: Optional[str] = Field(None, description="Analysis of the error")
    fix_applied: Optional[str] = Field(None, description="Description of fix applied")
    plan_summary: Optional[str] = Field(
        default=None,
        description="High-level summary of the fix plan followed for this attempt",
    )
    plan_steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step actions derived from the analysis",
    )


class CodeResponse(BaseModel):
    """Response from code generation"""
    code: str = Field(..., description="Final generated code")
    language: Language = Field(..., description="Programming language used")
    test_passed: bool = Field(..., description="Whether final code passed tests")
    final_test_result: TestResult = Field(..., description="Final test execution result")
    retry_attempts: List[RetryAttempt] = Field(default=[], description="All retry attempts")
    total_retries: int = Field(..., description="Total number of retries")
    generation_time_ms: float = Field(..., description="Total generation time in milliseconds")
    tokens_used: int = Field(default=0, description="Total tokens used")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    token_usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Detailed token usage {'prompt': int, 'completion': int, 'total': int}",
    )
    token_cost_usd: Optional[float] = Field(
        default=None,
        description="Estimated LLM cost in USD (same as cost_usd for compatibility)",
    )
    initial_plan_summary: Optional[str] = Field(
        default=None,
        description="Summary of the LLM's initial plan before coding",
    )
    initial_plan_steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step plan generated before coding begins",
    )
    samples: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Sample evaluations from the final successful run",
    )
    learning: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Learning system feedback with reward, strategy, and breakdown",
    )


class ProgressUpdate(BaseModel):
    """Streaming progress update"""
    step: str = Field(..., description="Current step")
    message: str = Field(..., description="Progress message")
    attempt: Optional[int] = Field(None, description="Current attempt number")
    test_passed: Optional[bool] = Field(None, description="Test result if available")


class CodeMetrics(BaseModel):
    """Metrics for code assistant performance"""
    total_requests: int
    success_rate: float
    avg_retries: float
    avg_generation_time_ms: float
    languages_used: dict
