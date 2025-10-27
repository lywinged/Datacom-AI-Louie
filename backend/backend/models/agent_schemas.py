"""
Data models for Planning Agent (Task 3.3).
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class TripConstraints(BaseModel):
    """User constraints for trip planning"""
    budget: Optional[float] = Field(None, description="Maximum budget")
    currency: str = Field(default="USD", description="Currency code (USD, NZD, etc)")
    days: Optional[int] = Field(None, description="Number of days")
    origin_city: Optional[str] = Field(None, description="Departure city")
    destination_city: Optional[str] = Field(None, description="Destination city")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    preferences: Optional[List[str]] = Field(default=[], description="User preferences (beach, museum, etc)")


class PlanRequest(BaseModel):
    """Request to create a trip plan"""
    prompt: str = Field(..., description="Natural language trip request", min_length=1)
    constraints: Optional[TripConstraints] = Field(default=None, description="Trip constraints")
    max_iterations: int = Field(default=10, description="Max agent iterations")


class ToolCall(BaseModel):
    """A single tool call made by the agent"""
    tool_name: str = Field(..., description="Name of the tool called")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if tool failed")
    execution_time_ms: float = Field(..., description="Tool execution time in milliseconds")


class ReasoningStep(BaseModel):
    """A reasoning step in the agent's scratch-pad"""
    step_number: int = Field(..., description="Step number")
    thought: str = Field(..., description="Agent's reasoning")
    action: Optional[str] = Field(None, description="Action to take")
    observation: Optional[str] = Field(None, description="Observation from action")


class FlightInfo(BaseModel):
    """Flight information"""
    airline: str
    flight_number: str
    departure_time: str
    arrival_time: str
    price: float
    currency: str
    duration_hours: float
    stops: Optional[int] = 0
    original_price: Optional[float] = Field(
        default=None,
        description="Original price before conversion to itinerary currency",
    )
    original_currency: Optional[str] = Field(
        default=None,
        description="Original currency code before conversion",
    )


class WeatherInfo(BaseModel):
    """Weather information"""
    date: str
    temperature_celsius: float
    condition: str
    precipitation_chance: float


class AttractionInfo(BaseModel):
    """Tourist attraction information"""
    name: str
    category: str
    rating: float
    price_range: str
    description: str


class TripItinerary(BaseModel):
    """Complete trip itinerary"""
    destination: str
    flights: List[FlightInfo] = Field(default=[])
    weather_forecast: List[WeatherInfo] = Field(default=[])
    attractions: List[AttractionInfo] = Field(default=[])
    total_cost: float
    currency: str
    daily_plan: List[Dict[str, Any]] = Field(default=[])
    total_cost_usd: Optional[float] = Field(
        default=None,
        description="Total cost converted to USD for consistent reporting"
    )
    total_cost_nzd: Optional[float] = Field(None, description="Total cost normalized to NZD for learning")
    cost_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Detailed cost breakdown in itinerary currency",
    )
    fx_rates: Optional[Dict[str, float]] = Field(
        default=None,
        description="Exchange rates used relative to itinerary currency",
    )
    currency_note: Optional[str] = Field(
        default=None,
        description="Notes about currency conversions applied",
    )


class PlanResponse(BaseModel):
    """Response from planning agent"""
    itinerary: TripItinerary = Field(..., description="Generated trip itinerary")
    reasoning_trace: List[ReasoningStep] = Field(..., description="Agent's reasoning steps")
    tool_calls: List[ToolCall] = Field(..., description="All tool calls made")
    total_iterations: int = Field(..., description="Total iterations used")
    planning_time_ms: float = Field(..., description="Total planning time in milliseconds")
    constraints_satisfied: bool = Field(..., description="Whether all constraints were satisfied")
    constraint_violations: List[str] = Field(default=[], description="List of violated constraints")
    constraint_satisfaction: Optional[float] = Field(None, description="Constraint satisfaction score (0..1) for learning")
    tool_errors_count: int = Field(default=0, description="Number of tool errors for learning")
    strategy_used: Optional[Dict[str, Any]] = Field(None, description="Strategy selected by learning system")
    llm_token_usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Aggregated LLM token usage {'prompt': int, 'completion': int, 'total': int}",
    )
    llm_cost_usd: Optional[float] = Field(default=None, description="Estimated LLM cost in USD for planning")


class AgentMetrics(BaseModel):
    """Metrics for agent performance"""
    total_plans: int
    success_plans: int
    partial_plans: int
    failed_plans: int
    success_rate: float = Field(
        default=0.0,
        description="Overall success rate as a percentage (0-100)"
    )
    avg_planning_time_ms: float
    avg_tool_calls_per_plan: float
    constraint_satisfaction_rate: float
    tool_success_rate: float
    total_cost_usd: float = Field(
        default=0.0,
        description="Cumulative reported cost across all plans (normalized to USD)"
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent planning runs with outcome and timings"
    )
