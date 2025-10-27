"""
Planning Agent Service (Task 3.3).
Autonomous agent with multi-tool orchestration for trip planning.
"""
import os
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI

from backend.config.settings import settings, OPENAI_CONFIG
from backend.models.agent_schemas import (
    PlanRequest, PlanResponse, TripItinerary, ReasoningStep, ToolCall,
    FlightInfo, WeatherInfo, AttractionInfo
)
from backend.services.agent_tools import get_agent_tools
from backend.services.autoplan_learn.adapter import AutoPlanAdapter
from backend.services.token_counter import get_token_counter, TokenUsage
from backend.services.agent_metrics_store import planning_metrics_store
from backend.services.fx_service import get_fx_service
from backend.utils.openai import sanitize_messages

logger = logging.getLogger(__name__)

# Currency conversion rates to NZD (simplified)
CURRENCY_TO_NZD = {
    "NZD": 1.0,
    "USD": 1.65,
    "AUD": 1.08,
    "GBP": 2.15,
    "EUR": 1.82,
    "JPY": 0.011,
}

# Basic lat/lon data for popular cities so we can do quick feasibility checks
CITY_COORDINATES = {
    "auckland": (-36.8509, 174.7645),
    "wellington": (-41.2865, 174.7762),
    "sydney": (-33.8688, 151.2093),
    "melbourne": (-37.8136, 144.9631),
    "beijing": (39.9042, 116.4074),
    "shanghai": (31.2304, 121.4737),
    "hong kong": (22.3193, 114.1694),
    "singapore": (1.3521, 103.8198),
    "tokyo": (35.6762, 139.6503),
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "london": (51.5072, -0.1276),
    "paris": (48.8566, 2.3522),
    "dubai": (25.2048, 55.2708),
    "bangkok": (13.7563, 100.5018),
    "delhi": (28.6139, 77.2090),
    "san francisco": (37.7749, -122.4194),
    "berlin": (52.5200, 13.4050),
    "mumbai": (19.0760, 72.8777),
}

CITY_COUNTRY = {
    "auckland": "new zealand",
    "wellington": "new zealand",
    "sydney": "australia",
    "melbourne": "australia",
    "beijing": "china",
    "shanghai": "china",
    "hong kong": "china",
    "singapore": "singapore",
    "tokyo": "japan",
    "new york": "united states",
    "los angeles": "united states",
    "london": "united kingdom",
    "paris": "france",
    "dubai": "united arab emirates",
    "bangkok": "thailand",
    "delhi": "india",
    "san francisco": "united states",
    "berlin": "germany",
    "mumbai": "india",
}

COUNTRY_ALIASES = {
    "new zealand": "new zealand",
    "nz": "new zealand",
    "aotearoa": "new zealand",
    "china": "china",
    "prc": "china",
    "united states": "united states",
    "usa": "united states",
    "us": "united states",
    "america": "united states",
    "united kingdom": "united kingdom",
    "uk": "united kingdom",
    "england": "united kingdom",
    "great britain": "united kingdom",
    "australia": "australia",
    "singapore": "singapore",
    "japan": "japan",
    "france": "france",
    "thailand": "thailand",
    "india": "india",
    "united arab emirates": "united arab emirates",
    "uae": "united arab emirates",
}

COUNTRY_CURRENCY = {
    "new zealand": "NZD",
    "australia": "AUD",
    "china": "CNY",
    "singapore": "SGD",
    "japan": "JPY",
    "united states": "USD",
    "united kingdom": "GBP",
    "france": "EUR",
    "united arab emirates": "AED",
    "thailand": "THB",
    "india": "INR",
    "germany": "EUR",
}

# Baseline daily estimates (USD) used for conversion
BASE_ACCOM_NIGHT_USD = 120.0
BASE_MEALS_PER_DAY_USD = 50.0
BASE_MISC_PER_DAY_USD = 35.0

DISTANCE_FLIGHT_THRESHOLD_KM = 700  # below this prefer ground transport
MAX_ONE_DAY_DISTANCE_KM = 1000  # realistic cap for same-day round trip
MIN_FLIGHT_COST_PER_KM_USD = 0.12  # conservative lower bound for fares per km (USD baseline)

class PlanningAgent:
    """Autonomous planning agent with tool orchestration"""

    def __init__(self, enable_learning: bool = True):
        """Initialize planning agent with OpenAI client and learning system"""
        api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY or OPENAI_CONFIG.get("api_key")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        base_url = os.getenv("OPENAI_BASE_URL") or OPENAI_CONFIG.get("base_url")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)
        self.token_counter = get_token_counter()
        self.fx = get_fx_service()
        self.model_name = (
            os.getenv("OPENAI_MODEL")
            or OPENAI_CONFIG.get("model")
            or settings.MODEL_NAME
            or "gpt-3.5-turbo"
        )
        self.tools = get_agent_tools()

        # Initialize learning system
        self.enable_learning = enable_learning
        self.learner = None
        if enable_learning:
            memory_path = os.getenv("AGENT_MEMORY_PATH", "data/agent_experiences.jsonl")
            self.learner = AutoPlanAdapter(memory_path)
            logger.info(f"🎓 Learning system enabled (memory: {memory_path})")

        logger.info(f"✅ PlanningAgent initialized with model: {self.model_name}")

    async def create_plan(self, request: PlanRequest) -> PlanResponse:
        """
        Create a trip plan using autonomous agent with tools.

        Args:
            request: Plan request with prompt and constraints

        Returns:
            Complete plan response with itinerary and reasoning trace
        """
        start_time = time.time()

        logger.info(f"🤖 Planning agent started: {request.prompt[:100]}...")

        # Initialize tracking
        reasoning_steps: List[ReasoningStep] = []
        tool_calls_made: List[ToolCall] = []
        iterations = 0
        tool_errors_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        llm_total_cost_usd = 0.0

        origin_currency = self._prepare_currency(request.constraints)

        # LEARNING INTEGRATION: Choose strategy before planning
        strategy = None
        signature = None
        if self.enable_learning and self.learner:
            # Extract destination city from constraints or prompt
            destination_city = None
            if request.constraints:
                # Try to extract from prompt or use a simple heuristic
                for city in ["Auckland", "London", "Paris", "Tokyo", "Sydney", "Singapore", "New York"]:
                    if city.lower() in request.prompt.lower():
                        destination_city = city
                        break

            budget_nzd = self._convert_to_nzd(
                request.constraints.budget if request.constraints else None,
                origin_currency,
            )

            constraints_dict = {
                "city": destination_city,
                "days": request.constraints.days if request.constraints else None,
                "budget_nzd": budget_nzd,
                "date_range": f"{request.constraints.start_date}..{request.constraints.end_date}"
                    if request.constraints and request.constraints.start_date
                    else None
            }

            tools_available = ["search_flights", "get_weather_forecast", "search_attractions"]

            ctx = self.learner.choose(
                prompt=request.prompt,
                constraints=constraints_dict,
                tools_used=tools_available,
                error_digest=None
            )

            strategy = ctx["strategy"]
            signature = ctx["signature"]

            logger.info(f"🎓 Learning system selected strategy: tool_order={strategy.get('tool_order')}, "
                       f"temp={strategy.get('model_temp')}, k={strategy.get('attractions_k')}")

        is_local_trip = self._is_local_trip(
            request.constraints.origin_city if request.constraints else None,
            request.constraints.destination_city if request.constraints else None,
        )

        system_prompt = self._build_system_prompt(request.constraints, strategy, local_trip=is_local_trip)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt},
        ]

        tools_definitions = self._get_tool_definitions()
        if is_local_trip:
            tools_definitions = [
                tool for tool in tools_definitions
                if tool["function"]["name"] != "search_flights"
            ]

        # Before engaging the LLM loop, run feasibility checks. If infeasible,
        # we return early with explanations instead of wasting tokens.
        feasibility = self._evaluate_feasibility(request)
        if not feasibility["feasible"]:
            logger.warning("❌ Request deemed infeasible: %s", feasibility["issues"])
            itinerary = TripItinerary(
                destination=request.constraints.destination_city if request.constraints else "Unknown",
                flights=[],
                weather_forecast=[],
                attractions=[],
                total_cost=0.0,
                currency=origin_currency,
                daily_plan=[]
            )
            planning_time_ms = (time.time() - start_time) * 1000
            reasoning_steps.append(
                ReasoningStep(
                    step_number=0,
                    thought="Request flagged as infeasible by pre-flight checks",
                    action="feasibility_check",
                    observation=json.dumps({
                        "issues": feasibility["issues"],
                        "recommendations": feasibility["recommendations"],
                    })
                )
            )
            violations = feasibility["issues"][:]
            if feasibility["recommendations"]:
                violations.extend(feasibility["recommendations"])
            return PlanResponse(
                itinerary=itinerary,
                reasoning_trace=reasoning_steps,
                tool_calls=tool_calls_made,
                total_iterations=0,
                planning_time_ms=planning_time_ms,
                constraints_satisfied=False,
                constraint_violations=violations,
                constraint_satisfaction=0.0,
                tool_errors_count=0,
                strategy_used=None,
                llm_token_usage=None,
                llm_cost_usd=0.0,
            )

        try:
            # Agent loop
            while iterations < request.max_iterations:
                iterations += 1

                logger.info(f"🔄 Iteration {iterations}/{request.max_iterations}")

                # Call LLM with function calling (use strategy temperature if available)
                temperature = strategy.get("model_temp", 0.7) if strategy else 0.7
                sanitized_messages = sanitize_messages(messages)

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=sanitized_messages,
                    tools=tools_definitions,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=1500
                )

                assistant_message = response.choices[0].message

                usage = getattr(response, "usage", None)
                if usage:
                    total_prompt_tokens += usage.prompt_tokens
                    total_completion_tokens += usage.completion_tokens
                    usage_obj = TokenUsage(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                        model=self.model_name,
                        timestamp=datetime.utcnow(),
                    )
                    llm_total_cost_usd += self.token_counter.estimate_cost(usage_obj)

                # Add assistant message to history
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') else None
                })

                # Log reasoning step
                if assistant_message.content:
                    reasoning_steps.append(ReasoningStep(
                        step_number=iterations,
                        thought=assistant_message.content[:500],  # Truncate for storage
                        action=None,
                        observation=None
                    ))

                # Check if agent wants to call tools
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    # Execute each tool call
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        logger.info(f"🔧 Calling tool: {tool_name} with args: {tool_args}")

                        # Execute tool
                        tool_result = await self._execute_tool(tool_name, tool_args)

                        # Track tool errors for learning
                        if not tool_result.get("success"):
                            tool_errors_count += 1

                        # Track tool call
                        tool_calls_made.append(ToolCall(
                            tool_name=tool_name,
                            arguments=tool_args,
                            result=tool_result if tool_result.get("success") else None,
                            error=tool_result.get("error"),
                            execution_time_ms=tool_result.get("execution_time_ms", 0)
                        ))

                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps(tool_result or {}) if tool_result is not None else "{}"
                        })

                        # Update reasoning step
                        reasoning_steps.append(ReasoningStep(
                            step_number=iterations,
                            thought=f"Executed {tool_name}",
                            action=tool_name,
                            observation=json.dumps(tool_result)[:200]
                        ))

                # Check if agent is done (no more tool calls)
                elif response.choices[0].finish_reason == "stop":
                    logger.info("✅ Agent finished planning")
                    break

            # Extract final itinerary from conversation
            itinerary = self._extract_itinerary(messages, tool_calls_made, request, origin_currency)

            # Validate constraints
            constraints_satisfied, violations = self._validate_constraints(
                itinerary, request.constraints
            )

            # Calculate constraint satisfaction score for learning (0..1)
            constraint_satisfaction_score = self._calculate_satisfaction_score(
                itinerary, request.constraints, constraints_satisfied
            )

            # Convert total cost to NZD for learning
            total_cost_nzd = self._convert_to_nzd(
                itinerary.total_cost,
                itinerary.currency
            )
            itinerary.total_cost_nzd = total_cost_nzd

            planning_time_ms = (time.time() - start_time) * 1000

            logger.info(f"✅ Planning completed in {planning_time_ms:.1f}ms with {len(tool_calls_made)} tool calls")

            llm_token_usage_dict = None
            if total_prompt_tokens or total_completion_tokens:
                llm_token_usage_dict = {
                    "prompt": total_prompt_tokens,
                    "completion": total_completion_tokens,
                    "total": total_prompt_tokens + total_completion_tokens,
                }

            # LEARNING INTEGRATION: Record results after planning
            if self.enable_learning and self.learner and strategy and signature:
                budget_nzd = self._convert_to_nzd(
                    request.constraints.budget if request.constraints else None,
                    origin_currency
                )

                learning_result = self.learner.record(
                    signature=signature,
                    strategy=strategy,
                    plan_cost=total_cost_nzd,
                    budget_nzd=budget_nzd,
                    tool_errors=tool_errors_count,
                    satisfaction_proxy=constraint_satisfaction_score
                )

                logger.info(f"🎓 Learning recorded: success={learning_result['success']}, "
                           f"reward={learning_result['reward']:.3f}")

            # Persist aggregated metrics (best-effort)
            try:
                planning_metrics_store.record_plan(
                    constraints_satisfied=constraints_satisfied,
                    planning_time_ms=planning_time_ms,
                    tool_calls=len(tool_calls_made),
                    tool_errors=tool_errors_count,
                    token_cost_usd=llm_total_cost_usd if llm_token_usage_dict else 0.0,
                )
            except Exception as metrics_err:
                logger.warning("Failed to record planning metrics: %s", metrics_err)

            return PlanResponse(
                itinerary=itinerary,
                reasoning_trace=reasoning_steps,
                tool_calls=tool_calls_made,
                total_iterations=iterations,
                planning_time_ms=planning_time_ms,
                constraints_satisfied=constraints_satisfied,
                constraint_violations=violations,
                constraint_satisfaction=constraint_satisfaction_score,
                tool_errors_count=tool_errors_count,
                strategy_used=strategy,
                llm_token_usage=llm_token_usage_dict,
                llm_cost_usd=llm_total_cost_usd if llm_token_usage_dict else 0.0,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            try:
                planning_metrics_store.record_failure(
                    planning_time_ms=elapsed_ms,
                    tool_calls=len(tool_calls_made),
                    error=str(e),
                )
            except Exception as metrics_err:
                logger.warning("Failed to record planning failure metrics: %s", metrics_err)

            logger.error(f"❌ Planning failed: {e}", exc_info=True)
            raise

    def _prepare_currency(self, constraints) -> str:
        """Determine itinerary currency based on origin and convert budget if needed."""
        origin_currency = self._infer_origin_currency(constraints)

        if constraints:
            source_currency = (constraints.currency or origin_currency).upper()
            if constraints.budget and source_currency != origin_currency:
                converted_budget = self._convert_currency(constraints.budget, source_currency, origin_currency)
                if converted_budget is not None:
                    constraints.budget = round(converted_budget, 2)
                else:
                    logger.warning(
                        "Failed to convert budget %s %.2f to %s",
                        source_currency,
                        constraints.budget,
                        origin_currency,
                    )
            constraints.currency = origin_currency

        return origin_currency

    def _infer_origin_currency(self, constraints) -> str:
        """Guess currency based on origin city/country with USD fallback."""
        default_currency = "USD"
        if not constraints:
            return default_currency

        if constraints.origin_city:
            normalized_city = constraints.origin_city.lower().strip()
            country = CITY_COUNTRY.get(normalized_city)
            if country:
                country_norm = self._resolve_country_name(country)
                if country_norm:
                    currency = COUNTRY_CURRENCY.get(country_norm)
                    if currency:
                        return currency

        if constraints.currency:
            return constraints.currency.upper()

        return default_currency

    def _resolve_country_name(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        normalized = name.lower().strip()
        return COUNTRY_ALIASES.get(normalized, normalized)

    def _get_fx_rate(self, base_currency: str, target_currency: str) -> Optional[float]:
        """Return FX rate from base -> target."""
        try:
            return self.fx.get_rate(base_currency, target_currency)
        except Exception as exc:
            logger.warning("Failed to get FX rate %s -> %s: %s", base_currency, target_currency, exc)
            return None

    def _convert_and_track(
        self,
        amount: Optional[float],
        source_currency: Optional[str],
        target_currency: str,
        fx_rates: Dict[str, float],
    ) -> Optional[float]:
        converted = self._convert_currency(amount, source_currency, target_currency)
        if (
            converted is not None
            and amount
            and source_currency
            and source_currency.upper() != target_currency
        ):
            rate = converted / float(amount) if amount else None
            if rate:
                fx_rates[source_currency.upper()] = rate
        return converted

    def _build_system_prompt(self, constraints, strategy: Optional[Dict] = None, local_trip: bool = False) -> str:
        """Build system prompt with constraints and strategy"""
        prompt = """You are an expert trip planning agent. Your job is to create detailed trip itineraries using available tools.

Available tools:
- search_flights: Find flights between cities
- get_weather_forecast: Get weather forecast for a location
- search_attractions: Find tourist attractions

"""
        if local_trip:
            prompt += (
                "\nIMPORTANT: This is a local trip (origin and destination are in the same city or country)."
                " Do not search for flights unless no ground transport exists."
                " Focus on activities, weather, local logistics, and ground transport options.\n\n"
            )

        # Add tool ordering guidance from strategy
        if strategy and "tool_order" in strategy:
            tool_order = strategy["tool_order"]
            prompt += f"\nRecommended tool calling sequence: {tool_order}\n"

        # Determine instructions based on local trip status
        flag_local = bool(local_trip)
        if not flag_local and constraints and constraints.origin_city and constraints.destination_city:
            origin_lower = constraints.origin_city.lower().strip()
            dest_lower = constraints.destination_city.lower().strip()
            if origin_lower == dest_lower:
                flag_local = True

        if flag_local:
            prompt += """
Your process:
1. Understand the user's trip requirements
2. SKIP flights search (origin and destination are the same - this is a local trip!)
3. Use get_weather_forecast to check weather conditions for the destination
4. Use search_attractions to find tourist attractions and activities
5. Create a detailed day-by-day itinerary
6. Estimate costs (accommodation, meals, attraction tickets, transportation)
7. Summarize the complete trip plan with:
   - Daily activities with specific attraction names
   - Weather information for each day
   - Estimated costs breakdown (accommodation: ~$X/night, meals: ~$Y/day, activities: ~$Z)
   - Total estimated cost
   - Recommendations and tips

COST ESTIMATION GUIDELINES (for local trips):
- Accommodation: Budget hotels $50-100/night, Mid-range $100-200/night, Luxury $200+/night
- Meals: Budget $20-30/day, Mid-range $40-60/day, Nice restaurants $80+/day
- Local transportation: $10-20/day (public transit, taxis)
- Attraction tickets: $10-30 per major attraction

"""
        else:
            prompt += """
Your process:
1. Understand the user's trip requirements
2. Use search_flights to find suitable flights (check multiple options)
3. Use get_weather_forecast to check weather conditions for the destination
4. Use search_attractions to find tourist attractions and activities
5. Create a detailed day-by-day itinerary
6. Calculate total costs (flights + accommodation + meals + activities)
7. Summarize the complete trip plan with:
   - Selected flights with prices
   - Daily activities with specific attraction names
   - Weather information for each day
   - Estimated costs breakdown (flights: $X, accommodation: ~$Y, meals: ~$Z, activities: ~$W)
   - Total estimated cost
   - Recommendations and tips

COST ESTIMATION GUIDELINES:
- Accommodation: Budget hotels $50-100/night, Mid-range $100-200/night, Luxury $200+/night
- Meals: Budget $20-30/day, Mid-range $40-60/day, Nice restaurants $80+/day
- Local transportation: $10-20/day (public transit, taxis)
- Attraction tickets: $10-30 per major attraction

"""

        if constraints:
            prompt += "\nConstraints to satisfy:\n"
            if constraints.budget:
                prompt += f"- Budget: {constraints.currency} {constraints.budget}\n"
                prompt += f"  ⚠️  You MUST stay within this budget - calculate all costs carefully!\n"
            if constraints.days:
                prompt += f"- Duration: {constraints.days} days\n"
                prompt += f"  ⚠️  Create a detailed plan for each of these {constraints.days} days!\n"
            if constraints.origin_city:
                prompt += f"- Departure city: {constraints.origin_city}\n"
            if constraints.destination_city:
                prompt += f"- Destination: {constraints.destination_city}\n"
            if flag_local:
                prompt += f"  ⚠️  NOTE: Origin and destination are the same! This is a LOCAL trip - do NOT search for flights!\n"
            if constraints.start_date:
                prompt += f"- Start date: {constraints.start_date}\n"
            if constraints.preferences:
                prompt += f"- Preferences: {', '.join(constraints.preferences)}\n"

        # Add strategy hints
        if strategy:
            if "attractions_k" in strategy:
                prompt += f"\nConsider up to {strategy['attractions_k']} attractions when planning.\n"

        prompt += """
IMPORTANT RULES:
1. After gathering all tool results, you MUST provide a detailed summary
2. Your summary MUST include:
   - Specific attraction names and activities for each day
   - Realistic cost estimates for accommodation and meals
   - Total cost calculation that fits within the budget
3. Be specific and concrete - avoid generic statements
4. If budget is tight, suggest budget-friendly options"""

        return prompt

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling tool definitions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_flights",
                    "description": "Search for flights between two cities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {
                                "type": "string",
                                "description": "Departure city or airport code"
                            },
                            "destination": {
                                "type": "string",
                                "description": "Arrival city or airport code"
                            },
                            "date": {
                                "type": "string",
                                "description": "Departure date in YYYY-MM-DD format"
                            },
                            "max_price": {
                                "type": "number",
                                "description": "Maximum price filter (optional)"
                            },
                            "currency": {
                                "type": "string",
                                "description": "Currency code (USD, NZD, etc)",
                                "default": "USD"
                            }
                        },
                        "required": ["origin", "destination", "date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get weather forecast for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format"
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days to forecast",
                                "default": 7
                            }
                        },
                        "required": ["location", "start_date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_attractions",
                    "description": "Search for tourist attractions in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            },
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by categories (museum, beach, park, etc)"
                            },
                            "min_rating": {
                                "type": "number",
                                "description": "Minimum rating (0-5)",
                                "default": 4.0
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call"""
        try:
            if tool_name == "search_flights":
                return await self.tools.search_flights(**arguments)
            elif tool_name == "get_weather_forecast":
                return await self.tools.get_weather_forecast(**arguments)
            elif tool_name == "search_attractions":
                return await self.tools.search_attractions(**arguments)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_itinerary(
        self,
        messages: List[Dict],
        tool_calls: List[ToolCall],
        request: PlanRequest,
        target_currency: str,
    ) -> TripItinerary:
        """Extract itinerary from tool calls and conversation."""

        target_currency = target_currency.upper()

        flights: List[FlightInfo] = []
        weather_forecast: List[WeatherInfo] = []
        attractions: List[AttractionInfo] = []
        fx_rates: Dict[str, float] = {}
        conversion_warnings: List[str] = []

        destination_from_flights = None
        origin_from_flights = None

        for tool_call in tool_calls:
            if not tool_call.result:
                continue

            if tool_call.tool_name == "search_flights":
                flight_data = tool_call.result.get("flights", [])
                for flight in flight_data[:2]:
                    original_price = flight.get("price")
                    original_currency = (flight.get("currency") or target_currency).upper()
                    flight["original_price"] = original_price
                    flight["original_currency"] = original_currency

                    converted_price = self._convert_and_track(
                        original_price,
                        original_currency,
                        target_currency,
                        fx_rates,
                    )
                    if converted_price is not None:
                        flight["price"] = round(converted_price, 2)
                        flight["currency"] = target_currency
                    else:
                        flight["price"] = round(float(original_price or 0.0), 2)
                        flight["currency"] = original_currency
                        if original_currency != target_currency:
                            conversion_warnings.append(
                                f"Could not convert flight price {original_currency} {original_price} to {target_currency}"
                            )

                    flights.append(FlightInfo(**flight))

                if tool_call.arguments:
                    destination_from_flights = tool_call.arguments.get("destination", destination_from_flights)
                    origin_from_flights = tool_call.arguments.get("origin", origin_from_flights)

            elif tool_call.tool_name == "get_weather_forecast":
                forecast_data = tool_call.result.get("forecast", [])
                for day in forecast_data:
                    weather_forecast.append(
                        WeatherInfo(
                            date=day["date"],
                            temperature_celsius=day["temperature_celsius"],
                            condition=day["condition"],
                            precipitation_chance=day["precipitation_chance"],
                        )
                    )

            elif tool_call.tool_name == "search_attractions":
                attraction_data = tool_call.result.get("attractions", [])
                for attr in attraction_data:
                    attractions.append(
                        AttractionInfo(
                            name=attr["name"],
                            category=attr["category"],
                            rating=attr["rating"],
                            price_range=attr["price_range"],
                            description=attr.get("description", ""),
                        )
                    )

        # Sum flight costs in target currency
        flight_cost = 0.0
        for flight in flights:
            currency_code = (flight.currency or target_currency).upper()
            price_for_total = flight.price

            if currency_code != target_currency:
                converted = self._convert_and_track(price_for_total, currency_code, target_currency, fx_rates)
                if converted is None:
                    rate = fx_rates.get(currency_code) or self._get_fx_rate(currency_code, target_currency)
                    if rate:
                        fx_rates[currency_code] = rate
                        converted = price_for_total * rate
                    else:
                        conversion_warnings.append(
                            f"Using original price for flight ({currency_code} {price_for_total}) due to missing FX rate"
                        )
                        converted = price_for_total
                else:
                    flight.currency = target_currency
                    flight.price = round(converted, 2)
                price_for_total = converted
            else:
                flight.price = round(flight.price, 2)
                if flight.original_price and flight.original_currency:
                    self._convert_and_track(
                        flight.original_price,
                        flight.original_currency,
                        target_currency,
                        fx_rates,
                    )

            flight_cost += price_for_total

        # Determine trip duration
        num_days = request.constraints.days if request.constraints and request.constraints.days else 3
        nights = max(num_days - 1, 0)

        accommodation_cost = self._convert_and_track(
            BASE_ACCOM_NIGHT_USD * nights,
            "USD",
            target_currency,
            fx_rates,
        )
        if accommodation_cost is None:
            accommodation_cost = BASE_ACCOM_NIGHT_USD * nights
            conversion_warnings.append("Accommodation estimate uses USD baseline due to missing FX rate.")

        meal_cost = self._convert_and_track(
            BASE_MEALS_PER_DAY_USD * num_days,
            "USD",
            target_currency,
            fx_rates,
        )
        if meal_cost is None:
            meal_cost = BASE_MEALS_PER_DAY_USD * num_days
            conversion_warnings.append("Meal estimate uses USD baseline due to missing FX rate.")

        other_costs = self._convert_and_track(
            BASE_MISC_PER_DAY_USD * num_days,
            "USD",
            target_currency,
            fx_rates,
        )
        if other_costs is None:
            other_costs = BASE_MISC_PER_DAY_USD * num_days
            conversion_warnings.append("Local transport estimate uses USD baseline due to missing FX rate.")

        total_cost = flight_cost + accommodation_cost + meal_cost + other_costs

        # Extract destination from tool calls or prompt
        destination = (
            request.constraints.destination_city
            if request.constraints and request.constraints.destination_city
            else destination_from_flights or "Unknown"
        )

        if request.constraints and not request.constraints.destination_city and destination_from_flights:
            request.constraints.destination_city = destination_from_flights
        if request.constraints and not request.constraints.origin_city and origin_from_flights:
            request.constraints.origin_city = origin_from_flights

        # Daily plan
        daily_plan = []
        if request.constraints and request.constraints.days:
            attractions_per_day = max(2, len(attractions) // num_days) if num_days else len(attractions)
            for day in range(num_days):
                day_weather = weather_forecast[day] if day < len(weather_forecast) else None
                start_idx = day * attractions_per_day
                end_idx = start_idx + attractions_per_day
                day_attractions = attractions[start_idx:end_idx] if start_idx < len(attractions) else []

                day_plan = {
                    "day": day + 1,
                    "activities": day_attractions,
                    "weather": {
                        "temperature": day_weather.temperature_celsius if day_weather else "N/A",
                        "condition": day_weather.condition if day_weather else "N/A",
                    }
                    if day_weather
                    else None,
                }
                daily_plan.append(day_plan)

        total_cost_usd = None
        rate_target_to_usd = self._get_fx_rate(target_currency, "USD")
        if rate_target_to_usd:
            fx_rates[f"{target_currency}->USD"] = rate_target_to_usd
            total_cost_usd = round(total_cost * rate_target_to_usd, 2)
        else:
            converted = self._convert_currency(total_cost, target_currency, "USD")
            if converted is not None:
                total_cost_usd = round(converted, 2)

        total_cost_nzd = self._convert_currency(total_cost, target_currency, "NZD")
        if total_cost_nzd is not None:
            total_cost_nzd = round(total_cost_nzd, 2)

        cost_breakdown = {
            "flights": round(flight_cost, 2),
            "accommodation": round(accommodation_cost, 2),
            "meals": round(meal_cost, 2),
            "other": round(other_costs, 2),
        }

        total_cost = round(total_cost, 2)

        currency_note = None
        if conversion_warnings:
            currency_note = " ".join(conversion_warnings)

        return TripItinerary(
            destination=destination,
            flights=flights,
            weather_forecast=weather_forecast,
            attractions=attractions,
            total_cost=total_cost,
            currency=target_currency,
            daily_plan=daily_plan,
            total_cost_usd=total_cost_usd,
            total_cost_nzd=total_cost_nzd,
            cost_breakdown=cost_breakdown,
            fx_rates=fx_rates or None,
            currency_note=currency_note,
        )

    def _validate_constraints(
        self,
        itinerary: TripItinerary,
        constraints
    ) -> tuple[bool, List[str]]:
        """Validate if itinerary satisfies constraints"""
        violations = []

        if not constraints:
            return True, violations

        # Check budget constraint
        if constraints.budget and itinerary.total_cost > constraints.budget:
            violations.append(
                f"Budget exceeded: {itinerary.currency} {itinerary.total_cost} > {constraints.budget}"
            )

        # Check days constraint
        if constraints.days and len(itinerary.daily_plan) != constraints.days:
            violations.append(
                f"Duration mismatch: {len(itinerary.daily_plan)} days planned, {constraints.days} requested"
            )

        return len(violations) == 0, violations

    def _convert_currency(
        self,
        amount: Optional[float],
        source_currency: Optional[str],
        target_currency: str,
    ) -> Optional[float]:
        """Convert amount using live FX rates."""
        if amount is None or not source_currency:
            return None
        try:
            converted = self.fx.convert(float(amount), source_currency, target_currency)
            return None if converted is None else float(converted)
        except Exception as exc:
            logger.warning(
                "Currency conversion failed for %s -> %s: %s",
                source_currency,
                target_currency,
                exc,
            )
            return None

    def _convert_to_nzd(self, amount: Optional[float], currency: str) -> Optional[float]:
        """Convert amount to NZD using FX service."""
        return self._convert_currency(amount, currency, "NZD")

    def _calculate_satisfaction_score(
        self,
        itinerary: TripItinerary,
        constraints,
        constraints_satisfied: bool
    ) -> float:
        """
        Calculate constraint satisfaction score (0..1) for learning.

        Factors:
        - Budget satisfaction (50%)
        - Days match (30%)
        - Has flights (10%)
        - Has attractions (10%)
        """
        score = 0.0

        # Budget satisfaction (50%)
        if constraints and constraints.budget:
            if itinerary.total_cost <= constraints.budget:
                score += 0.5
            else:
                # Partial credit if close
                overage = (itinerary.total_cost - constraints.budget) / constraints.budget
                score += max(0.0, 0.5 * (1.0 - overage))

        # Days match (30%)
        if constraints and constraints.days:
            if len(itinerary.daily_plan) == constraints.days:
                score += 0.3
            else:
                # Partial credit if close
                diff = abs(len(itinerary.daily_plan) - constraints.days)
                score += max(0.0, 0.3 * (1.0 - diff / max(1, constraints.days)))

        # Has flights (10%)
        if itinerary.flights:
            score += 0.1

        # Has attractions (10%)
        if itinerary.attractions:
            score += 0.1

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Feasibility helpers
    # ------------------------------------------------------------------

    def _normalize_location(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return name.strip().lower()

    def _get_country_for_city(self, city: Optional[str]) -> Optional[str]:
        norm = self._normalize_location(city)
        if not norm:
            return None
        return CITY_COUNTRY.get(norm)

    def _get_country_label(self, name: Optional[str]) -> Optional[str]:
        norm = self._normalize_location(name)
        if not norm:
            return None
        if norm in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[norm]
        city_country = CITY_COUNTRY.get(norm)
        if city_country:
            return city_country
        return None

    def _is_local_trip(self, origin: Optional[str], destination: Optional[str]) -> bool:
        origin_norm = self._normalize_location(origin)
        dest_norm = self._normalize_location(destination)
        if not origin_norm or not dest_norm:
            return False
        if origin_norm == dest_norm:
            return True
        origin_country = self._get_country_label(origin)
        dest_country = self._get_country_label(destination)
        if origin_country and dest_country and origin_country == dest_country:
            return True
        if origin_country and dest_country:
            if origin_country == dest_norm:
                return True
            if dest_country == origin_norm:
                return True
        return False

    def is_local_trip(self, origin: Optional[str], destination: Optional[str]) -> bool:
        return self._is_local_trip(origin, destination)

    def _city_coords(self, city: Optional[str]) -> Optional[Tuple[float, float]]:
        if not city:
            return None
        return CITY_COORDINATES.get(city.lower().strip())

    def _haversine_km(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        from math import radians, sin, cos, sqrt, atan2

        lat1, lon1 = map(radians, a)
        lat2, lon2 = map(radians, b)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * 6371 * atan2(sqrt(h), sqrt(1 - h))

    def _min_flight_cost(self, distance_km: float, target_currency: str) -> float:
        base_usd = distance_km * MIN_FLIGHT_COST_PER_KM_USD
        base_usd = max(base_usd, 120.0)
        converted = self._convert_currency(base_usd, "USD", target_currency)
        return converted if converted is not None else base_usd

    def _minutes_needed(self, distance_km: float) -> float:
        flight_hours = distance_km / 800 if distance_km > 0 else 0
        return (flight_hours + 6) * 60  # add buffer hours for check-in/security/local travel

    def _evaluate_feasibility(self, request: PlanRequest) -> Dict[str, Any]:
        issues: List[str] = []
        recommendations: List[str] = []

        constraints = request.constraints
        if not constraints:
            return {"feasible": True, "issues": issues, "recommendations": recommendations}

        origin = constraints.origin_city
        destination = constraints.destination_city
        budget = constraints.budget
        currency = constraints.currency or "USD"
        days = constraints.days

        origin_coords = self._city_coords(origin)
        destination_coords = self._city_coords(destination)
        distance_km = None

        if origin_coords and destination_coords:
            distance_km = self._haversine_km(origin_coords, destination_coords)

        # Same-city check
        if origin and destination:
            origin_norm = origin.strip().lower()
            dest_norm = destination.strip().lower()
            if origin_norm == dest_norm:
                issues.append("Origin and destination are the same city; flights are unnecessary.")
                recommendations.append("Consider using local transport (bus, rideshare, metro) instead of flights.")

        if self._is_local_trip(origin, destination):
            recommendations.append("Origin and destination are within the same area; focus on local transport and activities rather than flights.")

        # Distance/time feasibility
        if distance_km is not None:
            if distance_km < 5:
                recommendations.append("Distance is extremely short; focus on local activities instead of travel.")
            elif distance_km < DISTANCE_FLIGHT_THRESHOLD_KM:
                recommendations.append("Distance is short; ground transport (train/bus) may be more practical than flights.")

            if days is not None and days <= 1:
                minutes_needed = self._minutes_needed(distance_km)
                if minutes_needed > 24 * 60:
                    issues.append(
                        f"Round trip travel requires at least {minutes_needed/60:.1f} hours which exceeds the available time."
                    )
                    recommendations.append("Extend the trip duration or consider a virtual alternative.")
            elif distance_km > 15000 and (days is None or days < 3):
                issues.append("Ultra long-haul travel generally requires more than two days to be practical.")
                recommendations.append("Increase the trip to at least 4-5 days for long-haul flights.")
        else:
            recommendations.append("Could not determine distance for these cities; results may be approximate.")

        # Budget feasibility (rough lower bound)
        if budget is not None:
            comparison_currency = currency
            if distance_km is not None and distance_km > 0:
                min_flight_cost = self._min_flight_cost(distance_km, comparison_currency)
                nights = max(days - 1, 0) if days else 1
                meal_days = days if days else 2

                accom_unit = self._convert_currency(BASE_ACCOM_NIGHT_USD, "USD", comparison_currency)
                meals_unit = self._convert_currency(BASE_MEALS_PER_DAY_USD, "USD", comparison_currency)
                misc_unit = self._convert_currency(BASE_MISC_PER_DAY_USD, "USD", comparison_currency)

                if accom_unit is None:
                    accom_unit = BASE_ACCOM_NIGHT_USD
                if meals_unit is None:
                    meals_unit = BASE_MEALS_PER_DAY_USD
                if misc_unit is None:
                    misc_unit = BASE_MISC_PER_DAY_USD

                min_trip_cost = (
                    min_flight_cost
                    + accom_unit * nights
                    + meals_unit * meal_days
                    + misc_unit * meal_days
                )

                if budget < min_trip_cost * 0.5:
                    issues.append(
                        f"Budget of {comparison_currency} {budget:.0f} appears far below an estimated minimum of {comparison_currency} {min_trip_cost:.0f}."
                    )
                    recommendations.append("Consider increasing the budget significantly or choosing a closer destination.")

        feasible = len(issues) == 0
        return {"feasible": feasible, "issues": issues, "recommendations": recommendations}


# Singleton instance
_agent_instance: Optional[PlanningAgent] = None


def get_planning_agent() -> PlanningAgent:
    """Get singleton instance of planning agent"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PlanningAgent()
    return _agent_instance
