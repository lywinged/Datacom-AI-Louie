
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

@dataclass(frozen=True)
class Strategy:
    tool_order: str          # e.g., "flights->weather->attractions" or "weather->attractions->flights"
    attractions_k: int       # how many POIs to consider before pruning
    price_weight: float      # weight for budget fit
    rating_weight: float     # weight for quality
    diversity_weight: float  # encourage diverse POI categories
    model_temp: float        # LLM creativity when drafting itinerary
    max_tool_retries: int    # robustness to flaky APIs
    skip_tools: List[str]|None = None  # Tools to skip for this strategy (e.g., ["search_flights"] for local trips)

    def to_dict(self) -> Dict[str,Any]:
        return asdict(self)

DEFAULT_CANDIDATES: List[Dict[str,Any]] = [
    # Original 4 strategies (balanced)
    Strategy("weather->attractions->flights", attractions_k=20, price_weight=0.5, rating_weight=0.3, diversity_weight=0.2, model_temp=0.3, max_tool_retries=1).to_dict(),
    Strategy("flights->weather->attractions", attractions_k=15, price_weight=0.6, rating_weight=0.25, diversity_weight=0.15, model_temp=0.2, max_tool_retries=2).to_dict(),
    Strategy("weather->flights->attractions", attractions_k=30, price_weight=0.4, rating_weight=0.4, diversity_weight=0.2, model_temp=0.35, max_tool_retries=2).to_dict(),
    Strategy("attractions->weather->flights", attractions_k=25, price_weight=0.45, rating_weight=0.35, diversity_weight=0.2, model_temp=0.25, max_tool_retries=1).to_dict(),

    # Budget-focused strategies (price_weight dominant)
    Strategy("attractions->flights->weather", attractions_k=10, price_weight=0.75, rating_weight=0.15, diversity_weight=0.1, model_temp=0.15, max_tool_retries=1).to_dict(),
    Strategy("flights->attractions->weather", attractions_k=12, price_weight=0.7, rating_weight=0.2, diversity_weight=0.1, model_temp=0.2, max_tool_retries=1).to_dict(),

    # Quality-focused strategies (rating_weight dominant)
    Strategy("weather->attractions->flights", attractions_k=40, price_weight=0.2, rating_weight=0.6, diversity_weight=0.2, model_temp=0.4, max_tool_retries=3).to_dict(),
    Strategy("attractions->weather->flights", attractions_k=35, price_weight=0.25, rating_weight=0.55, diversity_weight=0.2, model_temp=0.35, max_tool_retries=2).to_dict(),

    # Fast planning strategies (low k, low temp, fewer retries)
    Strategy("flights->weather->attractions", attractions_k=8, price_weight=0.5, rating_weight=0.3, diversity_weight=0.2, model_temp=0.1, max_tool_retries=1).to_dict(),
    Strategy("weather->flights->attractions", attractions_k=10, price_weight=0.5, rating_weight=0.35, diversity_weight=0.15, model_temp=0.15, max_tool_retries=1).to_dict(),

    # Exploration strategies (high diversity, high temp, large k)
    Strategy("attractions->flights->weather", attractions_k=50, price_weight=0.3, rating_weight=0.3, diversity_weight=0.4, model_temp=0.6, max_tool_retries=2).to_dict(),
    Strategy("weather->attractions->flights", attractions_k=45, price_weight=0.35, rating_weight=0.25, diversity_weight=0.4, model_temp=0.5, max_tool_retries=2).to_dict(),
]
