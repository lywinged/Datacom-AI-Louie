
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from .signature import build_signature, Signature
from .memory import ExperienceMemory
from .strategies import DEFAULT_CANDIDATES

class AutoPlanAdapter:
    def __init__(self, memory_path: str):
        self.mem = ExperienceMemory(memory_path)

    def choose(self, prompt: str, constraints: Dict[str,Any], tools_used: List[str], error_digest: Optional[str]=None) -> Dict[str,Any]:
        sig = build_signature(prompt, constraints, tools_used, error_digest)
        strat = self.mem.choose_strategy(sig, DEFAULT_CANDIDATES)
        return {"signature": sig, "strategy": strat}

    def _extract_failure_pattern(self, tool_calls: List[Dict[str,Any]]) -> Optional[str]:
        """Extract failure pattern from tool calls for learning.

        Returns:
            Pattern string like "flights_timeout_3x", "weather_rate_limit", "attractions_no_results"
        """
        if not tool_calls:
            return None

        error_counts = {}
        error_types = []

        for tc in tool_calls:
            tool_name = tc.get("tool_name", "")
            result = tc.get("result", {})
            error = result.get("error") if isinstance(result, dict) else None

            if error:
                # Classify error type
                error_str = str(error).lower()
                if "timeout" in error_str or "timed out" in error_str:
                    error_type = f"{tool_name}_timeout"
                elif "rate limit" in error_str or "too many requests" in error_str:
                    error_type = f"{tool_name}_rate_limit"
                elif "no results" in error_str or "not found" in error_str or "empty" in error_str:
                    error_type = f"{tool_name}_no_results"
                elif "auth" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
                    error_type = f"{tool_name}_auth_error"
                elif "connection" in error_str or "network" in error_str:
                    error_type = f"{tool_name}_connection_error"
                else:
                    error_type = f"{tool_name}_error"

                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                error_types.append(error_type)

        if not error_counts:
            return None

        # Build pattern string
        # If same error happened multiple times, annotate with count
        patterns = []
        for err_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            if count >= 3:
                patterns.append(f"{err_type}_{count}x")
            elif count >= 2:
                patterns.append(f"{err_type}_2x")
            else:
                patterns.append(err_type)

        return "+".join(patterns[:3])  # Max 3 patterns to keep it concise

    def record(self, signature: Signature, strategy: Dict[str,Any], plan_cost: float, budget_nzd: float|None,
               tool_errors: int, satisfaction_proxy: float, tool_calls: Optional[List[Dict[str,Any]]]=None) -> Dict[str,Any]:
        """Record planning outcome with enhanced multidimensional reward calculation."""

        # MULTIDIMENSIONAL REWARD CALCULATION

        # 1. Budget Score (40%): How well did we meet the budget constraint?
        budget_score = 1.0  # Default if no budget specified
        if budget_nzd is not None and budget_nzd > 0 and plan_cost is not None:
            if plan_cost <= budget_nzd:
                # Under budget - perfect score
                budget_score = 1.0
            else:
                # Over budget - smooth penalty based on overspend ratio
                overspend_ratio = (plan_cost - budget_nzd) / budget_nzd
                # Exponential decay: 10% over = 0.9, 20% over = 0.8, 50% over = 0.5, etc.
                budget_score = max(0.0, 1.0 - overspend_ratio)

        # 2. Quality Score (30%): satisfaction_proxy measures POI quality/diversity
        quality_score = max(0.0, min(1.0, satisfaction_proxy))

        # 3. Reliability Score (30%): How reliable were the tool calls?
        reliability_score = 1.0
        if tool_calls:
            total_calls = len(tool_calls)
            if total_calls > 0:
                success_rate = (total_calls - tool_errors) / total_calls
                # Penalize multiple retries even if eventually successful
                retry_penalty = 0.0
                for tc in tool_calls:
                    if tc.get("retry_count", 0) > 0:
                        retry_penalty += 0.05  # 5% penalty per retried call

                reliability_score = max(0.0, success_rate - retry_penalty)

        # Final weighted reward
        reward = (
            0.4 * budget_score +
            0.3 * quality_score +
            0.3 * reliability_score
        )
        reward = max(0.0, min(1.0, reward))

        # Success determination: binary threshold at 0.6
        success = reward >= 0.6

        # Extract failure pattern if there were errors
        if tool_errors > 0 and tool_calls:
            failure_pattern = self._extract_failure_pattern(tool_calls)
            if failure_pattern:
                signature.failure_pattern = failure_pattern
                signature.failure_context = {
                    "tool_error_count": tool_errors,
                    "total_tool_calls": len(tool_calls),
                    "reward_breakdown": {
                        "budget": budget_score,
                        "quality": quality_score,
                        "reliability": reliability_score
                    }
                }

        self.mem.update_outcome(signature, strategy, success=success, reward=reward)
        return {
            "success": success,
            "reward": reward,
            "strategy": strategy.get("name", "unknown") if isinstance(strategy, dict) else str(strategy),
            "breakdown": {
                "budget": budget_score,
                "quality": quality_score,
                "reliability": reliability_score
            }
        }
