"""
Code Generation Learning Adapter
Integrates signature, memory, and strategies
"""
from __future__ import annotations
from typing import Dict, Any, Optional

from .signature import CodeGenSignature, build_signature
from .memory import ExperienceMemory
from .strategies import DEFAULT_CODE_STRATEGIES


class CodeGenAdapter:
    """Adapter for code generation learning system"""

    def __init__(self, memory_path: str):
        self.mem = ExperienceMemory(memory_path)

    def choose(self, language: str, task: str, test_framework: Optional[str] = None) -> Dict[str, Any]:
        """
        Choose strategy for code generation task.

        Args:
            language: Programming language
            task: Task description
            test_framework: Test framework (optional)

        Returns:
            Dict with "signature" and "strategy" keys
        """
        sig = build_signature(language, task, test_framework)
        strat = self.mem.choose_strategy(sig, DEFAULT_CODE_STRATEGIES)
        return {"signature": sig, "strategy": strat}

    def record(
        self,
        signature: CodeGenSignature,
        strategy: Dict[str, Any],
        test_passed: bool,
        retry_count: int,
        max_retries: int,
        generation_time_ms: float,
        code_length: int,
        estimated_loc: int,
        token_cost_usd: float,
    ) -> Dict[str, Any]:
        """
        Record code generation outcome with multidimensional reward calculation.

        Args:
            signature: Task signature
            strategy: Strategy used
            test_passed: Whether tests passed
            retry_count: Number of retries needed
            max_retries: Maximum retries allowed
            generation_time_ms: Generation time in milliseconds
            code_length: Lines of code generated
            estimated_loc: Estimated LOC for task
            token_cost_usd: Token cost in USD

        Returns:
            Dict with success, reward, strategy name, and breakdown
        """
        # MULTIDIMENSIONAL REWARD CALCULATION

        # 1. Success Score (50%): Did it pass tests?
        if test_passed:
            success_score = 1.0
        else:
            # Partial credit if we got close (fewer retries = closer)
            success_score = max(0.0, 1.0 - (retry_count / max(max_retries, 1)))

        # 2. Efficiency Score (25%): How many retries did it take?
        if retry_count == 0:
            efficiency_score = 1.0  # First try success!
        else:
            # Exponential penalty: 1 retry = 0.7, 2 retries = 0.5, 3+ retries = 0.3
            efficiency_score = max(0.3, 1.0 - (retry_count * 0.3))

        # 3. Quality Score (15%): Code conciseness (not too verbose)
        # Penalize if code is >2x estimated LOC (likely over-engineered)
        if estimated_loc > 0 and code_length > 0:
            loc_ratio = code_length / estimated_loc
            if loc_ratio <= 1.5:
                quality_score = 1.0
            elif loc_ratio <= 2.5:
                quality_score = 0.7
            else:
                quality_score = 0.4
        else:
            quality_score = 0.8

        # 4. Speed Score (10%): Did it generate quickly?
        # Target: ~100ms per estimated LOC
        if estimated_loc > 0:
            target_time_ms = estimated_loc * 100
            time_ratio = generation_time_ms / max(target_time_ms, 1000)
            if time_ratio <= 1.0:
                speed_score = 1.0
            elif time_ratio <= 2.0:
                speed_score = 0.7
            else:
                speed_score = 0.4
        else:
            speed_score = 0.7

        # Final weighted reward
        reward = (
            0.50 * success_score +
            0.25 * efficiency_score +
            0.15 * quality_score +
            0.10 * speed_score
        )
        reward = max(0.0, min(1.0, reward))

        # Success determination: binary threshold at 0.6
        success = reward >= 0.6

        # Update memory
        self.mem.update_outcome(signature, strategy, success=success, reward=reward)

        return {
            "success": success,
            "reward": reward,
            "strategy": strategy.get("name", "unknown"),
            "breakdown": {
                "success": success_score,
                "efficiency": efficiency_score,
                "quality": quality_score,
                "speed": speed_score
            }
        }
