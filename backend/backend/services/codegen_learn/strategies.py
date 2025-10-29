"""
Code Generation Strategy Definitions
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass(frozen=True)
class CodeGenStrategy:
    """Code generation strategy definition"""
    name: str                          # Strategy identifier

    # Planning phase
    initial_planning: str              # "detailed" | "minimal" | "none"
    plan_temperature: float            # Temperature for planning LLM call (0.1-0.5)

    # Generation phase
    generation_temperature: float      # Temperature for code generation (0.1-0.5)
    max_tokens_gen: int               # Max tokens for initial generation (1000-4000)
    include_type_hints: bool          # Force type hints/annotations

    # Testing phase
    test_first: bool                  # Generate tests before or after main code
    test_coverage_target: float       # Target test coverage (0.5-1.0)

    # Retry/Fix phase
    max_analysis_depth: int           # How detailed failure analysis should be (1-3)
    fix_temperature: float            # Temperature for fix generation (0.1-0.4)
    incremental_fix: bool             # Apply fixes incrementally or all at once

    # Robustness
    validation_rounds: int            # Pre-test validation rounds (0-2)
    fallback_to_simple: bool          # Fall back to simpler approach on repeated failures

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CODE_STRATEGIES: List[Dict[str, Any]] = [
    # Fast and pragmatic (low planning, fast generation)
    CodeGenStrategy(
        name="fast_pragmatic",
        initial_planning="minimal",
        plan_temperature=0.15,
        generation_temperature=0.25,
        max_tokens_gen=1500,
        include_type_hints=False,
        test_first=False,
        test_coverage_target=0.6,
        max_analysis_depth=1,
        fix_temperature=0.2,
        incremental_fix=True,
        validation_rounds=0,
        fallback_to_simple=True
    ).to_dict(),

    # Detailed planning (thorough upfront planning)
    CodeGenStrategy(
        name="detailed_planner",
        initial_planning="detailed",
        plan_temperature=0.2,
        generation_temperature=0.2,
        max_tokens_gen=2500,
        include_type_hints=True,
        test_first=False,
        test_coverage_target=0.8,
        max_analysis_depth=2,
        fix_temperature=0.2,
        incremental_fix=False,
        validation_rounds=1,
        fallback_to_simple=False
    ).to_dict(),

    # Test-driven (generate tests first)
    CodeGenStrategy(
        name="test_driven",
        initial_planning="detailed",
        plan_temperature=0.15,
        generation_temperature=0.3,
        max_tokens_gen=2000,
        include_type_hints=True,
        test_first=True,
        test_coverage_target=0.9,
        max_analysis_depth=2,
        fix_temperature=0.25,
        incremental_fix=True,
        validation_rounds=1,
        fallback_to_simple=False
    ).to_dict(),

    # Iterative refiner (focus on fixing)
    CodeGenStrategy(
        name="iterative_refiner",
        initial_planning="minimal",
        plan_temperature=0.25,
        generation_temperature=0.35,
        max_tokens_gen=1800,
        include_type_hints=False,
        test_first=False,
        test_coverage_target=0.7,
        max_analysis_depth=3,
        fix_temperature=0.15,
        incremental_fix=True,
        validation_rounds=0,
        fallback_to_simple=True
    ).to_dict(),

    # High quality (comprehensive approach)
    CodeGenStrategy(
        name="high_quality",
        initial_planning="detailed",
        plan_temperature=0.2,
        generation_temperature=0.2,
        max_tokens_gen=3000,
        include_type_hints=True,
        test_first=True,
        test_coverage_target=0.95,
        max_analysis_depth=3,
        fix_temperature=0.15,
        incremental_fix=False,
        validation_rounds=2,
        fallback_to_simple=False
    ).to_dict(),

    # Balanced (good default)
    CodeGenStrategy(
        name="balanced",
        initial_planning="minimal",
        plan_temperature=0.2,
        generation_temperature=0.3,
        max_tokens_gen=2000,
        include_type_hints=True,
        test_first=False,
        test_coverage_target=0.75,
        max_analysis_depth=2,
        fix_temperature=0.2,
        incremental_fix=True,
        validation_rounds=1,
        fallback_to_simple=True
    ).to_dict(),
]
