"""
Code Generation Task Signature
Identifies similar code generation tasks
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class CodeGenSignature:
    """Signature identifying a code generation task type"""
    language: str                      # "python", "rust", "bash", etc.
    task_type: str                     # "algorithm", "data_processing", "api", "cli", etc.
    complexity: str                    # "simple", "medium", "complex"
    test_framework: str                # "pytest", "unittest", "cargo_test", etc.
    has_external_deps: bool            # Needs pip/cargo dependencies?
    estimated_loc: int                 # Estimated lines of code (10-100+ bins)
    failure_pattern: Optional[str] = None  # e.g., "syntax_error", "logic_error", "timeout"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CodeGenSignature:
        return cls(**d)


def build_signature(language: str, task: str, test_framework: Optional[str] = None) -> CodeGenSignature:
    """Build signature from code request"""

    # Classify task type by keywords
    task_lower = task.lower()
    if any(kw in task_lower for kw in ["sort", "search", "algorithm", "fibonacci", "prime", "tree", "graph"]):
        task_type = "algorithm"
    elif any(kw in task_lower for kw in ["parse", "process", "csv", "json", "data", "transform"]):
        task_type = "data_processing"
    elif any(kw in task_lower for kw in ["api", "request", "http", "rest", "endpoint"]):
        task_type = "api"
    elif any(kw in task_lower for kw in ["cli", "command", "arg", "stdin", "argument"]):
        task_type = "cli"
    elif any(kw in task_lower for kw in ["class", "object", "inherit", "interface"]):
        task_type = "oop"
    else:
        task_type = "general"

    # Estimate complexity by task length and keywords
    complexity_indicators = sum([
        len(task.split()) > 50,  # Long description
        "complex" in task_lower or "advanced" in task_lower,
        "multiple" in task_lower or "several" in task_lower,
        "optimize" in task_lower or "efficient" in task_lower,
        "recursive" in task_lower or "dynamic programming" in task_lower,
    ])

    if complexity_indicators >= 3:
        complexity = "complex"
    elif complexity_indicators >= 1:
        complexity = "medium"
    else:
        complexity = "simple"

    # Estimate LOC
    if complexity == "simple":
        estimated_loc = 30
    elif complexity == "medium":
        estimated_loc = 80
    else:
        estimated_loc = 150

    # Check for external dependencies
    has_external_deps = any(kw in task_lower for kw in [
        "requests", "numpy", "pandas", "import", "library", "package", "dependency"
    ])

    return CodeGenSignature(
        language=language,
        task_type=task_type,
        complexity=complexity,
        test_framework=test_framework or "auto",
        has_external_deps=has_external_deps,
        estimated_loc=estimated_loc,
        failure_pattern=None
    )


def signature_sim(a: CodeGenSignature, b: CodeGenSignature) -> float:
    """
    Calculate similarity between two code generation signatures.
    Returns a score between 0.0 and 1.0
    """
    score = 0.0
    weights_sum = 0.0

    # Language match (weight: 0.3)
    if a.language == b.language:
        score += 0.3
    weights_sum += 0.3

    # Task type match (weight: 0.25)
    if a.task_type == b.task_type:
        score += 0.25
    elif a.task_type in ["algorithm", "data_processing"] and b.task_type in ["algorithm", "data_processing"]:
        score += 0.15  # Partial match for related types
    weights_sum += 0.25

    # Complexity match (weight: 0.2)
    complexity_map = {"simple": 0, "medium": 1, "complex": 2}
    a_comp = complexity_map.get(a.complexity, 1)
    b_comp = complexity_map.get(b.complexity, 1)
    comp_diff = abs(a_comp - b_comp)
    if comp_diff == 0:
        score += 0.2
    elif comp_diff == 1:
        score += 0.1  # Adjacent complexity levels
    weights_sum += 0.2

    # Test framework match (weight: 0.1)
    if a.test_framework == b.test_framework:
        score += 0.1
    weights_sum += 0.1

    # External deps match (weight: 0.1)
    if a.has_external_deps == b.has_external_deps:
        score += 0.1
    weights_sum += 0.1

    # LOC similarity (weight: 0.05)
    if a.estimated_loc > 0 and b.estimated_loc > 0:
        loc_ratio = min(a.estimated_loc, b.estimated_loc) / max(a.estimated_loc, b.estimated_loc)
        score += 0.05 * loc_ratio
    weights_sum += 0.05

    # Normalize
    return score / weights_sum if weights_sum > 0 else 0.0
