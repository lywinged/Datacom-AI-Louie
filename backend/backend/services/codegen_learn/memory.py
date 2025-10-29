"""
Code Generation Experience Memory
Stores and retrieves historical code generation experiences
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json
import pathlib
import heapq

from .signature import CodeGenSignature, signature_sim
from .bandit import Bandit


@dataclass
class Experience:
    signature: Dict[str, Any]
    strategy: Dict[str, Any]
    outcome: Dict[str, Any]
    reward: float


class ExperienceMemory:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bandits = {}  # cluster_id -> Bandit

    def append(self, exp: Experience):
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(exp), ensure_ascii=False) + "\n")

    def _iter(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def nearest(self, sig: CodeGenSignature, k: int = 8) -> List[Dict[str, Any]]:
        """Find k nearest historical experiences, with failure pattern downweighting."""
        heap = []
        idx = 0
        for rec in self._iter() or []:
            try:
                ss = CodeGenSignature.from_dict(rec["signature"])
            except Exception:
                continue
            sim = signature_sim(sig, ss)

            # Downweight failed cases with similar failure patterns
            if rec.get("outcome", {}).get("success") == False:
                # If both have failure patterns and they're similar, heavily downweight
                sig_pattern = sig.failure_pattern or ""
                ss_pattern = ss.failure_pattern or ""

                if sig_pattern and ss_pattern:
                    # Check if failure patterns overlap (e.g. both have "syntax_error")
                    sig_errors = set(sig_pattern.split("+"))
                    ss_errors = set(ss_pattern.split("+"))
                    overlap = len(sig_errors & ss_errors)

                    if overlap > 0:
                        # Similar failure pattern - downweight by 70%
                        sim *= 0.3
                    else:
                        # Different failure type - downweight by 40%
                        sim *= 0.6
                else:
                    # Generic failure without pattern - downweight by 50%
                    sim *= 0.5

            # Add index as tiebreaker to avoid comparing dicts when sim values are equal
            heapq.heappush(heap, (sim, idx, rec))
            idx += 1
            if len(heap) > k:
                heapq.heappop(heap)
        out = sorted(heap, key=lambda x: -x[0])
        return [{"sim": s, **r} for s, i, r in out]

    def choose_strategy(self, sig: CodeGenSignature, candidate_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Choose strategy using Thompson Sampling + Nearest Neighbor Exploitation.
        Returns the selected strategy dict.
        """
        # Rank by: (1) best historical success for similar signatures using each strategy key, (2) Thompson sample
        arms = [json.dumps(cs, sort_keys=True) for cs in candidate_strategies]
        b = self.bandits.setdefault(self._cluster_key(sig), Bandit())

        # Exploit: if nearest has a dominant winner, pick it
        nearest = self.nearest(sig, k=10)
        if nearest:
            # score each candidate by weighted reward among neighbors
            scores = {}
            for cs in candidate_strategies:
                key = json.dumps(cs, sort_keys=True)
                weighted = 0.0
                wsum = 0.0
                for item in nearest:
                    sim = item["sim"]
                    if item.get("strategy") == cs:
                        r = float(item.get("reward", 0.0))
                        weighted += sim * r
                        wsum += sim
                scores[key] = (weighted / wsum) if wsum > 0 else None

            # If any candidate has clear advantage, pick it
            # Use more conservative threshold (0.65 instead of 0.55) for code generation
            best_key = None
            best_score = -1e9
            for k, v in scores.items():
                if v is not None and v > best_score:
                    best_key, best_score = k, v
            if best_key is not None and best_score > 0.65 and wsum > 2.0:
                return json.loads(best_key)

        # Otherwise Thompson sample
        chosen_key = b.choose(arms)
        return json.loads(chosen_key)

    def update_outcome(self, sig: CodeGenSignature, strategy: Dict[str, Any], success: bool, reward: float):
        exp = Experience(signature=sig.to_dict(), strategy=strategy, outcome={"success": success}, reward=reward)
        self.append(exp)
        b = self.bandits.setdefault(self._cluster_key(sig), Bandit())
        b.update(json.dumps(strategy, sort_keys=True), success=success, reward=reward)

    def _cluster_key(self, sig: CodeGenSignature) -> str:
        """
        Cluster key by language + task_type + complexity.
        This groups similar code generation tasks together.
        """
        lang = sig.language.lower()
        task_type = sig.task_type.lower()
        complexity = sig.complexity.lower()
        return f"{lang}|{task_type}|{complexity}"
