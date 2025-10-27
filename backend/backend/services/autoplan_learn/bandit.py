
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import random, math

@dataclass
class ArmStats:
    a: float = 1.0   # successes + 1 (Beta prior)
    b: float = 1.0   # failures + 1
    avg_reward: float = 0.0
    n: int = 0

class Bandit:
    def __init__(self):
        self._arms: Dict[str, ArmStats] = {}

    def sample(self, arm: str) -> float:
        st = self._arms.get(arm)
        if not st:
            return random.betavariate(1.0,1.0)
        return random.betavariate(st.a, st.b)

    def choose(self, arms: List[str]) -> str:
        scores = {a: self.sample(a) for a in arms}
        return max(scores, key=scores.get)

    def update(self, arm: str, success: bool, reward: float|None=None):
        st = self._arms.setdefault(arm, ArmStats())
        if success: st.a += 1.0
        else: st.b += 1.0
        if reward is not None:
            st.n += 1
            st.avg_reward += (reward - st.avg_reward)/max(1,st.n)

    def stats(self) -> Dict[str,ArmStats]:
        return self._arms
