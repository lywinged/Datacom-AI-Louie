
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import math, re, json, hashlib, time

_WORD_RE = re.compile(r"[A-Za-z\u4e00-\u9fa5]+")

def _tokenize(s: str) -> List[str]:
    if not s: return []
    tokens = _WORD_RE.findall(s.lower())
    return tokens

def _tf(tokens: List[str]) -> Dict[str, float]:
    d = {}
    for t in tokens:
        d[t] = d.get(t, 0.0) + 1.0
    n = float(len(tokens) or 1.0)
    for k in list(d.keys()):
        d[k] /= n
    return d

def cosine(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    num = 0.0
    for k,v in a.items():
        if k in b: num += v*b[k]
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na==0 or nb==0: return 0.0
    return max(0.0, min(1.0, num/(na*nb)))

def jaccard(a: List[str], b: List[str]) -> float:
    A = set(a); B=set(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A&B)/len(A|B)

@dataclass
class Signature:
    # Task & user constraints
    prompt: str
    city: str|None = None
    days: int|None = None
    budget_nzd: float|None = None
    date_range: str|None = None

    # Tooling & outcomes (for training)
    tools: List[str]|None = None
    errors: List[str]|None = None
    last_failure_digest: str|None = None

    # Failure pattern learning (NEW)
    failure_pattern: str|None = None  # e.g. "flights_timeout_3x", "weather_rate_limit", "attractions_no_results"
    failure_context: Dict[str,Any]|None = None  # Additional failure metadata

    # Derived fields (tokens)
    _prompt_tokens: List[str]|None = None

    def to_dict(self) -> Dict[str,Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: Dict[str,Any]) -> "Signature":
        return Signature(**{k:d.get(k) for k in [
            "prompt","city","days","budget_nzd","date_range",
            "tools","errors","last_failure_digest","_prompt_tokens",
            "failure_pattern","failure_context"
        ]})

def build_signature(prompt: str, constraints: Dict[str,Any], tool_names: List[str], error_digest: str|None) -> Signature:
    city = constraints.get("city") or None
    days = constraints.get("days") or None
    budget = constraints.get("budget_nzd") or constraints.get("budget") or None
    date_range = constraints.get("date_range") or None
    toks = _tokenize(prompt + " " + (error_digest or ""))
    return Signature(
        prompt=prompt, city=city, days=days, budget_nzd=budget, date_range=date_range,
        tools=tool_names or [], errors=[], last_failure_digest=error_digest,
        _prompt_tokens=toks
    )

def signature_sim(a: Signature, b: Signature) -> float:
    # hybrid: 0.6 * TF-cosine over prompt+error tokens + 0.1*tool overlap + 0.2*city/days match + 0.1*budget closeness
    ca = _tf(a._prompt_tokens or []); cb = _tf(b._prompt_tokens or [])
    cos = cosine(ca, cb)

    tool_overlap = 0.0
    if a.tools and b.tools:
        tool_overlap = len(set(a.tools) & set(b.tools)) / max(1,len(set(a.tools) | set(b.tools)))

    city_days = 0.0
    if a.city and b.city and a.city.lower()==b.city.lower():
        city_days += 0.6
    if a.days and b.days and a.days==b.days:
        city_days += 0.4

    budget_close = 0.0
    if a.budget_nzd and b.budget_nzd:
        diff = abs(a.budget_nzd - b.budget_nzd)
        budget_close = max(0.0, 1.0 - min(diff/500.0, 1.0))  # within $500 seen as similar

    return 0.6*cos + 0.1*tool_overlap + 0.2*city_days + 0.1*budget_close
