"""
Persistent metrics store for the planning agent.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


def _default_metrics() -> Dict[str, Any]:
    return {
        "totals": {
            "total_plans": 0,
            "success": 0,
            "partial": 0,
            "failure": 0,
            "total_token_cost_usd": 0.0,
            "total_planning_time_ms": 0.0,
            "total_tool_calls": 0,
            "total_tool_errors": 0,
        },
        "history": [],
    }


@dataclass
class HistoryEntry:
    """Single planning run record."""

    timestamp: str
    outcome: str
    planning_time_ms: float
    tool_calls: int
    tool_errors: int
    token_cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        cost_value = self.token_cost_usd or 0.0
        data["token_cost_usd"] = cost_value
        # Backward compatibility for any consumers expecting cost_usd
        data["cost_usd"] = cost_value
        return data


class PlanningMetricsStore:
    """Thread-safe metrics store persisted to JSON."""

    def __init__(
        self,
        path: Optional[str] = None,
        history_limit: int = 200,
    ):
        self.path = path or os.getenv("PLANNING_METRICS_PATH", "./data/planning_metrics.json")
        self.history_limit = history_limit
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = _default_metrics()
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load persisted metrics from disk."""
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load planning metrics (%s): %s", self.path, exc)
            return

        totals = raw.get("totals") or {}
        history = raw.get("history") or []

        # Merge with defaults to avoid missing keys
        merged_totals = _default_metrics()["totals"]
        merged_totals.update({k: totals.get(k, v) for k, v in merged_totals.items()})
        # Backward compatibility: migrate legacy total_cost_usd field
        legacy_total_cost = totals.get("total_cost_usd")
        if legacy_total_cost is not None and "total_token_cost_usd" not in totals:
            merged_totals["total_token_cost_usd"] = legacy_total_cost

        with self._lock:
            self._data["totals"] = merged_totals
            # Only keep valid dict entries
            history_entries = [entry for entry in history if isinstance(entry, dict)]
            for entry in history_entries:
                if "token_cost_usd" not in entry:
                    entry["token_cost_usd"] = entry.get("cost_usd", 0.0)
            self._data["history"] = history_entries

    def _save(self) -> None:
        """Persist metrics to disk."""
        directory = os.path.dirname(self.path) or "."
        try:
            os.makedirs(directory, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            # Do not raise – metrics should never break primary flow
            logger.warning("Failed to persist planning metrics (%s): %s", self.path, exc)

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_plan(
        self,
        *,
        constraints_satisfied: bool,
        planning_time_ms: float,
        tool_calls: int,
        tool_errors: int,
        token_cost_usd: Optional[float],
    ) -> None:
        outcome = "success" if constraints_satisfied else "partial"
        entry = HistoryEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            outcome=outcome,
            planning_time_ms=planning_time_ms,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            token_cost_usd=token_cost_usd,
        )

        with self._lock:
            totals = self._data["totals"]
            totals["total_plans"] += 1
            totals[outcome] += 1
            totals["total_planning_time_ms"] += planning_time_ms
            totals["total_tool_calls"] += tool_calls
            totals["total_tool_errors"] += tool_errors
            if token_cost_usd:
                totals["total_token_cost_usd"] += token_cost_usd

            history: List[Dict[str, Any]] = self._data["history"]
            history.append(entry.to_dict())
            if len(history) > self.history_limit:
                self._data["history"] = history[-self.history_limit:]

        self._save()

    def record_failure(
        self,
        *,
        planning_time_ms: float,
        tool_calls: int,
        error: Optional[str] = None,
    ) -> None:
        entry = HistoryEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            outcome="failure",
            planning_time_ms=planning_time_ms,
            tool_calls=tool_calls,
            tool_errors=0,
            token_cost_usd=0.0,
        ).to_dict()
        if error:
            entry["error"] = error

        with self._lock:
            totals = self._data["totals"]
            totals["total_plans"] += 1
            totals["failure"] += 1
            totals["total_planning_time_ms"] += planning_time_ms
            totals["total_tool_calls"] += tool_calls

            history: List[Dict[str, Any]] = self._data["history"]
            history.append(entry)
            if len(history) > self.history_limit:
                self._data["history"] = history[-self.history_limit:]

        self._save()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            totals = dict(self._data["totals"])
            history = list(self._data["history"])

        total_plans = totals["total_plans"]
        success = totals["success"]
        partial = totals["partial"]
        failure = totals["failure"]
        total_cost = totals.get("total_token_cost_usd", 0.0)
        if not total_cost:
            total_cost = totals.get("total_cost_usd", 0.0)  # legacy support
        total_planning_time = totals["total_planning_time_ms"]
        total_tool_calls = totals["total_tool_calls"]
        total_tool_errors = totals["total_tool_errors"]

        avg_planning_time = (
            total_planning_time / total_plans if total_plans > 0 else 0.0
        )
        avg_tool_calls = (
            total_tool_calls / total_plans if total_plans > 0 else 0.0
        )

        success_rate_pct = (success / total_plans * 100.0) if total_plans > 0 else 0.0
        constraint_rate = success / total_plans if total_plans > 0 else 0.0
        tool_success_rate = (
            (total_tool_calls - total_tool_errors) / total_tool_calls
            if total_tool_calls > 0
            else 1.0
        )

        return {
            "total_plans": total_plans,
            "success_plans": success,
            "partial_plans": partial,
            "failed_plans": failure,
            "success_rate": success_rate_pct,
            "avg_planning_time_ms": avg_planning_time,
            "avg_tool_calls_per_plan": avg_tool_calls,
            "constraint_satisfaction_rate": constraint_rate,
            "tool_success_rate": tool_success_rate,
            "total_cost_usd": total_cost,
            "history": history,
        }

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._data["history"][-limit:])

    def reset(self) -> None:
        """Reset all metrics and remove persisted file."""
        with self._lock:
            self._data = _default_metrics()
        # Best effort: remove persisted file so future loads start clean
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception as exc:
            logger.warning("Failed to remove planning metrics file (%s): %s", self.path, exc)

# Shared singleton used across the application
planning_metrics_store = PlanningMetricsStore()
