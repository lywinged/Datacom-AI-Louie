#!/usr/bin/env python3
"""
Enhanced Learning Test System with Diverse Scenarios and Real-time Dashboard
"""

from __future__ import annotations
import json
import random
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

# Test scenario data class
@dataclass
class TripScenario:
    name: str
    category: str
    origin: str
    destination: str
    days: int
    budget_nzd: int
    preferences: List[str]
    expected_difficulty: str
    learning_focus: str

# Cities data
NZ_CITIES = {
    "major": ["Auckland", "Wellington", "Christchurch", "Queenstown"],
    "secondary": ["Dunedin", "Hamilton", "Tauranga", "Napier", "Nelson", "Rotorua"],
}

INTERNATIONAL_CITIES = {
    "short_haul": ["Sydney", "Melbourne", "Brisbane"],
    "medium_haul": ["Singapore", "Bangkok", "Tokyo"],
    "long_haul": ["London", "Paris", "New York", "Los Angeles"],
}

def generate_diverse_scenarios(num_scenarios: int = 50) -> List[TripScenario]:
    """Generate diverse test scenarios"""
    scenarios = []
    
    # Budget optimization scenarios (20%)
    for i in range(num_scenarios // 5):
        budget_levels = [
            ("budget_extreme", 400, 600, "extreme"),
            ("budget_low", 600, 1000, "hard"),
            ("budget_medium", 1000, 2000, "medium"),
            ("budget_high", 2000, 3500, "easy"),
        ]
        name, min_b, max_b, diff = random.choice(budget_levels)
        dest = random.choice(NZ_CITIES["major"] + NZ_CITIES["secondary"])
        orig = random.choice([c for c in NZ_CITIES["major"] if c != dest])
        
        scenarios.append(TripScenario(
            name=f"{name}_{dest}_{i}",
            category="budget_optimization",
            origin=orig,
            destination=dest,
            days=random.choice([2, 3, 4]),
            budget_nzd=random.randint(min_b, max_b),
            preferences=["budget-friendly"],
            expected_difficulty=diff,
            learning_focus="Budget optimization under constraints"
        ))
    
    # Duration variety (20%)
    for i in range(num_scenarios // 5):
        days_options = [2, 3, 4, 5, 7, 10]
        days = random.choice(days_options)
        diff = "easy" if days <= 3 else "medium" if days <= 5 else "hard"
        
        scenarios.append(TripScenario(
            name=f"duration_{days}days_{i}",
            category="duration_handling",
            origin="Auckland",
            destination=random.choice(NZ_CITIES["major"]),
            days=days,
            budget_nzd=random.randint(1000, 3000),
            preferences=["sightseeing"],
            expected_difficulty=diff,
            learning_focus=f"Optimize {days}-day itinerary planning"
        ))
    
    # Local vs International (30%)
    for i in range(num_scenarios * 3 // 10):
        if random.random() < 0.5:
            # Local
            orig = random.choice(NZ_CITIES["major"])
            dest = random.choice([c for c in NZ_CITIES["major"] + NZ_CITIES["secondary"] if c != orig])
            budget = random.randint(800, 2000)
            diff = "easy"
            focus = "Local travel optimization"
        else:
            # International
            orig = "Auckland"
            city_type = random.choice(["short_haul", "medium_haul"])
            dest = random.choice(INTERNATIONAL_CITIES[city_type])
            budget = random.randint(1500, 4000)
            diff = "medium"
            focus = f"International travel ({city_type})"
        
        scenarios.append(TripScenario(
            name=f"travel_{dest}_{i}",
            category="local_vs_international",
            origin=orig,
            destination=dest,
            days=random.choice([3, 4, 5]),
            budget_nzd=budget,
            preferences=["adventure", "culture"],
            expected_difficulty=diff,
            learning_focus=focus
        ))
    
    # Extreme constraints (15%)
    for i in range(num_scenarios * 15 // 100):
        scenarios.append(TripScenario(
            name=f"extreme_{i}",
            category="extreme_constraints",
            origin=random.choice(NZ_CITIES["major"]),
            destination=random.choice(NZ_CITIES["major"]),
            days=random.choice([1, 2, 10, 14]),
            budget_nzd=random.randint(500, 1500),
            preferences=["flexible"],
            expected_difficulty="extreme",
            learning_focus="Handle extreme constraints"
        ))
    
    # Mixed (15%)
    for i in range(num_scenarios * 15 // 100):
        all_cities = NZ_CITIES["major"] + NZ_CITIES["secondary"] + INTERNATIONAL_CITIES["short_haul"]
        scenarios.append(TripScenario(
            name=f"mixed_{i}",
            category="mixed_complexity",
            origin=random.choice(NZ_CITIES["major"]),
            destination=random.choice(all_cities),
            days=random.choice([2, 3, 4, 5, 7]),
            budget_nzd=random.randint(800, 4000),
            preferences=random.sample(["adventure", "culture", "food", "nature"], k=2),
            expected_difficulty=random.choice(["easy", "medium", "hard"]),
            learning_focus="Test strategy generalization"
        ))
    
    random.shuffle(scenarios)
    return scenarios

class LearningMetricsTracker:
    """Track learning metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.category_performance = defaultdict(lambda: {
            "total": 0, "success": 0, "avg_reward": 0.0, "rewards": []
        })
        self.difficulty_performance = defaultdict(lambda: {
            "total": 0, "success": 0, "avg_reward": 0.0
        })
        self.tool_order_performance = defaultdict(lambda: {
            "count": 0, "avg_reward": 0.0, "rewards": []
        })
        self.learning_objectives = {
            "budget_optimization": {"target": 0.85, "current": 0.0, "progress": 0.0},
            "duration_handling": {"target": 0.80, "current": 0.0, "progress": 0.0},
            "tool_order_optimization": {"target": 0.90, "current": 0.0, "progress": 0.0},
            "local_vs_international": {"target": 0.80, "current": 0.0, "progress": 0.0},
            "extreme_constraints": {"target": 0.70, "current": 0.0, "progress": 0.0},
            "mixed_complexity": {"target": 0.75, "current": 0.0, "progress": 0.0},
        }
    
    def record_result(self, scenario: TripScenario, response: Dict[str, Any]):
        learning = response.get("learning") or {}
        reward = learning.get("reward", 0.0)
        success = learning.get("success", False)
        strategy = response.get("strategy_used") or {}
        tool_order = strategy.get("tool_order", "unknown")
        
        # Category performance
        cat = scenario.category
        self.category_performance[cat]["total"] += 1
        if success:
            self.category_performance[cat]["success"] += 1
        self.category_performance[cat]["rewards"].append(reward)
        self.category_performance[cat]["avg_reward"] = sum(self.category_performance[cat]["rewards"]) / len(self.category_performance[cat]["rewards"])
        
        # Difficulty
        diff = scenario.expected_difficulty
        self.difficulty_performance[diff]["total"] += 1
        if success:
            self.difficulty_performance[diff]["success"] += 1
        self.difficulty_performance[diff]["avg_reward"] = (
            (self.difficulty_performance[diff]["avg_reward"] * (self.difficulty_performance[diff]["total"] - 1) + reward) /
            self.difficulty_performance[diff]["total"]
        )
        
        # Tool order
        self.tool_order_performance[tool_order]["count"] += 1
        self.tool_order_performance[tool_order]["rewards"].append(reward)
        self.tool_order_performance[tool_order]["avg_reward"] = sum(self.tool_order_performance[tool_order]["rewards"]) / len(self.tool_order_performance[tool_order]["rewards"])
        
        # Update objectives
        obj_key = scenario.category
        if obj_key in self.learning_objectives:
            obj = self.learning_objectives[obj_key]
            if "rewards" not in obj:
                obj["rewards"] = []
            obj["rewards"].append(reward)
            obj["current"] = sum(obj["rewards"]) / len(obj["rewards"])
            obj["progress"] = min(100, (obj["current"] / obj["target"]) * 100)
        
        # Tool order optimization
        if self.tool_order_performance:
            best = max(self.tool_order_performance.values(), key=lambda x: x["avg_reward"])
            self.learning_objectives["tool_order_optimization"]["current"] = best["avg_reward"]
            self.learning_objectives["tool_order_optimization"]["progress"] = min(100, (best["avg_reward"] / 0.90) * 100)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        return {
            "learning_objectives": self.learning_objectives,
            "category_performance": dict(self.category_performance),
            "difficulty_performance": dict(self.difficulty_performance),
            "tool_order_performance": dict(self.tool_order_performance),
        }

def call_planning_api(scenario: TripScenario, api_url: str = "http://localhost:8888/api/agent/plan") -> Dict[str, Any]:
    payload = {
        "prompt": f"Plan a {scenario.days}-day trip to {scenario.destination} with a budget of ${scenario.budget_nzd} NZD",
        "constraints": {
            "origin_city": scenario.origin,
            "destination_city": scenario.destination,
            "days": scenario.days,
            "budget": scenario.budget_nzd,
            "currency": "NZD",
            "preferences": scenario.preferences,
        },
        "max_iterations": 4
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "error": str(e),
            "learning": {"success": False, "reward": 0.0, "breakdown": {"budget_score": 0.0, "quality_score": 0.0, "reliability_score": 0.0}}
        }

def run_enhanced_learning_test(num_iterations: int = 50, output_dir: str = "./eval"):
    print(f"🚀 Enhanced Learning Test - {num_iterations} iterations")
    print("=" * 80)
    print("\n🎯 Learning Objectives:")
    objectives = [
        "1. Budget Optimization - Target: 0.85",
        "2. Trip Duration Handling - Target: 0.80",
        "3. Tool Order Optimization - Target: 0.90",
        "4. Local vs International - Target: 0.80",
        "5. Extreme Constraints - Target: 0.70",
        "6. Mixed Complexity - Target: 0.75",
    ]
    for obj in objectives:
        print(f"  {obj}")
    print()
    
    scenarios = generate_diverse_scenarios(num_iterations)
    tracker = LearningMetricsTracker()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{num_iterations}] {scenario.name}")
        print(f"  Category: {scenario.category} | Difficulty: {scenario.expected_difficulty}")
        print(f"  {scenario.origin} -> {scenario.destination} | {scenario.days}d | ${scenario.budget_nzd}")
        
        start_time = time.time()
        response = call_planning_api(scenario)
        elapsed = time.time() - start_time
        
        if "error" not in response:
            learning = response.get("learning") or {}
            reward = learning.get("reward", 0.0)
            breakdown = learning.get("breakdown") or {}
            print(f"  ✅ Reward: {reward:.3f} | B:{breakdown.get('budget_score',0):.2f} Q:{breakdown.get('quality_score',0):.2f} R:{breakdown.get('reliability_score',0):.2f}")
            tracker.record_result(scenario, response)
        else:
            print(f"  ❌ Error: {response['error']}")
        
        results.append({"iteration": i, "scenario": scenario.__dict__, "response": response, "elapsed": elapsed})
        
        if i % 10 == 0:
            print(f"\n📊 Progress Update (Iteration {i}):")
            dashboard_data = tracker.get_dashboard_data()
            for obj_name, obj_data in dashboard_data["learning_objectives"].items():
                progress = obj_data.get("progress", 0)
                current = obj_data.get("current", 0)
                target = obj_data.get("target", 0)
                bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
                print(f"  {obj_name:30s} [{bar}] {progress:5.1f}% ({current:.3f}/{target:.3f})")
        
        time.sleep(0.2)
    
    # Save results
    results_file = output_path / f"enhanced_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({"scenarios": [s.__dict__ for s in scenarios], "results": results, "dashboard": tracker.get_dashboard_data()}, f, indent=2)
    
    # Final report
    print("\n" + "=" * 80)
    print("📈 FINAL LEARNING REPORT")
    print("=" * 80)
    dashboard_data = tracker.get_dashboard_data()
    
    print("\n🎯 Learning Objectives:")
    for obj_name, obj_data in dashboard_data["learning_objectives"].items():
        progress = obj_data.get("progress", 0)
        current = obj_data.get("current", 0)
        target = obj_data.get("target", 0)
        status = "✅" if progress >= 100 else "🔶" if progress >= 80 else "⚠️"
        print(f"  {status} {obj_name:30s} {current:.3f}/{target:.3f} ({progress:5.1f}%)")
    
    print("\n📊 Category Performance:")
    for cat, perf in dashboard_data["category_performance"].items():
        success_rate = (perf["success"] / perf["total"] * 100) if perf["total"] > 0 else 0
        print(f"  {cat:25s} | Tests:{perf['total']:3d} | Success:{success_rate:5.1f}% | Reward:{perf['avg_reward']:.3f}")
    
    print("\n🔧 Top Tool Orders:")
    sorted_tools = sorted(dashboard_data["tool_order_performance"].items(), key=lambda x: -x[1]["avg_reward"])
    for i, (tool_order, perf) in enumerate(sorted_tools[:5], 1):
        print(f"  {i}. {tool_order:40s} | Count:{perf['count']:3d} | Reward:{perf['avg_reward']:.3f}")
    
    print(f"\n✅ Results saved to {results_file}")
    print("=" * 80)
    
    return results, tracker

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", "-n", type=int, default=50)
    parser.add_argument("--output-dir", "-o", type=str, default="./eval")
    args = parser.parse_args()
    
    run_enhanced_learning_test(num_iterations=args.iterations, output_dir=args.output_dir)
