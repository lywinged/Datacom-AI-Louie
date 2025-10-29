#!/usr/bin/env python3
"""
Automated Learning Visualization Tool
Generates 100 trip planning interactions and visualizes learning improvements
"""

from __future__ import annotations
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from datetime import datetime, timedelta
import requests

# Trip planning request templates
CITIES = [
    "Auckland", "Wellington", "Queenstown", "Christchurch", "Rotorua",
    "Dunedin", "Hamilton", "Tauranga", "Napier", "Nelson"
]

TRIP_TEMPLATES = [
    "Plan a {days}-day trip to {city} with a budget of ${budget} NZD",
    "I want to visit {city} for {days} days, budget is ${budget} NZD",
    "Help me plan a {days}-day vacation in {city}, I have ${budget} NZD",
    "Create an itinerary for {city}, {days} days, ${budget} NZD budget",
    "Planning a {days}-day adventure to {city} with ${budget} NZD to spend",
]

BUDGET_RANGES = [
    (500, 1000),    # Budget trips
    (1000, 2000),   # Mid-range
    (2000, 3500),   # Comfortable
    (3500, 5000),   # Premium
]

DAYS_OPTIONS = [2, 3, 4, 5, 7]

def generate_test_requests(n: int = 100) -> List[Dict[str, Any]]:
    """Generate n diverse trip planning requests"""
    requests_list = []

    for i in range(n):
        city = random.choice(CITIES)
        days = random.choice(DAYS_OPTIONS)
        budget_range = random.choice(BUDGET_RANGES)
        budget = random.randint(budget_range[0], budget_range[1])
        template = random.choice(TRIP_TEMPLATES)

        prompt = template.format(city=city, days=days, budget=budget)

        # Add date range for some requests
        start_date = datetime.now() + timedelta(days=random.randint(7, 60))
        date_range = None
        if random.random() > 0.5:
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {(start_date + timedelta(days=days)).strftime('%Y-%m-%d')}"

        requests_list.append({
            "id": i + 1,
            "prompt": prompt,
            "city": city,
            "days": days,
            "budget": budget,
            "date_range": date_range
        })

    return requests_list


def call_planning_agent(request: Dict[str, Any], api_url: str = "http://localhost:8888/api/agent/plan") -> Dict[str, Any]:
    """Call the trip planning agent API"""
    payload = {
        "prompt": request["prompt"],
        "constraints": {
            "city": request["city"],
            "days": request["days"],
            "budget_nzd": request["budget"],
            "date_range": request.get("date_range")
        }
    }

    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def extract_metrics(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract learning metrics from agent response"""
    learning = response.get("learning", {})
    itinerary = response.get("itinerary", {})

    return {
        "reward": learning.get("reward", 0.0),
        "success": learning.get("success", False),
        "budget_score": learning.get("breakdown", {}).get("budget_score", 0.0),
        "quality_score": learning.get("breakdown", {}).get("quality_score", 0.0),
        "reliability_score": learning.get("breakdown", {}).get("reliability_score", 0.0),
        "strategy": response.get("strategy_used", {}),
        "tool_errors": response.get("tool_errors_count", 0),
        "total_cost": itinerary.get("total_cost_nzd", 0.0),
    }


def run_experiments(n: int = 100, output_dir: str = "eval") -> Dict[str, Any]:
    """Run n automated trip planning experiments"""
    print(f"🚀 Starting {n} automated trip planning experiments...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    requests_list = generate_test_requests(n)
    results = []

    for i, req in enumerate(requests_list, 1):
        print(f"\n[{i}/{n}] Processing: {req['prompt'][:60]}...")

        start_time = time.time()
        response = call_planning_agent(req)
        elapsed = time.time() - start_time

        if "error" in response:
            print(f"  ❌ Error: {response['error']}")
            metrics = {
                "reward": 0.0,
                "success": False,
                "budget_score": 0.0,
                "quality_score": 0.0,
                "reliability_score": 0.0,
                "strategy": {},
                "tool_errors": 0,
                "total_cost": 0.0,
            }
        else:
            metrics = extract_metrics(response)
            print(f"  ✅ Reward: {metrics['reward']:.3f} | Budget: {metrics['budget_score']:.2f} | Quality: {metrics['quality_score']:.2f} | Reliability: {metrics['reliability_score']:.2f}")

        result = {
            "iteration": i,
            "request": req,
            "metrics": metrics,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }

        results.append(result)

        # Save intermediate results every 10 iterations
        if i % 10 == 0:
            with open(output_path / "intermediate_results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

        # Small delay to avoid overwhelming the API
        time.sleep(0.5)

    # Save final results
    results_file = output_path / f"learning_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Experiment complete! Results saved to {results_file}")

    return {
        "results": results,
        "output_file": str(results_file)
    }


def visualize_learning(results: List[Dict[str, Any]], output_dir: str = "eval"):
    """Create comprehensive learning visualizations"""
    print("\n📊 Generating visualizations...")

    output_path = Path(output_dir)

    # Extract time series data
    iterations = [r["iteration"] for r in results]
    rewards = [r["metrics"]["reward"] for r in results]
    budget_scores = [r["metrics"]["budget_score"] for r in results]
    quality_scores = [r["metrics"]["quality_score"] for r in results]
    reliability_scores = [r["metrics"]["reliability_score"] for r in results]
    tool_errors = [r["metrics"]["tool_errors"] for r in results]

    # Calculate moving averages (window=10)
    def moving_average(data, window=10):
        if len(data) < window:
            return data
        return [sum(data[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(data))]

    rewards_ma = moving_average(rewards)
    budget_ma = moving_average(budget_scores)
    quality_ma = moving_average(quality_scores)
    reliability_ma = moving_average(reliability_scores)

    # Strategy distribution
    strategy_counts = {}
    for r in results:
        strategy_key = json.dumps(r["metrics"]["strategy"], sort_keys=True)
        strategy_counts[strategy_key] = strategy_counts.get(strategy_key, 0) + 1

    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))

    # 1. Learning Curve (Reward over time)
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(iterations, rewards, alpha=0.3, s=20, label='Raw reward', color='lightblue')
    ax1.plot(iterations, rewards_ma, linewidth=2, label='Moving avg (10)', color='darkblue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning Curve: Reward Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Multidimensional Scores
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(iterations, budget_ma, label='Budget Score', linewidth=2, color='green')
    ax2.plot(iterations, quality_ma, label='Quality Score', linewidth=2, color='orange')
    ax2.plot(iterations, reliability_ma, label='Reliability Score', linewidth=2, color='purple')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Score')
    ax2.set_title('Multidimensional Score Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Tool Errors Over Time
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(iterations, tool_errors, alpha=0.5, s=30, color='red')
    error_ma = moving_average(tool_errors)
    ax3.plot(iterations, error_ma, linewidth=2, label='Moving avg (10)', color='darkred')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Tool Errors')
    ax3.set_title('Tool Error Rate Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Success Rate Over Time (sliding window)
    ax4 = plt.subplot(3, 3, 4)
    success_flags = [1 if r["metrics"]["success"] else 0 for r in results]
    success_rate = moving_average(success_flags, window=20)
    ax4.plot(iterations, success_rate, linewidth=2, color='green')
    ax4.fill_between(iterations, success_rate, alpha=0.3, color='green')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Success Rate (20-iteration window)')
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3)

    # 5. Strategy Distribution
    ax5 = plt.subplot(3, 3, 5)
    top_strategies = sorted(strategy_counts.items(), key=lambda x: -x[1])[:8]
    strategy_labels = [f"S{i+1}" for i in range(len(top_strategies))]
    strategy_values = [v for k, v in top_strategies]
    bars = ax5.barh(strategy_labels, strategy_values, color='skyblue')
    ax5.set_xlabel('Frequency')
    ax5.set_title('Top 8 Strategy Selection Frequency')
    ax5.grid(True, alpha=0.3, axis='x')

    # Color the bars by frequency
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(strategy_values[i] / max(strategy_values)))

    # 6. Reward Distribution Histogram
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(rewards, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax6.axvline(sum(rewards) / len(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {sum(rewards)/len(rewards):.3f}')
    ax6.set_xlabel('Reward')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Reward Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Budget vs Quality Scatter
    ax7 = plt.subplot(3, 3, 7)
    scatter = ax7.scatter(budget_scores, quality_scores, c=rewards, s=50, alpha=0.6, cmap='viridis')
    ax7.set_xlabel('Budget Score')
    ax7.set_ylabel('Quality Score')
    ax7.set_title('Budget vs Quality (color=reward)')
    plt.colorbar(scatter, ax=ax7, label='Reward')
    ax7.grid(True, alpha=0.3)

    # 8. Cumulative Success Count
    ax8 = plt.subplot(3, 3, 8)
    cumulative_success = [sum(success_flags[:i+1]) for i in range(len(success_flags))]
    ax8.plot(iterations, cumulative_success, linewidth=2, color='darkgreen')
    ax8.fill_between(iterations, cumulative_success, alpha=0.3, color='green')
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Cumulative Successes')
    ax8.set_title('Cumulative Success Count')
    ax8.grid(True, alpha=0.3)

    # 9. Learning Efficiency (Reward per iteration quartiles)
    ax9 = plt.subplot(3, 3, 9)
    quartile_size = len(results) // 4
    quartiles = []
    quartile_labels = ['Q1\n(1-25)', 'Q2\n(26-50)', 'Q3\n(51-75)', 'Q4\n(76-100)']
    for i in range(4):
        start_idx = i * quartile_size
        end_idx = start_idx + quartile_size if i < 3 else len(results)
        quartile_rewards = rewards[start_idx:end_idx]
        quartiles.append(quartile_rewards)

    bp = ax9.boxplot(quartiles, labels=quartile_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax9.set_ylabel('Reward')
    ax9.set_title('Learning Progress by Quartile')
    ax9.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save visualization
    viz_file = output_path / f"learning_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {viz_file}")

    # Generate summary statistics
    summary = {
        "total_iterations": len(results),
        "overall_success_rate": sum(success_flags) / len(success_flags),
        "average_reward": sum(rewards) / len(rewards),
        "reward_improvement": (sum(rewards[-20:]) / 20) - (sum(rewards[:20]) / 20) if len(rewards) >= 40 else 0,
        "average_budget_score": sum(budget_scores) / len(budget_scores),
        "average_quality_score": sum(quality_scores) / len(quality_scores),
        "average_reliability_score": sum(reliability_scores) / len(reliability_scores),
        "total_tool_errors": sum(tool_errors),
        "average_tool_errors": sum(tool_errors) / len(tool_errors),
        "unique_strategies_used": len(strategy_counts),
        "most_popular_strategy_frequency": max(strategy_counts.values()) if strategy_counts else 0,
    }

    # Save summary
    summary_file = output_path / "learning_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📈 Learning Summary:")
    print(f"  Total Iterations: {summary['total_iterations']}")
    print(f"  Overall Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"  Average Reward: {summary['average_reward']:.3f}")
    print(f"  Reward Improvement (first 20 vs last 20): {summary['reward_improvement']:+.3f}")
    print(f"  Budget Score: {summary['average_budget_score']:.3f}")
    print(f"  Quality Score: {summary['average_quality_score']:.3f}")
    print(f"  Reliability Score: {summary['average_reliability_score']:.3f}")
    print(f"  Average Tool Errors: {summary['average_tool_errors']:.2f}")
    print(f"  Unique Strategies Used: {summary['unique_strategies_used']}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated Learning Visualization Tool")
    parser.add_argument("--iterations", "-n", type=int, default=100, help="Number of test iterations (default: 100)")
    parser.add_argument("--output-dir", "-o", type=str, default="eval", help="Output directory")
    parser.add_argument("--api-url", type=str, default="http://localhost:8888/api/agent/plan", help="Trip planning agent API URL")
    parser.add_argument("--skip-experiments", action="store_true", help="Skip experiments and only visualize existing results")
    parser.add_argument("--results-file", type=str, help="Path to existing results file for visualization")

    args = parser.parse_args()

    if args.skip_experiments and args.results_file:
        print(f"📂 Loading existing results from {args.results_file}...")
        with open(args.results_file, "r") as f:
            data = json.load(f)
            results = data if isinstance(data, list) else data.get("results", [])

        visualize_learning(results, args.output_dir)
    else:
        # Run experiments
        experiment_data = run_experiments(n=args.iterations, output_dir=args.output_dir)

        # Visualize results
        visualize_learning(experiment_data["results"], args.output_dir)

    print("\n🎉 All done! Check the output directory for results and visualizations.")
