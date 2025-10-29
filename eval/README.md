# Enhanced Learning Test Results

This directory contains automated learning test results from the enhanced learning system.

## Directory Structure

```
eval/
├── README.md                                    # This file
├── enhanced_learning_YYYYMMDD_HHMMSS.json      # Test results (one per run)
└── [future visualization files]                 # Charts and graphs
```

## File Format

Each test run generates a JSON file with the following structure:

```json
{
  "scenarios": [
    {
      "name": "scenario_name",
      "category": "budget_optimization | duration_handling | local_vs_international | extreme_constraints | mixed_complexity",
      "origin": "City",
      "destination": "City",
      "days": 3,
      "budget_nzd": 1500,
      "preferences": ["adventure", "culture"],
      "expected_difficulty": "easy | medium | hard | extreme",
      "learning_focus": "Description of what this scenario tests"
    }
  ],
  "results": [
    {
      "iteration": 1,
      "scenario": {...},
      "response": {
        "learning": {
          "success": true,
          "reward": 0.970,
          "breakdown": {
            "budget_score": 1.00,
            "quality_score": 0.90,
            "reliability_score": 1.00
          }
        },
        "strategy_used": {
          "tool_order": "attractions->flights->weather",
          "model_temp": 0.6,
          "attractions_k": 50
        },
        "itinerary": {...},
        "tool_calls": [...],
        "planning_time_ms": 7500
      },
      "elapsed": 8.5
    }
  ],
  "dashboard": {
    "learning_objectives": {
      "budget_optimization": {
        "target": 0.85,
        "current": 0.808,
        "progress": 95.1
      },
      ...
    },
    "category_performance": {
      "budget_optimization": {
        "total": 6,
        "success": 4,
        "avg_reward": 0.808,
        "rewards": [0.45, 1.0, 0.95, ...]
      },
      ...
    },
    "difficulty_performance": {
      "easy": {
        "total": 12,
        "success": 11,
        "avg_reward": 0.95
      },
      ...
    },
    "tool_order_performance": {
      "attractions->flights->weather": {
        "count": 27,
        "avg_reward": 0.837,
        "rewards": [0.97, 0.85, ...]
      },
      ...
    }
  }
}
```

## Learning Objectives

The system tracks 6 learning objectives:

1. **Budget Optimization** (Target: 0.85)
   - Learn to optimize costs across different budget ranges
   - Scenarios: $400-$5000 NZD budgets

2. **Duration Handling** (Target: 0.80)
   - Adapt to different trip lengths
   - Scenarios: 2-14 day trips

3. **Tool Order Optimization** (Target: 0.90)
   - Find optimal tool calling sequence
   - Best: `attractions->flights->weather`

4. **Local vs International** (Target: 0.80)
   - Differentiate local and international travel strategies
   - Scenarios: NZ cities, Australia, Asia

5. **Extreme Constraints** (Target: 0.70)
   - Handle extreme budget/duration constraints
   - Scenarios: Ultra-low budget, ultra-short/long duration

6. **Mixed Complexity** (Target: 0.75)
   - Test strategy generalization
   - Scenarios: Random combinations

## Performance Metrics

### Reward Breakdown
Each test receives a reward (0.0-1.0) composed of:
- **Budget Score (40%)**: How well the plan stays within budget
- **Quality Score (30%)**: Quality of attractions and experiences
- **Reliability Score (30%)**: Tool success rate, no errors

### Success Criteria
- **Success**: Reward > 0.5
- **Partial Success**: 0.3 < Reward ≤ 0.5
- **Failure**: Reward ≤ 0.3

### Difficulty Levels
- **Easy**: Local trips, ample budget (Expected reward: 0.90+)
- **Medium**: Short international, medium budget (Expected: 0.80+)
- **Hard**: Long trips, tight budget (Expected: 0.60+)
- **Extreme**: Ultra constraints (Expected: 0.40+)

## How to Analyze Results

### 1. Check Overall Achievement
Look at `dashboard.learning_objectives`:
```python
import json
with open('enhanced_learning_20251029_075119.json') as f:
    data = json.load(f)

for obj, metrics in data['dashboard']['learning_objectives'].items():
    print(f"{obj}: {metrics['current']:.3f}/{metrics['target']:.3f} ({metrics['progress']:.1f}%)")
```

### 2. Analyze Category Performance
```python
for category, perf in data['dashboard']['category_performance'].items():
    success_rate = perf['success'] / perf['total'] * 100
    print(f"{category}: {success_rate:.1f}% success, avg reward {perf['avg_reward']:.3f}")
```

### 3. Find Best Tool Order
```python
tool_orders = data['dashboard']['tool_order_performance']
best = max(tool_orders.items(), key=lambda x: x[1]['avg_reward'])
print(f"Best tool order: {best[0]} (reward: {best[1]['avg_reward']:.3f})")
```

### 4. Identify Problem Scenarios
```python
low_reward_results = [r for r in data['results']
                      if r['response'].get('learning', {}).get('reward', 0) < 0.5]
print(f"Found {len(low_reward_results)} low-reward scenarios")
for r in low_reward_results:
    scenario = r['scenario']
    print(f"  - {scenario['name']}: {scenario['category']}, difficulty={scenario['expected_difficulty']}")
```

## Running New Tests

### Quick Test (20 iterations)
```bash
python enhanced_learning_test.py --iterations 20
```

### Standard Test (50 iterations)
```bash
python enhanced_learning_test.py --iterations 50
```

### Full Test (100 iterations)
```bash
python enhanced_learning_test.py --iterations 100
```

## Files in This Directory

- `enhanced_learning_YYYYMMDD_HHMMSS.json` - Full test results with all scenarios, responses, and dashboard metrics
- Future additions may include:
  - Visualization charts (PNG/SVG)
  - Summary reports (TXT/MD)
  - Comparison analyses across multiple runs

## Learning Data Persistence

Learning data is also saved to:
- `data/agent_experiences.jsonl` - Experience memory (one JSON per line)

This memory persists across test runs, allowing the system to improve over time through the Bandit algorithm.

## Documentation

For detailed information, see:
- `ENHANCED_LEARNING_SUMMARY_EN.md` - Complete learning system documentation
- `QUICK_START_LEARNING_EN.md` - Quick start guide
- `ENHANCED_LEARNING_SUMMARY.md` - 中文详细文档
- `QUICK_START_LEARNING.md` - 中文快速开始指南

---

**Last Updated**: 2025-10-29
**System Version**: Enhanced Learning v1.0
