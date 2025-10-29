# ✈️ Trip Planning Agent - Learning System

## Learning Objectives

The Trip Planning Agent learning system automatically optimizes trip planning strategies through historical data to achieve:

1. **Improve Budget Control Accuracy** - Learn how to generate optimal itineraries within budget
2. **Reduce Tool Call Errors** - Select reliable strategies to reduce API failure rates
3. **Enhance Trip Quality** - Balance number of attractions, hotel ratings, overall satisfaction
4. **Accelerate Planning Speed** - Improve efficiency by reducing iteration count

---

## Algorithm Implementation

### 1. **Multi-Armed Bandit (Thompson Sampling)**

Uses Thompson Sampling algorithm to automatically balance **Exploration** and **Exploitation**:

- **6 Strategies**:
  - `fast_basic`: Fast basic strategy (low temperature, fewer tool calls)
  - `thorough_planner`: Thorough planner (high k value, more attractions)
  - `budget_focused`: Budget priority (low budget tolerance)
  - `quality_seeker`: Quality priority (high hotel rating requirement)
  - `balanced`: Balanced strategy
  - `adaptive`: Adaptive strategy (high temperature, flexible adjustment)

- **Thompson Sampling**:
  - Each strategy maintains a Beta distribution (α, β)
  - Update distribution based on historical success/failure
  - Sample from distribution each time to select strategy with maximum value

### 2. **Nearest Neighbor Retrieval**

Retrieve best historical strategy based on task similarity:

- **Task Signature**: 7-dimensional vector
  ```python
  [
    log(budget_usd),           # Log budget (USD)
    days,                       # Days
    origin_continent_id,        # Origin continent ID
    destination_continent_id,   # Destination continent ID
    is_domestic,                # Domestic trip (0/1)
    is_budget_constrained,      # Budget constrained (0/1)
    has_explicit_preferences    # Has explicit preferences (0/1)
  ]
  ```

- **Similarity Calculation**: Euclidean distance (L2 norm)
- **Threshold**: 0.55 (use historical best strategy for similar tasks, otherwise use Bandit)

### 3. **Multi-Dimensional Reward**

Comprehensive evaluation of planning quality:

```python
reward = 0.40 * budget_score      # Budget control (40%)
       + 0.30 * quality_score     # Trip quality (30%)
       + 0.30 * reliability_score # Reliability (30%)
```

**Budget Score**:
```python
if total_cost <= budget:
    score = 1.0 - (cost_diff / budget)  # Closer to budget is better
else:
    score = max(0, 1.0 - 2 * overspend_ratio)  # Double penalty for overspending
```

**Quality Score**:
```python
attraction_score = min(1.0, num_attractions / expected_attractions)
accommodation_score = hotel_rating / 5.0
quality = 0.5 * attraction_score + 0.5 * accommodation_score
```

**Reliability Score**:
```python
reliability = max(0, 1.0 - (tool_errors / max_errors_threshold))
```

---

## Usage

### 1. **API Usage**

```python
from backend.services.planning_agent import PlanningAgent

# Enable learning on initialization
agent = PlanningAgent(enable_learning=True)

# Send request
response = await agent.create_plan(request)

# View learning feedback
if response.learning:
    print(f"Strategy: {response.learning['strategy']}")
    print(f"Reward: {response.learning['reward']:.3f}")
    print(f"Success: {response.learning['success']}")
```

### 2. **Learning Data Storage**

- **Path**: `data/trip_experiences.jsonl` (configurable via `TRIP_MEMORY_PATH` env var)
- **Format**: One JSON object per line
  ```json
  {
    "signature": [6.21, 3, 4, 4, 0, 1, 0],
    "strategy": "balanced",
    "reward": 0.823,
    "success": true,
    "metadata": {
      "budget_score": 0.876,
      "quality_score": 0.720,
      "reliability_score": 1.000,
      "timestamp": "2025-10-29T10:15:30"
    }
  }
  ```

### 3. **View Learning Effects**

**In Frontend**:
- Sidebar "✈️ 🎓 Trip Learning Dashboard" shows real-time:
  - Average reward trend
  - Strategy distribution
  - Learning objective progress

**Run Tests**:
```bash
# Basic functionality test
python test_trip_learning_basic.py

# Direct test (without starting API)
python test_trip_learning_direct.py

# 100-question comprehensive test
python test_trip_learning_tiered.py

# Visualization analysis
python test_trip_learning_visualization.py
```

---

## Key Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `TRIP_MEMORY_PATH` | `data/trip_experiences.jsonl` | Learning memory storage path |
| `similarity_threshold` | `0.55` | Nearest neighbor threshold |
| `max_memory_size` | `10000` | Maximum memory entries |
| `budget_tolerance` | Strategy-dependent | Budget tolerance (0.05-0.20) |
| `model_temp` | Strategy-dependent | GPT temperature (0.3-0.8) |

---

## Learning Effect Examples

**Initial** (no historical data):
- Average Reward: ~0.650
- Success Rate: ~70%
- Average Iterations: 2.5

**After 100 Iterations**:
- Average Reward: ~0.850
- Success Rate: ~92%
- Average Iterations: 1.8
- Strategy Convergence: `balanced` 45%, `thorough_planner` 30%

---

## Debugging Tips

1. **Check if learning is enabled**:
   ```python
   print(agent.learner)  # Should not be None
   ```

2. **View memory size**:
   ```bash
   wc -l data/trip_experiences.jsonl
   ```

3. **Analyze strategy distribution**:
   ```python
   from backend.services.autoplan_learn import TripAdapter
   adapter = TripAdapter("data/trip_experiences.jsonl")
   adapter.bandit.get_strategy_stats()
   ```

4. **Clear learning memory** (start fresh):
   ```bash
   rm data/trip_experiences.jsonl
   ```
