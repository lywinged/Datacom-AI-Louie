# 💻 Code Generation Agent - Learning System

## Learning Objectives

The Code Generation Agent learning system automatically selects optimal code generation strategies based on task difficulty and type:

1. **Improve First-Try Success Rate** - Learn which strategies work for different difficulty tasks, reduce retries
2. **Optimize Code Quality** - Select strategies that generate high-quality code (type hints, test coverage)
3. **Accelerate Generation Speed** - Use fast strategies for simple tasks, high-quality strategies for complex tasks
4. **Adapt to Task Types** - Choose appropriate strategies for algorithms, data processing, OOP, etc.

---

## Algorithm Implementation

### 1. **Multi-Armed Bandit (Thompson Sampling)**

Similar to Trip Agent, uses Thompson Sampling to balance exploration and exploitation:

- **6 Strategies**:
  - `fast_pragmatic`: Fast pragmatic (minimal planning, low temperature, quick generation)
  - `detailed_planner`: Detailed planner (complete plan, moderate temperature, full comments)
  - `test_driven`: Test-driven (write tests first, high coverage requirements)
  - `iterative_refiner`: Iterative refiner (incremental fixes, multiple validation rounds)
  - `high_quality`: High quality (low temperature, type hints, multiple validation)
  - `balanced`: Balanced strategy (moderate planning and quality requirements)

### 2. **Task Signature (7-dim mixed structure)**

Automatically analyzes task features:

```python
signature = {
    "language": "python",              # Programming language
    "task_type": "algorithm",          # Task type
    "complexity": "medium",            # Complexity (simple/medium/complex)
    "test_framework": "pytest",        # Test framework
    "has_external_deps": False,        # Has external dependencies
    "estimated_loc": 30,               # Estimated lines of code
    "failure_pattern": None            # Failure pattern (for retries)
}
```

**Complexity Evaluation Indicators**:
```python
complexity_indicators = [
    len(task.split()) > 50,                    # Description length
    "complex" or "advanced" in task,           # Keywords
    "multiple" or "several" in task,           # Multi-step
    "optimize" or "efficient" in task,         # Performance requirements
]
complexity = sum(indicators)  # 0-1: simple, 2: medium, 3+: complex
```

### 3. **Multi-Dimensional Reward**

Comprehensive evaluation of code generation quality:

```python
reward = 0.50 * success_score      # Success rate (50%)
       + 0.25 * efficiency_score   # Efficiency (25%)
       + 0.15 * quality_score      # Quality (15%)
       + 0.10 * speed_score        # Speed (10%)
```

### 4. **Nearest Neighbor with Conservative Threshold**

Due to GPT-4o instability, uses **more conservative threshold**:

- **Similarity Threshold**: `0.65` (vs Trip Agent's 0.55)
- **Distance Calculation**: Normalized Euclidean distance (7-dim signature vector)
- **Strategy**: Only reuse historical strategy for very similar tasks

---

## Usage

### 1. **API Usage**

```python
from backend.services.code_assistant import CodeAssistant

# Enable learning on initialization
assistant = CodeAssistant(enable_learning=True)

# Send request
response = await assistant.generate_code(
    task="Implement binary search",
    language="python",
    max_retries=3
)

# View learning feedback
if response.learning:
    print(f"Strategy: {response.learning['strategy']}")
    print(f"Reward: {response.learning['reward']:.3f}")
```

### 2. **Learning Data Storage**

- **Path**: `data/codegen_experiences.jsonl` (configurable via `CODEGEN_MEMORY_PATH`)
- **Format**: One JSON object per line

### 3. **View Learning Effects**

**In Frontend**:
- Sidebar "💻 🎓 Code Learning Dashboard" shows real-time stats

**Run Tests**:
```bash
# Quick test (10: 3 simple + 4 medium + 3 complex)
python test_code_learning_tiered.py --quick

# Full tiered test (100: 30 simple + 40 medium + 30 complex)
python test_code_learning_tiered.py

# Test specific difficulty
python test_code_learning_tiered.py --simple   # 30 tests
python test_code_learning_tiered.py --medium   # 40 tests
python test_code_learning_tiered.py --complex  # 30 tests

# Visualization
python test_code_learning_visualization.py
```

---

## Strategy Details

| Strategy | Planning Depth | GPT Temp | Use Case | Expected Difficulty |
|----------|----------------|----------|----------|-------------------|
| **fast_pragmatic** | Minimal | 0.2 | Simple utility functions | 🟢 Simple |
| **balanced** | Moderate | 0.3 | General tasks | 🟢🟡 Simple/Medium |
| **detailed_planner** | Detailed | 0.3 | Multi-step logic | 🟡 Medium |
| **test_driven** | Moderate | 0.3 | High test coverage needed | 🟡🔴 Medium/Complex |
| **high_quality** | Detailed | 0.2 | Complex algorithms | 🔴 Complex |
| **iterative_refiner** | Progressive | 0.4 | Multi-round optimization | 🟡🔴 Medium/Complex |

---

## Tiered Test Results Example

**Quick Test (10 questions) - Initial without learning data**:
```
🎯 Results by Difficulty Tier:
   🟢 SIMPLE: 3/3 (100.0%)
      Avg Retries: 0.00
      Avg Reward: 0.940

   🟡 MEDIUM: 4/4 (100.0%)
      Avg Retries: 0.00
      Avg Reward: 0.932

   🔴 COMPLEX: 3/3 (100.0%)
      Avg Retries: 0.33
      Avg Reward: 0.895
```

**After 100 Iterations (Expected)**:
- 🟢 Simple: 95%+ use `fast_pragmatic`/`balanced`
- 🟡 Medium: 60%+ use `detailed_planner`/`test_driven`
- 🔴 Complex: 70%+ use `high_quality`/`test_driven`

---

## Key Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `CODEGEN_MEMORY_PATH` | `data/codegen_experiences.jsonl` | Learning memory path |
| `similarity_threshold` | `0.65` | Nearest neighbor threshold (conservative) |
| `max_memory_size` | `10000` | Maximum memory entries |
| `max_retries` | `3` | Maximum retry attempts |

---

## Differences from Trip Agent Learning

| Dimension | Trip Agent | Code Agent |
|-----------|------------|------------|
| **Task Signature** | 7-dim numerical vector | 7-dim mixed structure (strings+numbers) |
| **Similarity Threshold** | 0.55 (standard) | 0.65 (conservative) |
| **Reward Weights** | Budget 40% + Quality 30% + Reliability 30% | Success 50% + Efficiency 25% + Quality 15% + Speed 10% |
| **Number of Strategies** | 6 | 6 |
| **Main Challenges** | Budget control, tool errors | GPT instability, large difficulty variance |
| **Learning Speed** | Fast (50-100 iterations converge) | Slow (requires tiered learning) |
