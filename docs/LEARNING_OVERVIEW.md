# 🎓 AI Assessment Project - Learning System Overview

This project implements two independent learning systems to optimize the performance of Trip Planning Agent and Code Generation Agent.

---

## 📁 File Organization

### Core Code
```
backend/backend/services/
├── autoplan_learn/          # ✈️ Trip Planning Learning
│   ├── strategies.py        # 6 planning strategies
│   ├── signature.py         # Task signature (7-dim numerical vector)
│   ├── memory.py            # Memory storage (JSONL)
│   ├── adapter.py           # Reward calculation & strategy selection
│   └── bandit.py            # Thompson Sampling implementation
│
└── codegen_learn/           # 💻 Code Generation Learning
    ├── strategies.py        # 6 code generation strategies
    ├── signature.py         # Task signature (7-dim mixed structure)
    ├── memory.py            # Memory storage (JSONL)
    ├── adapter.py           # Reward calculation & strategy selection
    └── bandit.py            # Thompson Sampling implementation
```

### Test Files
```
ai-assessment-deploy/
├── test_trip_learning_basic.py          # ✈️ Trip basic tests
├── test_trip_learning_direct.py         # ✈️ Trip direct test (no API)
├── test_trip_learning_visualization.py  # ✈️ Trip learning curve viz
├── test_trip_learning_tiered.py         # ✈️ Trip 100-test suite
├── test_code_learning_tiered.py         # 💻 Code tiered tests (100 questions)
└── test_code_learning_visualization.py  # 💻 Code learning curve viz
```

### Learning Data
```
data/
├── trip_experiences.jsonl      # ✈️ Trip learning memory
└── codegen_experiences.jsonl   # 💻 Code learning memory
```

### Documentation
```
docs/
├── LEARNING_OVERVIEW.md        # 📖 This document (overview)
├── TRIP_LEARNING.md            # ✈️ Trip Learning details
└── CODE_LEARNING.md            # 💻 Code Learning details
```

---

## 🎯 Comparison of Two Learning Systems

| Dimension | ✈️ Trip Planning | 💻 Code Generation |
|-----------|------------------|-------------------|
| **Learning Goals** | Budget control, reduce tool errors, improve quality | Improve first-try success, optimize quality, adapt to difficulty |
| **Task Signature** | 7-dim numerical vector (budget, days, geography) | 7-dim mixed (language, type, complexity) |
| **Number of Strategies** | 6 | 6 |
| **Similarity Threshold** | 0.55 (standard) | 0.65 (conservative) |
| **Reward Weights** | Budget 40% + Quality 30% + Reliability 30% | Success 50% + Efficiency 25% + Quality 15% + Speed 10% |
| **Main Challenges** | API instability, budget optimization | GPT uncertainty, large difficulty variance |
| **Convergence Speed** | Fast (50-100 iterations) | Slow (requires tiered learning) |
| **Test Cases** | 100 (diverse scenarios) | 100 (30 simple + 40 medium + 30 complex) |

---

## 🚀 Quick Start

### 1. Test Trip Learning

```bash
# Basic test (10 questions)
python test_trip_learning_basic.py --quick

# Full test (100 questions)
python test_trip_learning_tiered.py

# View learning curve
python test_trip_learning_visualization.py
```

### 2. Test Code Learning

```bash
# Quick test (10: 3 simple + 4 medium + 3 complex)
python test_code_learning_tiered.py --quick

# Full test (100 questions)
python test_code_learning_tiered.py

# Test simple only
python test_code_learning_tiered.py --simple

# View learning curve
python test_code_learning_visualization.py
```

### 3. View Real-time Dashboard

After starting the app, check the sidebar:
- **✈️ 🎓 Trip Learning Dashboard**: Shows Trip learning stats
- **💻 🎓 Code Learning Dashboard**: Shows Code learning stats

---

## 🧠 Core Algorithms

Both learning systems use the same algorithmic framework:

### 1. Thompson Sampling (Multi-Armed Bandit)

```python
# Each strategy maintains a Beta distribution
strategy.alpha += 1  # On success
strategy.beta += 1   # On failure

# Select strategy
samples = [beta.rvs(s.alpha, s.beta) for s in strategies]
best_strategy = strategies[argmax(samples)]
```

### 2. Nearest Neighbor Retrieval

```python
# Calculate task similarity
distance = euclidean(current_signature, historical_signature)

# Reuse if very similar
if distance < threshold:
    return best_historical_strategy
else:
    return thompson_sampling_strategy
```

### 3. Multi-Dimensional Reward

```python
# Trip: Budget + Quality + Reliability
trip_reward = 0.40 * budget_score + 0.30 * quality_score + 0.30 * reliability_score

# Code: Success + Efficiency + Quality + Speed
code_reward = 0.50 * success + 0.25 * efficiency + 0.15 * quality + 0.10 * speed
```

---

## 📊 Expected Learning Effects

### Trip Planning (After 100 iterations)

| Metric | Initial | After Learning | Improvement |
|--------|---------|----------------|-------------|
| Average Reward | 0.650 | 0.850 | +30.8% |
| Success Rate | 70% | 92% | +22% |
| Average Iterations | 2.5 | 1.8 | -28% |
| Budget Accuracy | 75% | 90% | +15% |

### Code Generation (After 100 iterations)

| Metric | Initial | After Learning | Improvement |
|--------|---------|----------------|-------------|
| Average Reward | 0.850 | 0.920 | +8.2% |
| First-try Success | 75% | 88% | +13% |
| Avg Retries (Simple) | 0.2 | 0.1 | -50% |
| Avg Retries (Medium) | 1.5 | 0.8 | -46.7% |
| Avg Retries (Complex) | 2.5 | 1.5 | -40% |

---

## 🔧 Configuration

### Environment Variables

```bash
# Trip Learning
export TRIP_MEMORY_PATH="data/trip_experiences.jsonl"

# Code Learning
export CODEGEN_MEMORY_PATH="data/codegen_experiences.jsonl"
```

### Adjust Similarity Thresholds

```python
# backend/services/autoplan_learn/adapter.py
self.similarity_threshold = 0.55  # Trip: standard threshold

# backend/services/codegen_learn/adapter.py
self.similarity_threshold = 0.65  # Code: conservative (GPT instability)
```

---

## 🐛 Debugging Guide

### Check if Learning is Enabled

```python
# Trip
from backend.services.planning_agent import PlanningAgent
agent = PlanningAgent(enable_learning=True)
print(agent.learner)  # Should not be None

# Code
from backend.services.code_assistant import CodeAssistant
assistant = CodeAssistant(enable_learning=True)
print(assistant.learner)  # Should not be None
```

### View Learning Data

```bash
# Memory size
wc -l data/trip_experiences.jsonl
wc -l data/codegen_experiences.jsonl

# Strategy distribution
grep -o '"strategy":"[^"]*"' data/trip_experiences.jsonl | sort | uniq -c
grep -o '"strategy":"[^"]*"' data/codegen_experiences.jsonl | sort | uniq -c

# Average reward
grep -o '"reward":[0-9.]*' data/trip_experiences.jsonl | \
  awk -F: '{sum+=$2; n++} END {print sum/n}'
```

### Reset Learning (Start Fresh)

```bash
rm data/trip_experiences.jsonl
rm data/codegen_experiences.jsonl
```

---

## 📚 Detailed Documentation

- **✈️ Trip Learning**: See [docs/TRIP_LEARNING.md](TRIP_LEARNING.md)
- **💻 Code Learning**: See [docs/CODE_LEARNING.md](CODE_LEARNING.md)

---

## 🎉 Summary

This learning system achieves:

1. ✅ **Automatic Strategy Optimization** - No manual tuning, system learns optimal strategies
2. ✅ **Historical Experience Reuse** - Similar tasks directly use best historical strategy
3. ✅ **Exploration-Exploitation Balance** - Thompson Sampling auto-balances exploration vs exploitation
4. ✅ **Multi-Dimensional Evaluation** - Considers success rate, efficiency, quality, cost, etc.
5. ✅ **Real-time Feedback** - Frontend Dashboard shows learning progress in real-time
6. ✅ **Tiered Testing** - Code Learning uses tiered tests for different difficulty levels

**Core Advantage**: Performance continuously improves with usage, no human intervention required.
