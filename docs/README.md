# 📚 AI Assessment Project - Documentation

Welcome to the AI Assessment Project learning system documentation.

## 📖 Documentation Index

### 🌟 Overview
- [**LEARNING_OVERVIEW.md**](LEARNING_OVERVIEW.md) - Learning System Overview (Start Here)
  - File organization structure
  - Comparison of two learning systems
  - Quick start guide
  - Core algorithm introduction
  - Expected learning effects

### ✈️ Trip Planning Learning
- [**TRIP_LEARNING.md**](TRIP_LEARNING.md) - Trip Agent Learning System Details
  - Learning objectives: budget control, reduce errors, improve quality
  - Algorithm implementation: Thompson Sampling + Nearest Neighbor
  - Usage: API calls, data storage, view effects
  - 6 strategy details
  - Debugging tips

### 💻 Code Generation Learning
- [**CODE_LEARNING.md**](CODE_LEARNING.md) - Code Agent Learning System Details
  - Learning objectives: improve success rate, optimize quality, adapt to difficulty
  - Algorithm implementation: Thompson Sampling + conservative threshold
  - Usage: API calls, tiered testing
  - 6 strategy details
  - Differences from Trip Learning

---

## 🚀 Quick Navigation

### I want to understand...

- **Overall learning system architecture** → Read [LEARNING_OVERVIEW.md](LEARNING_OVERVIEW.md)
- **How Trip Planning learns** → Read [TRIP_LEARNING.md](TRIP_LEARNING.md)
- **How Code Generation learns** → Read [CODE_LEARNING.md](CODE_LEARNING.md)
- **How to run tests** → Check "Quick Start" section in LEARNING_OVERVIEW.md
- **Where learning data is stored** → `data/trip_experiences.jsonl` and `data/codegen_experiences.jsonl`
- **How to view learning effects** → Frontend sidebar Learning Dashboard

### I want to...

- **Run Trip Learning tests**
  ```bash
  # Quick test
  python test_trip_learning_basic.py --quick

  # Full test
  python test_trip_learning_tiered.py

  # Visualization
  python test_trip_learning_visualization.py
  ```

- **Run Code Learning tests**
  ```bash
  # Quick test (10 questions)
  python test_code_learning_tiered.py --quick

  # Full test (100 questions)
  python test_code_learning_tiered.py

  # Test by difficulty
  python test_code_learning_tiered.py --simple
  python test_code_learning_tiered.py --medium
  python test_code_learning_tiered.py --complex
  ```

- **Clear learning memory (start fresh)**
  ```bash
  rm data/trip_experiences.jsonl
  rm data/codegen_experiences.jsonl
  ```

- **Analyze learning data**
  ```bash
  # View strategy distribution
  grep -o '"strategy":"[^"]*"' data/trip_experiences.jsonl | sort | uniq -c

  # View average reward
  grep -o '"reward":[0-9.]*' data/trip_experiences.jsonl | \
    awk -F: '{sum+=$2; n++} END {print sum/n}'
  ```

---

## 📊 Core Concepts Quick Reference

### Thompson Sampling (Multi-Armed Bandit)
- Automatically balances Exploration and Exploitation
- Each strategy maintains a Beta distribution (α, β)
- Update distribution based on success/failure
- Sample from distribution to select strategy with maximum value

### Nearest Neighbor Retrieval
- Calculate similarity between current and historical tasks
- If distance < threshold, reuse best historical strategy
- Otherwise use Thompson Sampling for exploration

### Multi-Dimensional Reward
- **Trip**: 40% budget + 30% quality + 30% reliability
- **Code**: 50% success + 25% efficiency + 15% quality + 10% speed

### Task Signature
- **Trip**: 7-dim numerical vector (budget, days, geography, etc.)
- **Code**: 7-dim mixed structure (language, type, complexity, etc.)

---

## 🎯 Learning System File Reference

| Function | Trip Learning | Code Learning |
|----------|---------------|---------------|
| **Core Code** | `backend/services/autoplan_learn/` | `backend/services/codegen_learn/` |
| **Strategy Definition** | `autoplan_learn/strategies.py` | `codegen_learn/strategies.py` |
| **Task Signature** | `autoplan_learn/signature.py` | `codegen_learn/signature.py` |
| **Memory Storage** | `autoplan_learn/memory.py` | `codegen_learn/memory.py` |
| **Reward Calculation** | `autoplan_learn/adapter.py` | `codegen_learn/adapter.py` |
| **Bandit** | `autoplan_learn/bandit.py` | `codegen_learn/bandit.py` |
| **Learning Data** | `data/trip_experiences.jsonl` | `data/codegen_experiences.jsonl` |
| **Basic Test** | `test_trip_learning_basic.py` | `test_code_learning_tiered.py --quick` |
| **Full Test** | `test_trip_learning_tiered.py` | `test_code_learning_tiered.py` |
| **Visualization** | `test_trip_learning_visualization.py` | `test_code_learning_visualization.py` |
| **Frontend Dashboard** | ✈️ 🎓 Trip Learning Dashboard | 💻 🎓 Code Learning Dashboard |

---

## 💡 Tips

1. **First Run**: System has no historical data, will randomly explore strategies
2. **After 50-100 iterations**: Trip Learning starts converging to optimal strategies
3. **Tiered Learning**: Code Learning requires accumulating experience at different difficulty levels
4. **Threshold Adjustment**: Code Learning uses more conservative threshold (0.65 vs 0.55) due to GPT instability
5. **Real-time Feedback**: Check learning progress in frontend sidebar in real-time

---

## 🔗 Related Resources

- Main project directory: `/Users/yilu/Downloads/ai_assessment_project/ai-assessment-deploy/`
- Test files: `test_*_learning*.py`
- Learning data: `data/*_experiences.jsonl`
- Visualization results: `eval/*.json`, `eval/*.png`

---

**Recommended Reading Order**:
1. 📖 [LEARNING_OVERVIEW.md](LEARNING_OVERVIEW.md) - Understand overall architecture
2. ✈️ [TRIP_LEARNING.md](TRIP_LEARNING.md) - Deep dive into Trip Learning
3. 💻 [CODE_LEARNING.md](CODE_LEARNING.md) - Deep dive into Code Learning
4. 🚀 Run tests and check frontend Dashboard

Happy exploring! 🎉
