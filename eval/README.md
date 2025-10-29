# 📊 eval/ - Test Results and Evaluation Data

This directory contains all test results and evaluation outputs from the learning system tests.

## 📁 File Types

### Trip Planning Learning Tests

**From `test_trip_learning_tiered.py`:**
- `enhanced_learning_YYYYMMDD_HHMMSS.json` - Complete test results with scenarios, results, and dashboard data
- `enhanced_learning_summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary report with learning objectives, category performance, and top tool orders

**From `test_trip_learning_visualization.py`:**
- `learning_experiment_YYYYMMDD_HHMMSS.json` - Experiment data with 100 iterations
- `learning_visualization_YYYYMMDD_HHMMSS.png` - Learning curve visualization charts
- `intermediate_results.jsonl` - Line-by-line intermediate results
- `learning_summary.json` - Overall learning summary statistics

### Code Generation Learning Tests

**From `test_code_learning_tiered.py`:**
- `code_learning_tiered_results_YYYYMMDD_HHMMSS.json` - Complete test results for all difficulty tiers (🟢🟡🔴)
- `code_learning_tiered_summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary with tier analysis, strategy distribution, and learning metrics

**From `test_code_learning_visualization.py`:**
- `learning_experiment_YYYYMMDD_HHMMSS.json` - Experiment data
- `learning_visualization_YYYYMMDD_HHMMSS.png` - Learning curve charts

---

## 📋 File Format Details

### JSON Result Files (*.json)

Complete structured data for programmatic analysis:

```json
{
  "test_num": 1,
  "difficulty": "simple",
  "language": "python",
  "task": "Write a function to check if a number is even",
  "passed": true,
  "retries": 0,
  "gen_time_ms": 9707,
  "total_time_s": 9.7,
  "strategy": "fast_pragmatic",
  "reward": 0.940,
  "learning_success": true,
  "breakdown": {
    "success": 1.000,
    "efficiency": 1.000,
    "quality": 1.000,
    "speed": 0.400
  }
}
```

### Summary Text Files (*_summary_*.txt)

Human-readable reports with formatted output:

```
================================================================================
💻 🎓 CODE GENERATION LEARNING - TIERED TEST SUMMARY
================================================================================

📊 Overall Statistics:
   Total Tests: 10
   Passed: 10 (100.0%)
   Failed: 0 (0.0%)

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
...
```

---

## 🔍 Finding Results

### Latest Results
```bash
# Code learning latest summary
ls -t eval/code_learning_tiered_summary_*.txt | head -1

# Trip learning latest summary
ls -t eval/enhanced_learning_summary_*.txt | head -1

# View latest code summary
cat $(ls -t eval/code_learning_tiered_summary_*.txt | head -1)
```

### Results by Date
```bash
# All results from today
ls eval/*_$(date +%Y%m%d)_*.txt

# All code results from a specific date
ls eval/code_learning_tiered_*_20251029_*.txt
```

### Compare Results
```bash
# Compare two test runs
diff eval/code_learning_tiered_summary_20251029_120000.txt \
     eval/code_learning_tiered_summary_20251029_130000.txt
```

---

## 📈 Using Results

### Share Results
Simply send the `*_summary_*.txt` file - it contains all the important metrics in a readable format.

### Analyze Trends
Use JSON files for programmatic analysis:

```python
import json
import glob

# Load all code learning results
results = []
for file in glob.glob('eval/code_learning_tiered_results_*.json'):
    with open(file) as f:
        results.append(json.load(f))

# Analyze reward trends
avg_rewards = [sum(r['reward'] for r in run) / len(run) for run in results]
print(f"Reward trend: {avg_rewards}")
```

### Archive Results
```bash
# Archive all results from a specific date
tar -czf results_20251029.tar.gz eval/*_20251029_*

# Clean old results (keep last 10 runs)
ls -t eval/code_learning_tiered_summary_*.txt | tail -n +11 | xargs rm
```

---

## 🎯 What to Look For

### Code Learning (`code_learning_tiered_summary_*.txt`)

**Key Metrics:**
- **Overall Pass Rate**: Should be >90% after learning
- **Tier Performance**:
  - 🟢 Simple: Should have 0-1 avg retries
  - 🟡 Medium: Should have 1-2 avg retries
  - 🔴 Complex: Should have 2-3 avg retries
- **Strategy Distribution**: Should match difficulty
  - Simple → `fast_pragmatic`, `balanced`
  - Medium → `detailed_planner`, `test_driven`
  - Complex → `high_quality`, `test_driven`
- **Learning Objectives**: Fast strategies for simple, quality strategies for complex

### Trip Learning (`enhanced_learning_summary_*.txt`)

**Key Metrics:**
- **Learning Objectives Progress**: Should approach targets over iterations
- **Category Performance**: All categories should have >80% success rate
- **Top Tool Orders**: Consistent high-reward patterns should emerge
- **Average Reward**: Should increase from ~0.65 to ~0.85

---

## 🧹 Maintenance

### Cleanup Old Results
```bash
# Keep only last 30 days
find eval/ -name "*.json" -mtime +30 -delete
find eval/ -name "*.txt" -mtime +30 -delete

# Keep only last 10 runs
ls -t eval/code_learning_tiered_* | tail -n +21 | xargs rm
```

### Backup Important Results
```bash
# Backup baseline results
mkdir -p eval/baselines
cp eval/code_learning_tiered_summary_20251029_100000.txt eval/baselines/baseline_v1.txt
```

---

## 📚 Related Documentation

- **Testing Guide**: [../TESTING.md](../TESTING.md)
- **Learning Overview**: [../docs/LEARNING_OVERVIEW.md](../docs/LEARNING_OVERVIEW.md)
- **Trip Learning**: [../docs/TRIP_LEARNING.md](../docs/TRIP_LEARNING.md)
- **Code Learning**: [../docs/CODE_LEARNING.md](../docs/CODE_LEARNING.md)

---

**Note**: All timestamps in filenames are in format `YYYYMMDD_HHMMSS` (e.g., `20251029_123456`)
