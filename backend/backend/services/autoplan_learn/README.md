
# AutoPlan Learn (Bandit + Memory for Task 3.3)

Plug-and-play module that lets your Autonomous Planning Agent **get better over time** without human labels.

## Files
- `signature.py` — builds a task signature and hybrid similarity
- `bandit.py` — Thompson Sampling bandit over strategy arms
- `memory.py` — JSONL experience store + nearest-neighbor retrieval
- `strategies.py` — default strategy candidates for planning agent
- `adapter.py` — 2 functions: `choose()` and `record()`

## How to integrate (pseudo)
```python
from autoplan_learn.adapter import AutoPlanAdapter

learner = AutoPlanAdapter(memory_path="/data/agent_experiences.jsonl")

# before planning:
ctx = learner.choose(
    prompt=user_prompt,
    constraints={"city":"Auckland","days":2,"budget_nzd":500,"date_range":"2025-10-25..2025-10-26"},
    tools_used=["flights_api","weather_api","attractions_api"]
)
sig = ctx["signature"]; strategy = ctx["strategy"]  # use in your agent to set temp/order/k/etc.

# ...run your planner using 'strategy'...
# compute proxies after plan:
plan_cost_nzd = itinerary.aggregates.total_cost_nzd
tool_errors = run_ctx.tool_errors_count
satisfaction_proxy = itinerary.aggregates.constraint_satisfaction   # 0..1

# after planning:
learner.record(sig, strategy, plan_cost_nzd, budget_nzd=500, tool_errors=tool_errors, satisfaction_proxy=satisfaction_proxy)
```

## Why this works without labels
- **Executable signals** (budget respected, tool failures) and **proxy scores** (constraint satisfaction) become a numeric reward.
- A **bandit** learns, per task cluster (city/days/budget), which strategy works best.
- A **nearest-neighbor memory** reuses wins from similar past tasks immediately.
```
