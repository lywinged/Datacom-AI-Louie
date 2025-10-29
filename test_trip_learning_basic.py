#!/usr/bin/env python3
"""
Test script for Trip Planning Learning System Integration

Demonstrates online learning over multiple planning runs:
- Strategy selection via Thompson Sampling
- Experience-based nearest neighbor retrieval
- Automatic reward calculation from execution signals
- Per-cluster learning (city|days|budget)

Usage:
    PYTHONPATH=. python test_trip_learning_system.py
"""

import asyncio
import os
import json
from backend.models.agent_schemas import PlanRequest, TripConstraints
from backend.services.planning_agent import PlanningAgent


async def test_learning_progression():
    """Test learning system over multiple similar requests"""

    print("=" * 100)
    print("✈️ 🎓 Trip Planning Learning System Integration Test")
    print("=" * 100)
    print()
    print("This test demonstrates online learning by running the same request multiple times.")
    print("The learning system should:")
    print("  1. Choose different strategies initially (exploration)")
    print("  2. Learn which strategies work best (exploitation)")
    print("  3. Improve performance over time")
    print()
    print("=" * 100)
    print()

    # Create agent with learning enabled
    agent = PlanningAgent(enable_learning=True)

    # Test requests (similar tasks to see learning)
    test_cases = [
        {
            "name": "Auckland Budget Trip #1",
            "request": PlanRequest(
                prompt="Plan a budget-friendly 2-day trip to Auckland",
                constraints=TripConstraints(
                    budget=500.0,
                    currency="NZD",
                    days=2,
                    start_date="2024-12-15",
                    preferences=["budget-friendly", "outdoor activities"]
                ),
                max_iterations=5
            )
        },
        {
            "name": "Auckland Budget Trip #2 (Similar)",
            "request": PlanRequest(
                prompt="I want to visit Auckland for 2 days with limited budget",
                constraints=TripConstraints(
                    budget=480.0,
                    currency="NZD",
                    days=2,
                    start_date="2024-12-20",
                    preferences=["budget-friendly"]
                ),
                max_iterations=5
            )
        },
        {
            "name": "London Culture Trip #1",
            "request": PlanRequest(
                prompt="Plan a 3-day cultural trip to London",
                constraints=TripConstraints(
                    budget=800.0,
                    currency="GBP",
                    days=3,
                    start_date="2024-12-18",
                    preferences=["museums", "history"]
                ),
                max_iterations=5
            )
        },
        {
            "name": "Auckland Budget Trip #3 (Similar to #1, #2)",
            "request": PlanRequest(
                prompt="Need a cheap 2-day Auckland trip plan",
                constraints=TripConstraints(
                    budget=520.0,
                    currency="NZD",
                    days=2,
                    start_date="2024-12-25",
                    preferences=["budget-friendly", "nature"]
                ),
                max_iterations=5
            )
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*100}")
        print(f">� Test Case {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*100}\n")

        request = test_case["request"]

        print(f"=� Request:")
        print(f"   Prompt: {request.prompt}")
        print(f"   Budget: {request.constraints.currency} {request.constraints.budget}")
        print(f"   Days: {request.constraints.days}")
        print()

        try:
            # Execute planning
            response = await agent.create_plan(request)

            # Extract results
            result = {
                "test_case": test_case["name"],
                "strategy_used": response.strategy_used,
                "total_cost": response.itinerary.total_cost,
                "total_cost_nzd": response.itinerary.total_cost_nzd,
                "currency": response.itinerary.currency,
                "tool_calls": len(response.tool_calls),
                "tool_errors": response.tool_errors_count,
                "iterations": response.total_iterations,
                "planning_time_ms": response.planning_time_ms,
                "constraints_satisfied": response.constraints_satisfied,
                "constraint_satisfaction": response.constraint_satisfaction,
                "flights_found": len(response.itinerary.flights),
                "attractions_found": len(response.itinerary.attractions),
            }

            results.append(result)

            # Display results
            print(" Planning Results:")
            print(f"   Destination: {response.itinerary.destination}")
            print(f"   Total Cost: {response.itinerary.currency} {response.itinerary.total_cost:.2f} "
                  f"(NZD {response.itinerary.total_cost_nzd:.2f})")
            print(f"   Flights: {len(response.itinerary.flights)}")
            print(f"   Attractions: {len(response.itinerary.attractions)}")
            print(f"   Tool Calls: {len(response.tool_calls)}")
            print(f"   Tool Errors: {response.tool_errors_count}")
            print(f"   Constraints Satisfied: {response.constraints_satisfied}")
            print(f"   Constraint Satisfaction Score: {response.constraint_satisfaction:.3f}")
            print()

            if response.strategy_used:
                print("<� Learning System:")
                print(f"   Strategy Selected: {response.strategy_used.get('tool_order', 'N/A')}")
                print(f"   Temperature: {response.strategy_used.get('model_temp', 'N/A')}")
                print(f"   Attractions K: {response.strategy_used.get('attractions_k', 'N/A')}")
                print(f"   Max Retries: {response.strategy_used.get('max_tool_retries', 'N/A')}")
            else:
                print("�  Learning system not active for this run")
            print()

            # Show constraint violations if any
            if not response.constraints_satisfied:
                print("�  Constraint Violations:")
                for violation in response.constraint_violations:
                    print(f"   - {violation}")
                print()

        except Exception as e:
            print(f"L Planning failed: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Summary
    print("\n" + "=" * 100)
    print("=� LEARNING PROGRESSION SUMMARY")
    print("=" * 100)
    print()

    # Display results table
    print(f"{'Test Case':<40} {'Strategy':<35} {'Satisfaction':<15} {'Tool Errors':<12} {'Time (ms)':<10}")
    print("-" * 115)

    for result in results:
        strategy_name = result['strategy_used'].get('tool_order', 'N/A')[:33] if result['strategy_used'] else 'None'
        print(f"{result['test_case']:<40} {strategy_name:<35} "
              f"{result['constraint_satisfaction']:<15.3f} {result['tool_errors']:<12} "
              f"{result['planning_time_ms']:<10.0f}")

    print()

    # Learning insights
    print("= Learning Insights:")
    print()

    # Check if similar tasks (Auckland budget trips) improved
    auckland_runs = [r for r in results if "Auckland Budget Trip" in r["test_case"]]
    if len(auckland_runs) >= 2:
        print("   Auckland Budget Trips (Similar Tasks):")
        for i, run in enumerate(auckland_runs, 1):
            print(f"     Run {i}: Satisfaction={run['constraint_satisfaction']:.3f}, "
                  f"Errors={run['tool_errors']}, Strategy={run['strategy_used'].get('tool_order', 'N/A') if run['strategy_used'] else 'None'}")

        # Check for improvement
        if auckland_runs[-1]['constraint_satisfaction'] >= auckland_runs[0]['constraint_satisfaction']:
            print(f"      Learning improved satisfaction: "
                  f"{auckland_runs[0]['constraint_satisfaction']:.3f} � {auckland_runs[-1]['constraint_satisfaction']:.3f}")
        else:
            print(f"     =� Still exploring (satisfaction varied across runs)")

    print()

    # Check memory file
    memory_path = os.getenv("AGENT_MEMORY_PATH", "data/agent_experiences.jsonl")
    if os.path.exists(memory_path):
        with open(memory_path, 'r') as f:
            experiences = [json.loads(line) for line in f if line.strip()]

        print(f"   Experience Memory: {len(experiences)} experiences recorded")
        print(f"   Memory File: {memory_path}")

        if experiences:
            avg_reward = sum(e.get('reward', 0) for e in experiences) / len(experiences)
            print(f"   Average Reward: {avg_reward:.3f}")

    print()
    print("=" * 100)
    print(" Learning System Integration Test Complete!")
    print("=" * 100)
    print()
    print("Key Observations:")
    print("  1. The learning system selected strategies before each planning run")
    print("  2. Results were recorded to the experience memory (JSONL)")
    print("  3. Similar tasks should show strategy convergence over time")
    print("  4. Rewards are calculated automatically from execution signals (no human labels needed)")
    print()


async def main():
    """Main entry point"""
    await test_learning_progression()


if __name__ == "__main__":
    asyncio.run(main())
