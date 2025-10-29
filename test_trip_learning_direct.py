#!/usr/bin/env python3
"""
Direct test of learning system without API server
"""
import sys
import os
import json
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.services.planning_agent import PlanningAgent
from backend.models.agent_schemas import PlanRequest, TripConstraints


async def test_learning():
    """Test learning system directly"""

    # Initialize agent with learning enabled
    agent = PlanningAgent(enable_learning=True)

    print("🧪 Testing learning system integration...\n")

    # Test case 1: Auckland trip
    request = PlanRequest(
        prompt="Plan a 3-day trip to Auckland with a budget of $1500 NZD",
        constraints=TripConstraints(
            budget=1500,
            currency="NZD",
            days=3,
            origin_city="Wellington",
            destination_city="Auckland"
        ),
        max_iterations=3
    )

    print(f"📝 Request: {request.prompt}")
    print(f"💰 Budget: {request.constraints.currency} {request.constraints.budget}")
    print(f"📅 Days: {request.constraints.days}\n")

    try:
        response = await agent.create_plan(request)

        print(f"✅ Plan created successfully!")
        print(f"   Iterations: {response.total_iterations}")
        print(f"   Planning time: {response.planning_time_ms:.1f}ms")
        print(f"   Tool calls: {len(response.tool_calls)}")
        print(f"   Tool errors: {response.tool_errors_count}")
        print(f"   Constraints satisfied: {response.constraints_satisfied}")
        print(f"   Total cost: {response.itinerary.currency} {response.itinerary.total_cost}")

        # Check learning result
        if response.learning:
            print(f"\n🎓 LEARNING RESULT:")
            print(f"   Success: {response.learning['success']}")
            print(f"   Reward: {response.learning['reward']:.3f}")
            print(f"   Budget score: {response.learning['breakdown']['budget_score']:.3f}")
            print(f"   Quality score: {response.learning['breakdown']['quality_score']:.3f}")
            print(f"   Reliability score: {response.learning['breakdown']['reliability_score']:.3f}")
        else:
            print(f"\n❌ NO LEARNING RESULT RETURNED!")

        # Check strategy
        if response.strategy_used:
            print(f"\n📊 STRATEGY USED:")
            print(f"   Tool order: {response.strategy_used.get('tool_order', 'N/A')}")
            print(f"   Model temp: {response.strategy_used.get('model_temp', 'N/A')}")
            print(f"   Attractions k: {response.strategy_used.get('attractions_k', 'N/A')}")
        else:
            print(f"\n❌ NO STRATEGY RETURNED!")

        return response

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(test_learning())

    if result and result.learning:
        print(f"\n\n✅ SUCCESS: Learning system is properly integrated!")
        print(f"   Reward accumulated: {result.learning['reward']:.3f}")
    else:
        print(f"\n\n❌ FAILED: Learning system not returning results!")
        sys.exit(1)
