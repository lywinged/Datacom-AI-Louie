#!/usr/bin/env python3
"""
Quick API readiness test before running the learning visualization
"""

import requests
import json
import sys

def test_api_health():
    """Test if the agent API is healthy"""
    print("1. Testing API health endpoint...")
    try:
        response = requests.get("http://localhost:8888/api/agent/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ API is healthy")
            return True
        else:
            print(f"   ❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")
        return False

def test_sample_request():
    """Test a simple trip planning request"""
    print("\n2. Testing sample trip planning request...")
    payload = {
        "prompt": "Plan a 3-day trip to Auckland with a budget of $1500 NZD",
        "constraints": {
            "city": "Auckland",
            "days": 3,
            "budget_nzd": 1500
        }
    }

    try:
        print("   Sending request (this may take 30-60 seconds)...")
        response = requests.post(
            "http://localhost:8888/api/agent/plan",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()

            # Check if response has expected structure
            has_plan = "plan" in data or "itinerary" in data
            has_learning = "learning" in data
            has_strategy = "strategy_used" in data

            print(f"   ✅ Request successful")
            print(f"      - Has plan: {has_plan}")
            print(f"      - Has learning data: {has_learning}")
            print(f"      - Has strategy info: {has_strategy}")

            if has_learning:
                learning = data.get("learning", {})
                reward = learning.get("reward", 0)
                success = learning.get("success", False)
                breakdown = learning.get("breakdown", {})

                print(f"      - Reward: {reward:.3f}")
                print(f"      - Success: {success}")
                if breakdown:
                    print(f"      - Budget score: {breakdown.get('budget_score', 0):.3f}")
                    print(f"      - Quality score: {breakdown.get('quality_score', 0):.3f}")
                    print(f"      - Reliability score: {breakdown.get('reliability_score', 0):.3f}")

            return True
        else:
            print(f"   ❌ Request failed with status {response.status_code}")
            print(f"      Response: {response.text[:200]}")
            return False

    except requests.Timeout:
        print("   ❌ Request timed out (>120s)")
        print("      The backend may be slow or stuck. Check docker logs.")
        return False
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return False

def test_learning_memory_path():
    """Check if learning memory directory exists"""
    print("\n3. Checking learning system setup...")
    from pathlib import Path

    # Check if the learning memory path exists (it will be created on first use)
    memory_dir = Path("/Users/yilu/Downloads/ai_assessment_project/data/autoplan_memory")

    if memory_dir.exists():
        print(f"   ✅ Learning memory directory exists")

        # Check for existing experience file
        exp_file = memory_dir / "experiences.jsonl"
        if exp_file.exists():
            with open(exp_file, "r") as f:
                lines = f.readlines()
            print(f"   📊 Found {len(lines)} existing experiences")
        else:
            print("   📝 No existing experiences (fresh start)")
    else:
        print(f"   📝 Learning memory will be created on first use")

    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n4. Checking Python dependencies...")

    missing = []

    try:
        import matplotlib
        print("   ✅ matplotlib installed")
    except ImportError:
        print("   ❌ matplotlib not installed")
        missing.append("matplotlib")

    try:
        import requests
        print("   ✅ requests installed")
    except ImportError:
        print("   ❌ requests not installed")
        missing.append("requests")

    if missing:
        print(f"\n   Install missing packages: pip install {' '.join(missing)}")
        return False

    return True

def main():
    print("=" * 60)
    print("🧪 Trip Agent Learning System - Readiness Test")
    print("=" * 60)

    results = []

    # Test 1: API Health
    results.append(test_api_health())

    # Test 2: Dependencies
    results.append(check_dependencies())

    # Test 3: Learning system
    results.append(test_learning_memory_path())

    # Test 4: Sample request (only if health check passed)
    if results[0]:
        results.append(test_sample_request())
    else:
        print("\n⚠️  Skipping sample request test (API health check failed)")
        print("   Please start the backend first:")
        print("   cd /Users/yilu/Downloads/ai_assessment_project")
        print("   docker-compose up -d backend")
        results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    test_names = [
        "API Health Check",
        "Python Dependencies",
        "Learning System Setup",
        "Sample Planning Request"
    ]

    for name, result in zip(test_names, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(results)

    if all_passed:
        print("\n🎉 All tests passed! Ready to run learning experiments.")
        print("\nNext steps:")
        print("  python test_learning_visualization.py -n 20   # Quick test (20 iterations)")
        print("  python test_learning_visualization.py         # Full experiment (100 iterations)")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running experiments.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
