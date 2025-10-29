#!/usr/bin/env python3
"""
💻 🎓 Code Generation Learning System - 100 Tiered Test Cases

This comprehensive test suite evaluates the code learning system across difficulty tiers:

🟢 SIMPLE (30 tests):
   - Basic operations, single-step logic
   - Expected: 0-1 retries
   - Strategies: fast_pragmatic, balanced
   - Example: "check if number is even", "sum two numbers"

🟡 MEDIUM (40 tests):
   - Multi-step logic, standard algorithms
   - Expected: 1-2 retries
   - Strategies: detailed_planner, test_driven, balanced
   - Example: "implement binary search", "parse CSV file"

🔴 COMPLEX (30 tests):
   - Advanced algorithms, intricate logic
   - Expected: 2-3 retries
   - Strategies: high_quality, test_driven
   - Example: "implement quicksort with DP", "solve N-Queens"

The system should learn to:
1. Select appropriate strategies for different difficulty levels
2. Adapt strategy based on historical success
3. Improve efficiency over time (fewer retries for similar tasks)
4. Achieve high success rates across all tiers

Usage:
    python test_code_learning_100.py [--quick] [--simple] [--medium] [--complex]

Options:
    --quick     Run only 10 tests (3 simple, 4 medium, 3 complex)
    --simple    Run only simple tests
    --medium    Run only medium tests
    --complex   Run only complex tests
"""

import sys
import time
import requests
import json
from datetime import datetime
from typing import List, Dict, Any
import statistics

BACKEND_URL = "http://localhost:8888"

# ============================================================================
# 🟢 SIMPLE TESTS (30) - Expected 0-1 retries
# ============================================================================
SIMPLE_TESTS = [
    # Basic arithmetic and logic (10)
    {"language": "python", "task": "Write a function to check if a number is even", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to sum two numbers", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to find the maximum of three numbers", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to check if a number is positive, negative, or zero", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to calculate the area of a rectangle", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to convert Celsius to Fahrenheit", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to check if a year is a leap year", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to count the number of digits in an integer", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to reverse a string", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to check if a string is empty or whitespace", "difficulty": "simple"},

    # Simple list operations (10)
    {"language": "python", "task": "Write a function to find the sum of all elements in a list", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to find the average of numbers in a list", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to find the minimum element in a list", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to count even numbers in a list", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to remove all None values from a list", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to check if a list is empty", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to get the first element of a list safely", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to concatenate two lists", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to check if an element exists in a list", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to create a list of squares from 1 to n", "difficulty": "simple"},

    # Simple string operations (10)
    {"language": "python", "task": "Write a function to count vowels in a string", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to capitalize the first letter of a string", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to check if a string starts with a given prefix", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to remove spaces from a string", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to repeat a string n times", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to check if a string contains only digits", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to convert a string to uppercase", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to count words in a sentence", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to replace spaces with underscores", "difficulty": "simple"},
    {"language": "python", "task": "Write a function to get the length of a string", "difficulty": "simple"},
]

# ============================================================================
# 🟡 MEDIUM TESTS (40) - Expected 1-2 retries
# ============================================================================
MEDIUM_TESTS = [
    # Algorithms (15)
    {"language": "python", "task": "Implement binary search for a sorted array", "difficulty": "medium"},
    {"language": "python", "task": "Write a function to check if a number is prime", "difficulty": "medium"},
    {"language": "python", "task": "Implement bubble sort algorithm", "difficulty": "medium"},
    {"language": "python", "task": "Calculate Fibonacci sequence up to n iteratively", "difficulty": "medium"},
    {"language": "python", "task": "Find the factorial of a number using recursion", "difficulty": "medium"},
    {"language": "python", "task": "Check if two strings are anagrams", "difficulty": "medium"},
    {"language": "python", "task": "Find all prime numbers up to N using Sieve of Eratosthenes", "difficulty": "medium"},
    {"language": "python", "task": "Implement linear search with early termination", "difficulty": "medium"},
    {"language": "python", "task": "Find the GCD of two numbers using Euclidean algorithm", "difficulty": "medium"},
    {"language": "python", "task": "Check if a string is a palindrome ignoring case and spaces", "difficulty": "medium"},
    {"language": "python", "task": "Find the second largest element in a list", "difficulty": "medium"},
    {"language": "python", "task": "Remove duplicates from a list while preserving order", "difficulty": "medium"},
    {"language": "python", "task": "Rotate an array to the right by k positions", "difficulty": "medium"},
    {"language": "python", "task": "Find the intersection of two lists", "difficulty": "medium"},
    {"language": "python", "task": "Convert decimal number to binary string", "difficulty": "medium"},

    # Data processing (15)
    {"language": "python", "task": "Parse CSV string and calculate average of numeric column", "difficulty": "medium"},
    {"language": "python", "task": "Read JSON string and extract nested field values", "difficulty": "medium"},
    {"language": "python", "task": "Filter list of dictionaries by multiple conditions", "difficulty": "medium"},
    {"language": "python", "task": "Group list of dictionaries by a key and count occurrences", "difficulty": "medium"},
    {"language": "python", "task": "Merge two dictionaries with priority to second dict values", "difficulty": "medium"},
    {"language": "python", "task": "Flatten a nested dictionary to dot-notation keys", "difficulty": "medium"},
    {"language": "python", "task": "Parse log lines and extract timestamps and error levels", "difficulty": "medium"},
    {"language": "python", "task": "Validate email addresses using regex", "difficulty": "medium"},
    {"language": "python", "task": "Parse URL and extract all query parameters into dict", "difficulty": "medium"},
    {"language": "python", "task": "Convert UNIX timestamp to human-readable datetime", "difficulty": "medium"},
    {"language": "python", "task": "Remove all HTML tags from a string using regex", "difficulty": "medium"},
    {"language": "python", "task": "Detect duplicate records in list of dicts by key", "difficulty": "medium"},
    {"language": "python", "task": "Clean and normalize text: lowercase, remove punctuation", "difficulty": "medium"},
    {"language": "python", "task": "Split a large string into chunks of max size", "difficulty": "medium"},
    {"language": "python", "task": "Convert snake_case to camelCase", "difficulty": "medium"},

    # OOP & Patterns (10)
    {"language": "python", "task": "Create a simple bank account class with deposit and withdraw", "difficulty": "medium"},
    {"language": "python", "task": "Implement a stack class with push, pop, and peek methods", "difficulty": "medium"},
    {"language": "python", "task": "Create a queue class with enqueue and dequeue methods", "difficulty": "medium"},
    {"language": "python", "task": "Implement a student class with name, grades, and average calculation", "difficulty": "medium"},
    {"language": "python", "task": "Create a shopping cart class with add item and total price methods", "difficulty": "medium"},
    {"language": "python", "task": "Implement a counter class that tracks increment and decrement", "difficulty": "medium"},
    {"language": "python", "task": "Create a temperature class that converts between C and F", "difficulty": "medium"},
    {"language": "python", "task": "Implement a simple point class with distance calculation", "difficulty": "medium"},
    {"language": "python", "task": "Create a rectangle class with area and perimeter methods", "difficulty": "medium"},
    {"language": "python", "task": "Implement a timer class that tracks elapsed time", "difficulty": "medium"},
]

# ============================================================================
# 🔴 COMPLEX TESTS (30) - Expected 2-3 retries
# ============================================================================
COMPLEX_TESTS = [
    # Advanced algorithms (15)
    {"language": "python", "task": "Implement quicksort algorithm with in-place partitioning", "difficulty": "complex"},
    {"language": "python", "task": "Implement merge sort algorithm", "difficulty": "complex"},
    {"language": "python", "task": "Solve the 0/1 knapsack problem using dynamic programming", "difficulty": "complex"},
    {"language": "python", "task": "Find longest common subsequence of two strings using DP", "difficulty": "complex"},
    {"language": "python", "task": "Implement a binary search tree with insert, search, and delete", "difficulty": "complex"},
    {"language": "python", "task": "Implement depth-first search for a graph represented as adjacency list", "difficulty": "complex"},
    {"language": "python", "task": "Implement breadth-first search for a graph", "difficulty": "complex"},
    {"language": "python", "task": "Find all permutations of a string using backtracking", "difficulty": "complex"},
    {"language": "python", "task": "Implement heap sort with max heap", "difficulty": "complex"},
    {"language": "python", "task": "Solve the coin change problem: minimum coins for amount", "difficulty": "complex"},
    {"language": "python", "task": "Implement Dijkstra's shortest path algorithm", "difficulty": "complex"},
    {"language": "python", "task": "Find the longest palindromic substring using DP", "difficulty": "complex"},
    {"language": "python", "task": "Implement an LRU cache with get and put in O(1)", "difficulty": "complex"},
    {"language": "python", "task": "Solve the N-Queens problem using backtracking", "difficulty": "complex"},
    {"language": "python", "task": "Implement Kruskal's minimum spanning tree algorithm", "difficulty": "complex"},

    # Advanced data structures & OOP (10)
    {"language": "python", "task": "Implement a linked list class with insert, delete, search, and reverse", "difficulty": "complex"},
    {"language": "python", "task": "Create a queue using two stacks", "difficulty": "complex"},
    {"language": "python", "task": "Implement a singleton pattern class with thread safety", "difficulty": "complex"},
    {"language": "python", "task": "Create a factory pattern for different shape objects", "difficulty": "complex"},
    {"language": "python", "task": "Implement a decorator class that logs method calls with arguments", "difficulty": "complex"},
    {"language": "python", "task": "Create an observable/observer pattern implementation", "difficulty": "complex"},
    {"language": "python", "task": "Implement a trie data structure for word storage and prefix search", "difficulty": "complex"},
    {"language": "python", "task": "Create a graph class with add edge, DFS, and BFS methods", "difficulty": "complex"},
    {"language": "python", "task": "Implement a priority queue using a heap", "difficulty": "complex"},
    {"language": "python", "task": "Create a hash table class with collision handling", "difficulty": "complex"},

    # Multi-language complex (5)
    {"language": "javascript", "task": "Implement promise-based retry logic with exponential backoff", "difficulty": "complex"},
    {"language": "javascript", "task": "Create a debounce function with leading and trailing options", "difficulty": "complex"},
    {"language": "javascript", "task": "Implement deep clone for nested objects with circular reference handling", "difficulty": "complex"},
    {"language": "rust", "task": "Create a generic stack data structure with push, pop, and peek", "difficulty": "complex"},
    {"language": "rust", "task": "Implement a simple linked list with insert and search", "difficulty": "complex"},
]

# Combine all tests
ALL_TESTS = (
    [(t, "🟢") for t in SIMPLE_TESTS] +
    [(t, "🟡") for t in MEDIUM_TESTS] +
    [(t, "🔴") for t in COMPLEX_TESTS]
)


def run_single_test(test_case: Dict[str, Any], tier_icon: str, test_num: int, total: int) -> Dict[str, Any]:
    """Run a single code generation test"""

    language = test_case["language"]
    task = test_case["task"]
    difficulty = test_case["difficulty"]

    print(f"\n{'='*80}")
    print(f"{tier_icon} Test {test_num}/{total}: {difficulty.upper()}")
    print(f"Language: {language}")
    print(f"Task: {task[:70]}...")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/code/generate",
            json={
                "task": task,
                "language": language,
                "max_retries": 3,
                "include_samples": False,
            },
            timeout=120
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            learning = result.get("learning", {})

            test_passed = result.get("test_passed", False)
            retries = result.get("total_retries", 0)
            gen_time = result.get("generation_time_ms", 0)

            # Extract learning data
            strategy = learning.get("strategy", "N/A")
            reward = learning.get("reward", 0.0)
            success = learning.get("success", False)
            breakdown = learning.get("breakdown", {})

            # Display results
            status_icon = "✅" if test_passed else "❌"
            print(f"\n{status_icon} Test Status: {'PASSED' if test_passed else 'FAILED'}")
            print(f"   Retries: {retries}")
            print(f"   Generation Time: {gen_time:.0f}ms")
            print(f"   Total Time: {elapsed:.1f}s")

            if learning:
                print(f"\n🎓 Learning:")
                print(f"   Strategy: {strategy}")
                print(f"   Reward: {reward:.3f}")
                print(f"   Success: {'✅' if success else '❌'}")
                if breakdown:
                    print(f"   Breakdown:")
                    print(f"     Success:    {breakdown.get('success', 0):.3f}")
                    print(f"     Efficiency: {breakdown.get('efficiency', 0):.3f}")
                    print(f"     Quality:    {breakdown.get('quality', 0):.3f}")
                    print(f"     Speed:      {breakdown.get('speed', 0):.3f}")

            return {
                "test_num": test_num,
                "difficulty": difficulty,
                "tier_icon": tier_icon,
                "language": language,
                "task": task,
                "passed": test_passed,
                "retries": retries,
                "gen_time_ms": gen_time,
                "total_time_s": elapsed,
                "strategy": strategy,
                "reward": reward,
                "learning_success": success,
                "breakdown": breakdown,
                "error": None
            }
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   {response.text[:200]}")
            return {
                "test_num": test_num,
                "difficulty": difficulty,
                "tier_icon": tier_icon,
                "language": language,
                "task": task,
                "passed": False,
                "error": f"HTTP {response.status_code}",
                "total_time_s": elapsed
            }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Exception: {e}")
        return {
            "test_num": test_num,
            "difficulty": difficulty,
            "tier_icon": tier_icon,
            "language": language,
            "task": task,
            "passed": False,
            "error": str(e),
            "total_time_s": elapsed
        }


def print_summary(results: List[Dict[str, Any]]) -> str:
    """Print comprehensive test summary with tier analysis and return summary text"""

    summary_lines = []

    def plog(text):
        """Print and log to summary"""
        print(text)
        summary_lines.append(text)

    plog("\n" + "="*80)
    plog("💻 🎓 CODE GENERATION LEARNING - TIERED TEST SUMMARY")
    plog("="*80)

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    failed = total - passed

    # Overall statistics
    plog(f"\n📊 Overall Statistics:")
    plog(f"   Total Tests: {total}")
    plog(f"   Passed: {passed} ({passed/total*100:.1f}%)")
    plog(f"   Failed: {failed} ({failed/total*100:.1f}%)")

    # Tier breakdown
    plog(f"\n🎯 Results by Difficulty Tier:")
    for difficulty in ["simple", "medium", "complex"]:
        tier_results = [r for r in results if r.get("difficulty") == difficulty]
        if tier_results:
            tier_icon = "🟢" if difficulty == "simple" else ("🟡" if difficulty == "medium" else "🔴")
            tier_passed = sum(1 for r in tier_results if r.get("passed"))
            tier_total = len(tier_results)
            tier_rate = tier_passed / tier_total * 100

            # Average retries for this tier
            tier_retries = [r.get("retries", 0) for r in tier_results if r.get("retries") is not None]
            avg_retries = statistics.mean(tier_retries) if tier_retries else 0

            # Average reward for this tier
            tier_rewards = [r.get("reward", 0) for r in tier_results if r.get("reward") is not None]
            avg_reward = statistics.mean(tier_rewards) if tier_rewards else 0

            plog(f"\n   {tier_icon} {difficulty.upper()}: {tier_passed}/{tier_total} ({tier_rate:.1f}%)")
            plog(f"      Avg Retries: {avg_retries:.2f}")
            plog(f"      Avg Reward: {avg_reward:.3f}")

    # Time statistics
    total_time = sum(r.get("total_time_s", 0) for r in results)
    avg_time = total_time / total if total > 0 else 0
    plog(f"\n⏱️  Time Statistics:")
    plog(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    plog(f"   Average Time per Test: {avg_time:.1f}s")

    # Learning statistics
    rewards = [r.get("reward", 0) for r in results if r.get("reward") is not None]
    if rewards:
        avg_reward = statistics.mean(rewards)
        median_reward = statistics.median(rewards)
        plog(f"\n🎓 Learning Statistics:")
        plog(f"   Average Reward: {avg_reward:.3f}")
        plog(f"   Median Reward: {median_reward:.3f}")
        plog(f"   Best Reward: {max(rewards):.3f}")
        plog(f"   Worst Reward: {min(rewards):.3f}")

    # Retry statistics
    retries_data = [r.get("retries", 0) for r in results if r.get("retries") is not None]
    if retries_data:
        avg_retries = statistics.mean(retries_data)
        zero_retry = sum(1 for r in retries_data if r == 0)
        plog(f"\n⚡ Efficiency Statistics:")
        plog(f"   Average Retries: {avg_retries:.2f}")
        plog(f"   First-Try Success: {zero_retry}/{len(retries_data)} ({zero_retry/len(retries_data)*100:.1f}%)")

    # Strategy distribution by tier
    plog(f"\n🧠 Strategy Distribution by Tier:")
    for difficulty in ["simple", "medium", "complex"]:
        tier_results = [r for r in results if r.get("difficulty") == difficulty]
        if tier_results:
            tier_icon = "🟢" if difficulty == "simple" else ("🟡" if difficulty == "medium" else "🔴")
            tier_strategies = {}
            for r in tier_results:
                strat = r.get("strategy", "N/A")
                if strat != "N/A":
                    tier_strategies[strat] = tier_strategies.get(strat, 0) + 1

            if tier_strategies:
                plog(f"\n   {tier_icon} {difficulty.upper()}:")
                for strategy, count in sorted(tier_strategies.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / len(tier_results) * 100
                    plog(f"      {strategy}: {count} ({percentage:.1f}%)")

    # Overall strategy distribution
    strategies = {}
    for r in results:
        strat = r.get("strategy", "N/A")
        if strat != "N/A":
            strategies[strat] = strategies.get(strat, 0) + 1

    if strategies:
        plog(f"\n🎯 Overall Strategy Distribution:")
        for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(results) * 100
            plog(f"   {strategy}: {count} times ({percentage:.1f}%)")

    # Language breakdown
    plog(f"\n🔤 Results by Language:")
    languages = {}
    for r in results:
        lang = r.get("language", "unknown")
        if lang not in languages:
            languages[lang] = {"total": 0, "passed": 0}
        languages[lang]["total"] += 1
        if r.get("passed"):
            languages[lang]["passed"] += 1

    for language, stats in sorted(languages.items()):
        rate = stats["passed"] / stats["total"] * 100
        plog(f"   {language}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    # Errors
    errors = [r for r in results if r.get("error")]
    if errors:
        plog(f"\n❌ Errors ({len(errors)}):")
        for err in errors[:5]:  # Show first 5 errors
            plog(f"   {err['tier_icon']} Test {err['test_num']} ({err['difficulty']}): {err['error'][:60]}")
        if len(errors) > 5:
            plog(f"   ... and {len(errors) - 5} more errors")

    # Learning objectives
    plog(f"\n🎯 Learning Objectives Check:")

    # Check if system selects appropriate strategies for difficulty
    simple_results = [r for r in results if r.get("difficulty") == "simple"]
    complex_results = [r for r in results if r.get("difficulty") == "complex"]

    if simple_results and complex_results:
        simple_fast = sum(1 for r in simple_results if r.get("strategy") in ["fast_pragmatic", "balanced"])
        complex_quality = sum(1 for r in complex_results if r.get("strategy") in ["high_quality", "test_driven", "detailed_planner"])

        simple_rate = simple_fast / len(simple_results) * 100 if simple_results else 0
        complex_rate = complex_quality / len(complex_results) * 100 if complex_results else 0

        plog(f"   ✓ Fast strategies for simple tasks: {simple_fast}/{len(simple_results)} ({simple_rate:.1f}%)")
        plog(f"   ✓ Quality strategies for complex tasks: {complex_quality}/{len(complex_results)} ({complex_rate:.1f}%)")

    plog("\n" + "="*80)
    plog("🎉 Tiered test suite completed!")
    plog("="*80)

    return "\n".join(summary_lines)


def main():
    """Main test runner"""

    # Parse command line arguments
    quick_mode = "--quick" in sys.argv
    simple_only = "--simple" in sys.argv
    medium_only = "--medium" in sys.argv
    complex_only = "--complex" in sys.argv

    # Select tests based on arguments
    if quick_mode:
        # 3 simple, 4 medium, 3 complex
        test_cases = (
            [(t, "🟢") for t in SIMPLE_TESTS[:3]] +
            [(t, "🟡") for t in MEDIUM_TESTS[:4]] +
            [(t, "🔴") for t in COMPLEX_TESTS[:3]]
        )
        print("⚡ Quick mode: Running 10 tests (3 simple, 4 medium, 3 complex)")
    elif simple_only:
        test_cases = [(t, "🟢") for t in SIMPLE_TESTS]
        print("🟢 Running SIMPLE tests only")
    elif medium_only:
        test_cases = [(t, "🟡") for t in MEDIUM_TESTS]
        print("🟡 Running MEDIUM tests only")
    elif complex_only:
        test_cases = [(t, "🔴") for t in COMPLEX_TESTS]
        print("🔴 Running COMPLEX tests only")
    else:
        test_cases = ALL_TESTS

    print("="*80)
    print("💻 🎓 CODE GENERATION LEARNING SYSTEM - TIERED TEST SUITE")
    print("="*80)
    print(f"\nTest distribution:")
    print(f"   🟢 Simple:  {len(SIMPLE_TESTS)} tests (expected 0-1 retries)")
    print(f"   🟡 Medium:  {len(MEDIUM_TESTS)} tests (expected 1-2 retries)")
    print(f"   🔴 Complex: {len(COMPLEX_TESTS)} tests (expected 2-3 retries)")
    print(f"\nTotal test cases: {len(test_cases)}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPress Ctrl+C to abort\n")

    # Check backend health
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
        else:
            print("⚠️  Backend returned non-200 status")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("Please ensure the backend is running!")
        return

    # Run tests
    results = []
    try:
        for i, (test_case, tier_icon) in enumerate(test_cases, 1):
            result = run_single_test(test_case, tier_icon, i, len(test_cases))
            results.append(result)

            # Small delay between tests
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite aborted by user!")
        print(f"Completed {len(results)}/{len(test_cases)} tests")

    # Print summary
    if results:
        summary_text = print_summary(results)

        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"eval/code_learning_tiered_results_{timestamp}.json"
        summary_file = f"eval/code_learning_tiered_summary_{timestamp}.txt"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        with open(summary_file, 'w') as f:
            f.write(summary_text)

        print(f"\n💾 Results saved to: {output_file}")
        print(f"💾 Summary saved to: {summary_file}")
    else:
        print("\n❌ No tests completed")


if __name__ == "__main__":
    main()
