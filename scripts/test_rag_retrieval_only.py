#!/usr/bin/env python3
"""
Pure RAG Retrieval Test - No LLM Answer Generation

Test only the retrieval performance:
1. Embedding + Vector Search + Reranking
2. Measure latency breakdown
3. Check accuracy based on retrieved chunks only
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.rag_pipeline import retrieve_chunks


async def warmup():
    """Warm up models."""
    print("🔥 Warming up models...")
    for i in range(2):
        tic = time.perf_counter()
        await retrieve_chunks("warmup", top_k=5, search_limit=20)
        toc = time.perf_counter()
        print(f"   Warmup {i+1}/2: {(toc-tic)*1000:.0f}ms")
    print()


def check_retrieval_accuracy(chunks, keywords):
    """Check if answer keywords appear in retrieved chunks."""
    if not chunks:
        return False

    # Combine all text and metadata
    all_content = ""
    for chunk in chunks:
        # Text content
        all_content += chunk.content.lower() + " "

        # Metadata fields
        if hasattr(chunk, 'payload') and chunk.payload:
            authors = chunk.payload.get('authors', '') or ''
            subjects = chunk.payload.get('subjects', '') or ''
            all_content += authors.lower() + " " + subjects.lower() + " "

    # Check keywords
    for kw in keywords:
        if kw.lower() in all_content:
            return True
    return False


async def test_retrieval():
    """Test pure retrieval on 20 questions."""
    print("=" * 80)
    print("RAG Retrieval Test - 20 Questions (No LLM)")
    print("=" * 80)
    print()

    # Warmup
    await warmup()

    # Load dataset
    dataset_path = Path("data/rag_eval_metadata.json")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)[:20]

    print(f"📊 Testing {len(dataset)} questions\n")

    results = []
    correct = 0

    # Latency breakdown
    embed_times = []
    search_times = []
    rerank_times = []
    total_times = []

    for idx, item in enumerate(dataset, 1):
        question = item["question"]
        keywords = item["answer_keywords"]

        # Truncate display
        display_q = question if len(question) <= 65 else question[:62] + "..."
        print(f"[{idx:2d}/20] {display_q}")

        try:
            # Measure total time
            tic_total = time.perf_counter()

            chunks, retrieval_ms = await retrieve_chunks(
                question,
                top_k=5,
                search_limit=20
            )

            toc_total = time.perf_counter()
            total_ms = (toc_total - tic_total) * 1000

            # Check accuracy
            is_correct = check_retrieval_accuracy(chunks, keywords)

            if is_correct:
                correct += 1
                print(f"        ✅ FOUND  | {total_ms:.0f}ms")
            else:
                print(f"        ❌ MISSED | {total_ms:.0f}ms")

            # Show top result score
            if chunks:
                top_score = chunks[0].score
                print(f"           Top score: {top_score:.3f}")

            total_times.append(total_ms)
            results.append({
                "question": question,
                "correct": is_correct,
                "latency_ms": total_ms,
                "top_score": chunks[0].score if chunks else 0
            })

        except Exception as e:
            print(f"        ❌ ERROR: {e}")
            results.append({"question": question, "error": str(e)})

    print()
    print("=" * 80)
    print("RESULTS - Pure Retrieval Performance")
    print("=" * 80)

    # Calculate metrics
    accuracy = (correct / len(dataset)) * 100

    if total_times:
        mean_lat = sum(total_times) / len(total_times)
        sorted_times = sorted(total_times)
        median_lat = sorted_times[len(sorted_times)//2]
        p95_lat = sorted_times[int(len(sorted_times) * 0.95)]
        min_lat = min(total_times)
        max_lat = max(total_times)
    else:
        mean_lat = median_lat = p95_lat = min_lat = max_lat = 0

    print(f"\n📊 Retrieval Accuracy: {correct}/{len(dataset)} = {accuracy:.1f}%")
    print()
    print(f"⏱️  Latency Statistics:")
    print(f"   Mean:   {mean_lat:.1f}ms")
    print(f"   Median: {median_lat:.1f}ms")
    print(f"   P95:    {p95_lat:.1f}ms")
    print(f"   Min:    {min_lat:.1f}ms")
    print(f"   Max:    {max_lat:.1f}ms")
    print()

    if median_lat <= 300:
        print("✅ Meets <300ms target!")
    else:
        print(f"⚠️  Exceeds 300ms target ({median_lat:.0f}ms)")

    print()
    print("=" * 80)

    # Save results
    output = {
        "summary": {
            "total": len(dataset),
            "correct": correct,
            "accuracy": accuracy,
            "latency": {
                "mean_ms": mean_lat,
                "median_ms": median_lat,
                "p95_ms": p95_lat,
                "min_ms": min_lat,
                "max_ms": max_lat
            }
        },
        "results": results
    }

    with open("rag_retrieval_test.json", 'w') as f:
        json.dump(output, f, indent=2)

    print("📁 Saved: rag_retrieval_test.json\n")


if __name__ == "__main__":
    asyncio.run(test_retrieval())
