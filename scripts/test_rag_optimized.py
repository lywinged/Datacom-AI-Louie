#!/usr/bin/env python3
"""
Optimized RAG Retrieval Test - Fetch Only Necessary Payload Fields

Key optimization: Use with_payload parameter to only fetch needed fields
instead of fetching the entire payload (which now includes large metadata)
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.qdrant_client import get_qdrant_client
from backend.services.embedding_service import get_embedding_model
from backend.config.settings import settings

COLLECTION_NAME = settings.QDRANT_COLLECTION


async def optimized_retrieve(question: str, top_k: int = 5) -> tuple[List, float]:
    """
    Optimized retrieval: only fetch 'text' field, skip metadata during search.

    This avoids reading large 'authors', 'subjects' fields from disk.
    """
    client = get_qdrant_client()
    embed_model = get_embedding_model()

    # 1. Embed query
    tic = time.perf_counter()
    query_embedding = await asyncio.to_thread(
        embed_model.encode, [question]
    )
    query_embedding = query_embedding[0].tolist()
    embed_ms = (time.perf_counter() - tic) * 1000

    # 2. Vector search - ONLY fetch 'text' field
    tic = time.perf_counter()
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=20,
        with_payload=["text", "source"],  # ✅ Only fetch what we need!
        # with_payload=True would fetch ALL fields including large metadata
    )
    search_ms = (time.perf_counter() - tic) * 1000

    # 3. Rerank
    tic = time.perf_counter()

    # Prepare candidates
    candidates = []
    for point in search_results:
        payload = point.payload or {}
        text = payload.get("text", "")
        source = payload.get("source", "")

        if text:
            candidates.append({
                "text": text,
                "source": source,
                "score": point.score
            })

    # Rerank
    if candidates and len(candidates) > 0:
        texts = [c["text"] for c in candidates]

        # Use reranker
        from backend.services.embedding_service import get_reranker
        reranker = get_reranker()

        scores = await asyncio.to_thread(
            reranker.compute_score,
            [[question, text] for text in texts]
        )

        # Combine
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        # Sort by rerank score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    rerank_ms = (time.perf_counter() - tic) * 1000

    total_ms = embed_ms + search_ms + rerank_ms

    # Return top-k
    top_candidates = candidates[:top_k]

    return top_candidates, total_ms, {
        "embed_ms": embed_ms,
        "search_ms": search_ms,
        "rerank_ms": rerank_ms
    }


async def warmup():
    """Warm up models."""
    print("🔥 Warming up models...")
    for i in range(2):
        tic = time.perf_counter()
        await optimized_retrieve("warmup test")
        toc = time.perf_counter()
        print(f"   Warmup {i+1}/2: {(toc-tic)*1000:.0f}ms")
    print()


def check_accuracy(candidates, keywords):
    """Check if keywords found in retrieved text."""
    if not candidates:
        return False

    all_text = " ".join(c["text"].lower() for c in candidates)

    for kw in keywords:
        if kw.lower() in all_text:
            return True
    return False


async def test_optimized():
    """Test optimized retrieval."""
    print("=" * 80)
    print("Optimized RAG Test - Only Fetch Necessary Payload Fields")
    print("=" * 80)
    print()

    await warmup()

    # Load dataset
    dataset_path = Path("data/rag_eval_metadata.json")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)[:20]

    print(f"📊 Testing {len(dataset)} questions")
    print(f"   Strategy: Fetch only 'text' + 'source' fields (skip metadata)\n")

    results = []
    correct = 0
    latencies = []

    # Breakdown
    embed_times = []
    search_times = []
    rerank_times = []

    for idx, item in enumerate(dataset, 1):
        question = item["question"]
        keywords = item["answer_keywords"]

        display_q = question if len(question) <= 65 else question[:62] + "..."
        print(f"[{idx:2d}/20] {display_q}")

        try:
            candidates, total_ms, breakdown = await optimized_retrieve(question, top_k=5)

            is_correct = check_accuracy(candidates, keywords)

            if is_correct:
                correct += 1
                print(f"        ✅ FOUND  | {total_ms:.0f}ms")
            else:
                print(f"        ❌ MISSED | {total_ms:.0f}ms")

            print(f"           Breakdown: embed={breakdown['embed_ms']:.0f}ms, "
                  f"search={breakdown['search_ms']:.0f}ms, "
                  f"rerank={breakdown['rerank_ms']:.0f}ms")

            latencies.append(total_ms)
            embed_times.append(breakdown['embed_ms'])
            search_times.append(breakdown['search_ms'])
            rerank_times.append(breakdown['rerank_ms'])

            results.append({
                "question": question,
                "correct": is_correct,
                "latency_ms": total_ms,
                "breakdown": breakdown
            })

        except Exception as e:
            print(f"        ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 80)
    print("RESULTS - Optimized Retrieval")
    print("=" * 80)

    accuracy = (correct / len(dataset)) * 100

    if latencies:
        mean_lat = sum(latencies) / len(latencies)
        median_lat = sorted(latencies)[len(latencies)//2]
    else:
        mean_lat = median_lat = 0

    print(f"\n📊 Accuracy: {correct}/{len(dataset)} = {accuracy:.1f}%")
    print(f"⏱️  Latency:  Mean={mean_lat:.0f}ms, Median={median_lat:.0f}ms")

    if median_lat <= 300:
        print(f"✅ Meets <300ms target!")
    else:
        print(f"⚠️  Exceeds 300ms ({median_lat:.0f}ms)")

    # Breakdown averages
    if embed_times:
        print(f"\n📊 Latency Breakdown (Average):")
        print(f"   Embedding: {sum(embed_times)/len(embed_times):.0f}ms")
        print(f"   Search:    {sum(search_times)/len(search_times):.0f}ms")
        print(f"   Rerank:    {sum(rerank_times)/len(rerank_times):.0f}ms")

    print("\n" + "=" * 80)

    # Save
    with open("rag_optimized_test.json", 'w') as f:
        json.dump({
            "summary": {
                "total": len(dataset),
                "correct": correct,
                "accuracy": accuracy,
                "mean_latency_ms": mean_lat,
                "median_latency_ms": median_lat
            },
            "results": results
        }, f, indent=2)

    print("📁 Saved: rag_optimized_test.json\n")


if __name__ == "__main__":
    asyncio.run(test_optimized())
