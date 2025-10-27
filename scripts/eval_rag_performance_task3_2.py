#!/usr/bin/env python3
"""
RAG System Evaluation via API - Random 60 Questions (20 per category)

This script evaluates the RAG system by making HTTP requests to the running Docker backend.
Randomly selects 20 questions from each of the 3 categories (metadata, keyword, semantic).

Warm-up uses 3 random questions from each category (9 total).

No need for local dependencies or PYTHONPATH configuration.

Usage:
    python3 scripts/eval_rag_api_random30.py
    python3 scripts/eval_rag_api_random30.py --backend http://localhost:8888 --seed 123
    python3 scripts/eval_rag_api_random30.py --output results.json
"""

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Set, Tuple

import requests

# Default paths
DEFAULT_BACKEND_URL = "http://localhost:8888"
EVAL_METADATA_PATH = "../data/rag_eval_metadata.json"
EVAL_KEYWORD_PATH = "../data/rag_eval_keyword.json"
EVAL_SEMANTIC_PATH = "../data/rag_eval_semantic.json"
ROUGE_THRESHOLD_DEFAULT = 0.6


def load_dataset(file_path: str) -> List[Dict]:
    """Load a single evaluation dataset."""
    full_path = Path(__file__).parent / file_path
    if not full_path.exists():
        print(f"⚠️  File not found: {full_path}")
        return []

    with open(full_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    else:
        print(f"⚠️  Unexpected format in {full_path}")
        return []


def warmup_backend(backend_url: str, all_datasets: Dict[str, List[Dict]], seed: int) -> None:
    """Warm up the backend with 3 random questions from each category (9 total)."""
    warmup_questions = []

    # Select 3 random questions from each category for warmup
    warmup_rng = random.Random(seed)  # Use separate RNG for warmup
    for category, questions in all_datasets.items():
        if len(questions) >= 3:
            selected = warmup_rng.sample(questions, 3)
        else:
            selected = questions[:3]  # Use first 3 if less available

        for q in selected:
            warmup_questions.append({
                "question": q.get("question", ""),
                "category": category
            })

    print(f"🔥 Warming up backend with {len(warmup_questions)} queries (3 from each category)...\n")

    for i, item in enumerate(warmup_questions, 1):
        question = item["question"]
        category = item["category"]

        try:
            response = requests.post(
                f"{backend_url}/api/rag/ask",
                json={
                    "question": question,
                    "top_k": 5,
                    "include_timings": True,
                    "reranker": "fallback",
                    "vector_limit": 6,
                    "content_char_limit": 300
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                timings = data.get("timings", {})
                embed_ms = timings.get("embed_ms", 0)
                total_ms = timings.get("total_ms", 0)

                if i % 3 == 0:  # Show progress every 3 queries
                    print(f"   Warmup {i}/{len(warmup_questions)} ({category}): embed={embed_ms:.1f}ms, total={total_ms:.1f}ms")
            else:
                print(f"   ⚠️  Warmup {i} failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ⚠️  Warmup {i} error: {e}")

    print("\n✅ Warmup complete!\n")


def _aliases_from_keywords(keywords: List[str], extra: Optional[List[str]] = None) -> Set[str]:
    """Generate aliases from keywords."""
    aliases: Set[str] = set()
    terms = list(keywords)
    if extra:
        terms.extend(extra)
    for keyword in terms:
        cleaned = keyword.strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        aliases.add(lower)
        aliases.add(re.sub(r"[^a-z0-9]+", " ", lower).strip())
        for part in re.split(r"[;,]", lower):
            part = part.strip()
            if part:
                aliases.add(part)
    return {alias for alias in aliases if alias}


def _lcs_len(a: List[str], b: List[str]) -> int:
    """Calculate longest common subsequence length."""
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def _rouge_l_f1(reference: str, candidate: str) -> float:
    """Calculate ROUGE-L F1 score."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = _lcs_len(ref_tokens, cand_tokens)
    rec = lcs / len(ref_tokens)
    prec = lcs / len(cand_tokens)
    if rec == 0 or prec == 0:
        return 0.0
    return 2 * rec * prec / (rec + prec)


def evaluate_answer(
    chunks: List[Dict],
    keywords: List[str],
    source_hints: List[str],
    extra_aliases: Optional[List[str]] = None,
    rouge_threshold: float = ROUGE_THRESHOLD_DEFAULT,
) -> Tuple[bool, bool, Optional[int]]:
    """
    Evaluate if the retrieved chunks contain the expected answer.
    Returns (top5_hit, top1_hit, best_rank).
    """
    if not chunks:
        return False, False, None

    # Extract text and metadata from chunks
    all_text = " ".join(chunk.get("content", "").lower() for chunk in chunks)
    all_sources = " ".join(chunk.get("source", "").lower() for chunk in chunks)
    all_authors = " ".join(
        str(chunk.get("metadata", {}).get("authors", "")).lower() for chunk in chunks
    )

    alias_set = _aliases_from_keywords(keywords, extra_aliases)

    def find_rank(target: str) -> Optional[int]:
        tgt = target.lower()
        for rank, chunk in enumerate(chunks, 1):
            if tgt in chunk.get("content", "").lower():
                return rank
        return None

    # Exact keyword match
    for alias in alias_set:
        if alias and alias in all_text:
            rank = find_rank(alias)
            top1 = alias in chunks[0].get("content", "").lower()
            return True, top1, rank

    # Source hint fallback
    for hint in source_hints:
        hint_lower = hint.lower()
        if hint_lower in all_sources or hint_lower in all_text:
            for alias in alias_set:
                parts = [part for part in alias.split() if len(part) > 3]
                if parts and any(part in all_text or part in all_authors for part in parts):
                    top1 = any(part in chunks[0].get("content", "").lower() for part in parts)
                    return True, top1, find_rank(parts[0])

    if alias_set and any(alias in all_authors for alias in alias_set):
        return True, False, None

    # ROUGE-L fallback (semantic match)
    if rouge_threshold and alias_set:
        best_score = 0.0
        best_rank = None
        for idx, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            s = max(_rouge_l_f1(alias, content) for alias in alias_set)
            if s > best_score:
                best_score = s
                best_rank = idx
        if best_score >= rouge_threshold:
            return True, (best_rank == 1), best_rank

    return False, False, None


def percentile(values: List[float], q: float) -> float:
    """Calculate percentile."""
    if not values:
        return 0.0
    values_sorted = sorted(values)
    index = int(len(values_sorted) * q)
    index = min(max(index, 0), len(values_sorted) - 1)
    return values_sorted[index]


def main():
    parser = argparse.ArgumentParser(
        description="RAG evaluation via API - Random 60 questions (20 per category)"
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND_URL,
        help=f"Backend URL (default: {DEFAULT_BACKEND_URL})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        default="eval_results_api_random60.json",
        help="Output JSON file (default: eval_results_api_random60.json)"
    )
    parser.add_argument(
        "--rouge-threshold",
        type=float,
        default=ROUGE_THRESHOLD_DEFAULT,
        help=f"ROUGE-L F1 threshold (default: {ROUGE_THRESHOLD_DEFAULT})"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RAG System Evaluation via API - Random 60 Questions (20 per category)")
    print("=" * 80)
    print()
    print(f"Backend URL: {args.backend}")
    print(f"Random seed: {args.seed}")
    print()

    # Check backend health
    try:
        health_response = requests.get(f"{args.backend}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Backend health check failed: HTTP {health_response.status_code}")
            return
        print("✅ Backend is healthy\n")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return

    # Load datasets
    all_datasets = {}

    metadata_data = load_dataset(EVAL_METADATA_PATH)
    if metadata_data:
        all_datasets["metadata"] = metadata_data
        print(f"📚 Loaded {len(metadata_data)} metadata questions")

    keyword_data = load_dataset(EVAL_KEYWORD_PATH)
    if keyword_data:
        all_datasets["keyword"] = keyword_data
        print(f"📚 Loaded {len(keyword_data)} keyword questions")

    semantic_data = load_dataset(EVAL_SEMANTIC_PATH)
    if semantic_data:
        all_datasets["semantic"] = semantic_data
        print(f"📚 Loaded {len(semantic_data)} semantic questions")

    if not all_datasets:
        print("❌ No datasets loaded!")
        return

    print()

    # Warmup with 3 questions from each category
    warmup_backend(args.backend, all_datasets, args.seed)

    # Randomly select 20 questions from each category
    random.seed(args.seed)

    dataset = []
    dataset_categories = []

    for category, questions in all_datasets.items():
        if len(questions) >= 20:
            selected = random.sample(questions, 20)
        else:
            selected = questions  # Use all if less than 20

        dataset.extend(selected)
        dataset_categories.extend([category] * len(selected))
        print(f"🎲 Randomly selected {len(selected)} questions from {category}")

    print()

    total_questions = len(dataset)
    print(f"📊 Total evaluation questions: {total_questions}\n")

    # Evaluate
    results = []
    latencies = []
    latency_breakdown = defaultdict(list)
    category_stats = defaultdict(lambda: {"top5": 0, "top1": 0, "total": 0})
    category_latency = defaultdict(
        lambda: {"embed": [], "vector": [], "rerank": [], "total": []}
    )

    top5_correct = top1_correct = 0
    hit_ranks = Counter()

    for idx, (item, category) in enumerate(zip(dataset, dataset_categories), 1):
        question = item.get("question")
        keywords = item.get("answer_keywords", [])
        source_hints = item.get("source_hints", [])
        answer_aliases = item.get("answer_aliases", [])

        print(f"[{idx}/{total_questions}] ({category}) {question}")

        try:
            response = requests.post(
                f"{args.backend}/api/rag/ask",
                json={
                    "question": question,
                    "top_k": 5,
                    "include_timings": True,
                    "reranker": "fallback",
                    "vector_limit": 10,
                    "content_char_limit": 500
                },
                timeout=30
            )

            if response.status_code != 200:
                print(f"  ❌ ERROR: HTTP {response.status_code}")
                category_stats[category]["total"] += 1
                continue

            data = response.json()
            chunks = data.get("citations", [])  # API returns "citations", not "chunks"
            timings = data.get("timings", {})

            retrieval_ms = timings.get("total_ms", 0.0)
            embed_ms = timings.get("embed_ms", 0.0)
            vector_ms = timings.get("vector_ms", 0.0)
            rerank_ms = timings.get("rerank_ms", 0.0)

            latencies.append(retrieval_ms)
            latency_breakdown["embed"].append(embed_ms)
            latency_breakdown["vector"].append(vector_ms)
            latency_breakdown["rerank"].append(rerank_ms)
            category_latency[category]["embed"].append(embed_ms)
            category_latency[category]["vector"].append(vector_ms)
            category_latency[category]["rerank"].append(rerank_ms)
            category_latency[category]["total"].append(retrieval_ms)

            # Evaluate answer
            top5_hit, top1_hit, best_rank = evaluate_answer(
                chunks,
                keywords,
                source_hints,
                answer_aliases,
                args.rouge_threshold,
            )

            category_stats[category]["total"] += 1
            if top5_hit:
                top5_correct += 1
                category_stats[category]["top5"] += 1
            if top1_hit:
                top1_correct += 1
                category_stats[category]["top1"] += 1
            if best_rank is not None:
                hit_ranks[best_rank] += 1

            print(
                f"  {'✅' if top5_hit else '❌'} Top-5 | "
                f"{'✅' if top1_hit else '❌'} Top-1 | "
                f"Retrieval {retrieval_ms:.1f}ms (embed {embed_ms:.1f}ms, "
                f"vector {vector_ms:.1f}ms, rerank {rerank_ms:.1f}ms)"
            )

            results.append({
                "question": question,
                "top5_correct": top5_hit,
                "top1_correct": top1_hit,
                "category": category,
                "best_rank": best_rank,
                "retrieval_ms": retrieval_ms,
                "embed_ms": embed_ms,
                "vector_ms": vector_ms,
                "rerank_ms": rerank_ms,
            })

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            category_stats[category]["total"] += 1
            results.append({
                "question": question,
                "error": str(e),
                "category": category,
            })

        print()

    # Calculate summary statistics
    successful_latencies = [lat for lat in latencies if lat > 0]

    category_breakdown = {
        cat: {
            "total": stats["total"],
            "top5_accuracy_percent": (
                (stats["top5"] / stats["total"] * 100) if stats["total"] else 0.0
            ),
            "top1_accuracy_percent": (
                (stats["top1"] / stats["total"] * 100) if stats["total"] else 0.0
            ),
            "latency": {
                stage: {
                    "mean_ms": mean(values) if values else 0.0,
                    "median_ms": median(values) if values else 0.0,
                }
                for stage, values in category_latency[cat].items()
            },
        }
        for cat, stats in category_stats.items()
    }

    summary = {
        "total_questions": total_questions,
        "top5_correct": top5_correct,
        "top1_correct": top1_correct,
        "top5_accuracy_percent": (
            (top5_correct / total_questions * 100) if total_questions else 0.0
        ),
        "top1_accuracy_percent": (
            (top1_correct / total_questions * 100) if total_questions else 0.0
        ),
        "random_seed": args.seed,
        "backend_url": args.backend,
        "latency": {
            "mean_ms": mean(successful_latencies) if successful_latencies else 0.0,
            "median_ms": percentile(successful_latencies, 0.5),
            "p95_ms": percentile(successful_latencies, 0.95),
            "min_ms": min(successful_latencies) if successful_latencies else 0.0,
            "max_ms": max(successful_latencies) if successful_latencies else 0.0,
        },
        "latency_breakdown": {
            stage: {
                "mean_ms": mean(values) if values else 0.0,
                "median_ms": percentile(values, 0.5),
                "p95_ms": percentile(values, 0.95),
            }
            for stage, values in latency_breakdown.items()
        },
        "hit_rank_distribution": dict(hit_ranks),
        "category_breakdown": category_breakdown,
    }

    # Print summary
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"🎯 Top-5 Accuracy: {summary['top5_accuracy_percent']:.1f}%")
    print(f"🎯 Top-1 Accuracy: {summary['top1_accuracy_percent']:.1f}%")
    print(f"⏱️  Mean Latency:  {summary['latency']['mean_ms']:.1f}ms")
    print(f"⏱️  Median Latency:{summary['latency']['median_ms']:.1f}ms")
    print(f"⏱️  P95 Latency:   {summary['latency']['p95_ms']:.1f}ms")
    print(f"🎲 Random Seed:   {summary['random_seed']}")
    print("📚 Category Breakdown:")
    for cat, stats in summary["category_breakdown"].items():
        latency = stats.get("latency", {})
        total_latency = latency.get("total", {})
        embed_latency = latency.get("embed", {})
        vector_latency = latency.get("vector", {})
        rerank_latency = latency.get("rerank", {})

        print(f"\n   {cat.upper()}:")
        print(f"     Total: {stats['total']} questions")
        print(f"     Top-5 Accuracy: {stats['top5_accuracy_percent']:.1f}%")
        print(f"     Top-1 Accuracy: {stats['top1_accuracy_percent']:.1f}%")
        print(f"     Retrieval Time - Mean: {total_latency.get('mean_ms', 0):.1f}ms, Median: {total_latency.get('median_ms', 0):.1f}ms")
        print(f"     Embed Time     - Mean: {embed_latency.get('mean_ms', 0):.1f}ms, Median: {embed_latency.get('median_ms', 0):.1f}ms")
        print(f"     Vector Time    - Mean: {vector_latency.get('mean_ms', 0):.1f}ms, Median: {vector_latency.get('median_ms', 0):.1f}ms")
        print(f"     Rerank Time    - Mean: {rerank_latency.get('mean_ms', 0):.1f}ms, Median: {rerank_latency.get('median_ms', 0):.1f}ms")
    print()

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"📁 Results saved to {output_path}\n")


if __name__ == "__main__":
    main()
