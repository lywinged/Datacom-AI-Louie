#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG System Evaluation - Random 30 Questions (10 from each category)

Pipeline:
1. Warm up embedding + reranker models (switch to fallback MiniLM if CPU latency is high)
2. Randomly select 10 questions from each category (metadata, keyword, semantic)
3. Evaluate total 30 questions
4. Report rich metrics (top-1/top-5 accuracy, latency distribution, reranker usage)
5. Persist JSON results for regression tracking
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

# Ensure project root is on sys.path and load .env
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env", override=False)

from typing import TYPE_CHECKING

# Default model paths (can be overridden by environment variables)
os.environ.setdefault("ONNX_EMBED_MODEL_PATH", "./models/minilm-embed-int8")
os.environ.setdefault("ONNX_RERANK_MODEL_PATH", "./models/minilm-reranker-int8")

# If primary reranker model is missing but fallback exists, default to fallback.
primary_reranker = Path(os.environ["ONNX_RERANK_MODEL_PATH"])
fallback_reranker = os.getenv("RERANK_FALLBACK_MODEL_PATH")
if not primary_reranker.exists() and fallback_reranker and Path(fallback_reranker).exists():
    os.environ["ONNX_RERANK_MODEL_PATH"] = fallback_reranker

import backend.services.rag_pipeline as rag_pipeline
from backend.services.rag_pipeline import retrieve_chunks
from backend.services.metadata_index import warm_metadata_index
from backend.services.onnx_inference import (
    get_reranker_model,
    reranker_is_cpu_only,
    switch_to_fallback_reranker,
    set_reranker_model_path,
)
from backend.config.settings import settings

WARMUP_SWITCH_THRESHOLD_MS = float(os.getenv("WARMUP_SWITCH_THRESHOLD_MS", "300"))
EVAL_METADATA_DATASET_PATH = os.getenv("EVAL_METADATA_DATASET_PATH") or os.getenv("EVAL_DATASET_PATH") or "data/rag_eval_metadata.json"
EVAL_KEYWORD_DATASET_PATH = os.getenv("EVAL_KEYWORD_DATASET_PATH") or os.getenv("EVAL_CONTENT_DATASET_PATH") or "data/rag_eval_keyword.json"
EVAL_SEMANTIC_DATASET_PATH = os.getenv("EVAL_SEMANTIC_DATASET_PATH") or "data/rag_eval_semantic.json"
ROUGE_THRESHOLD_DEFAULT = float(os.getenv("ROUGE_THRESHOLD", "0.6"))


async def warmup_models() -> None:
    """Warm up models and switch reranker if CPU latency is too high."""
    print("🔥 Warming up models...\n")

    print("   Building metadata index...", end=" ")
    try:
        warm_metadata_index()
        print("✅")
    except Exception as exc:
        print(f"⚠️  {exc}")

    warmup_queries = [
        "quick warmup for embedding",
        "another warmup query to stabilize cache",
        "final warmup question to settle models",
    ]

    for idx, query in enumerate(warmup_queries, 1):
        print(f"   Warmup {idx}/{len(warmup_queries)}: ", end="", flush=True)
        try:
            _, _, timings = await retrieve_chunks(
                query,
                top_k=5,
                search_limit=10,
                include_timings=True,
            )

            total_ms = timings.get("total_ms", 0.0)
            rerank_ms = timings.get("rerank_ms", 0.0)
            reranker_model = timings.get("reranker_model_path") or getattr(
                get_reranker_model(), "model_path", "unknown"
            )
            print(f"✅ {total_ms:.1f}ms (rerank {rerank_ms:.1f}ms, model={Path(reranker_model).name})")

            if (
                reranker_is_cpu_only()
                and total_ms > WARMUP_SWITCH_THRESHOLD_MS
                and switch_to_fallback_reranker()
            ):
                new_model = getattr(get_reranker_model(), "model_path", "unknown")
                print(
                    f"      ↪️  Warmup latency {total_ms:.1f}ms exceeded {WARMUP_SWITCH_THRESHOLD_MS:.0f}ms;"
                    f" switched reranker to {Path(new_model).name}"
                )
        except Exception as exc:
            print(f"⚠️  Warmup failed: {exc}")

    print("\n✅ Model warmup complete!\n")


def load_dataset(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _aliases_from_keywords(keywords: List[str], extra: Optional[List[str]] = None) -> Set[str]:
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
    chunks: List[object],
    keywords: List[str],
    source_hints: List[str],
    extra_aliases: Optional[List[str]] = None,
    rouge_threshold: float = ROUGE_THRESHOLD_DEFAULT,
) -> Tuple[bool, bool, Optional[int]]:
    """Return (top5_hit, top1_hit, best_rank) for a question."""
    if not chunks:
        return False, False, None

    all_text = " ".join(chunk.content.lower() for chunk in chunks)
    all_sources = " ".join((chunk.source or "").lower() for chunk in chunks)
    all_authors = " ".join(
        str(chunk.metadata.get("authors", "") or "").lower()
        for chunk in chunks
    )
    alias_set = _aliases_from_keywords(keywords, extra_aliases)

    def find_rank(target: str) -> Optional[int]:
        tgt = target.lower()
        for rank, chunk in enumerate(chunks, 1):
            if tgt in chunk.content.lower():
                return rank
        return None

    # Exact keyword match
    for alias in alias_set:
        if alias and alias in all_text:
            rank = find_rank(alias)
            top1 = alias in chunks[0].content.lower()
            return True, top1, rank

    # Source hint fallback
    for hint in source_hints:
        hint_lower = hint.lower()
        if hint_lower in all_sources or hint_lower in all_text:
            for alias in alias_set:
                parts = [part for part in alias.split() if len(part) > 3]
                if parts and any(part in all_text or part in all_authors for part in parts):
                    top1 = any(part in chunks[0].content.lower() for part in parts)
                    return True, top1, find_rank(parts[0])

    if alias_set and any(alias in all_authors for alias in alias_set):
        return True, False, None

    # ROUGE-L fallback (semantic match)
    if rouge_threshold and alias_set:
        best_score = 0.0
        best_rank = None
        for idx, c in enumerate(chunks, 1):
            s = max(_rouge_l_f1(alias, c.content) for alias in alias_set)
            if s > best_score:
                best_score = s
                best_rank = idx
        if best_score >= rouge_threshold:
            return True, (best_rank == 1), best_rank

    return False, False, None


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    index = int(len(values_sorted) * q)
    index = min(max(index, 0), len(values_sorted) - 1)
    return values_sorted[index]


async def evaluate(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("RAG System Evaluation - Random 30 Questions (10 per category)")
    print("=" * 80)
    print()

    reranker_label: str

    if args.reranker_path:
        override_path = Path(args.reranker_path).expanduser().resolve()
        if not override_path.exists():
            raise FileNotFoundError(f"Reranker path not found: {override_path}")
        os.environ["ONNX_RERANK_MODEL_PATH"] = str(override_path)
        set_reranker_model_path(str(override_path))
        print(f"🛠️  Forced reranker: {override_path}")
        reranker_label = "custom"
        if "bge" in override_path.name or "bge" in str(override_path).lower():
            rag_pipeline._reranker_switch_locked = True
            settings.RERANK_CPU_SWITCH_THRESHOLD_MS = 1e9
            settings.RERANK_FALLBACK_MODEL_PATH = None
        else:
            rag_pipeline._reranker_switch_locked = False
            settings.RERANK_CPU_SWITCH_THRESHOLD_MS = float(os.getenv("RERANK_CPU_SWITCH_THRESHOLD_MS", "450.0"))
            settings.RERANK_FALLBACK_MODEL_PATH = os.getenv("RERANK_FALLBACK_MODEL_PATH")
        print(f"   Reranker switch locked: {rag_pipeline._reranker_switch_locked}")
    elif args.reranker_choice:
        presets = {
            "bge": Path("./models/bge-reranker-int8"),
            "bge-fp32": Path("./models/bge-reranker-int8"),
            "minilm": Path("./models/minilm-reranker-onnx"),
            "minilm-fp32": Path("./models/minilm-reranker-onnx"),
        }
        preset_path = presets[args.reranker_choice].resolve()
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset reranker path missing: {preset_path}")
        if args.reranker_choice.endswith("fp32"):
            os.environ["USE_INT8_QUANTIZATION"] = "false"
        else:
            os.environ["USE_INT8_QUANTIZATION"] = "true"
        os.environ["ONNX_RERANK_MODEL_PATH"] = str(preset_path)
        set_reranker_model_path(str(preset_path))
        print(f"🛠️  Using preset reranker '{args.reranker_choice}' -> {preset_path}")
        reranker_label = args.reranker_choice
        if args.reranker_choice.startswith("bge"):
            rag_pipeline._reranker_switch_locked = True
            settings.RERANK_CPU_SWITCH_THRESHOLD_MS = 1e9
            settings.RERANK_FALLBACK_MODEL_PATH = None
        else:
            rag_pipeline._reranker_switch_locked = False
            settings.RERANK_CPU_SWITCH_THRESHOLD_MS = float(os.getenv("RERANK_CPU_SWITCH_THRESHOLD_MS", "450.0"))
            settings.RERANK_FALLBACK_MODEL_PATH = os.getenv("RERANK_FALLBACK_MODEL_PATH")
        print(f"   Reranker switch locked: {rag_pipeline._reranker_switch_locked}")
    else:
        # Default to MiniLM preset
        
        preset_path = Path("./models/minilm-reranker-onnx").resolve()
        if not preset_path.exists():
            raise FileNotFoundError(f"Default MiniLM reranker not found at {preset_path}")
        os.environ["ONNX_RERANK_MODEL_PATH"] = str(preset_path)
        set_reranker_model_path(str(preset_path))
        os.environ["USE_INT8_QUANTIZATION"] = "true"
        rag_pipeline._reranker_switch_locked = False
        settings.RERANK_CPU_SWITCH_THRESHOLD_MS = float(os.getenv("RERANK_CPU_SWITCH_THRESHOLD_MS", "450.0"))
        settings.RERANK_FALLBACK_MODEL_PATH = os.getenv("RERANK_FALLBACK_MODEL_PATH")
        reranker_label = "minilm"
        print(f"🛠️  Using default reranker 'minilm' -> {preset_path}")

    await warmup_models()

    # Load all three datasets
    all_datasets = {}

    if Path(EVAL_METADATA_DATASET_PATH).exists():
        all_datasets["metadata"] = load_dataset(EVAL_METADATA_DATASET_PATH)
        print(f"📚 Loaded {len(all_datasets['metadata'])} metadata questions")
    else:
        print(f"⚠️  Metadata dataset not found: {EVAL_METADATA_DATASET_PATH}")

    if Path(EVAL_KEYWORD_DATASET_PATH).exists():
        all_datasets["keyword"] = load_dataset(EVAL_KEYWORD_DATASET_PATH)
        print(f"📚 Loaded {len(all_datasets['keyword'])} keyword questions")
    else:
        print(f"⚠️  Keyword dataset not found: {EVAL_KEYWORD_DATASET_PATH}")

    if Path(EVAL_SEMANTIC_DATASET_PATH).exists():
        all_datasets["semantic"] = load_dataset(EVAL_SEMANTIC_DATASET_PATH)
        print(f"📚 Loaded {len(all_datasets['semantic'])} semantic questions")
    else:
        print(f"⚠️  Semantic dataset not found: {EVAL_SEMANTIC_DATASET_PATH}")

    if not all_datasets:
        raise RuntimeError("No datasets found!")

    # Randomly select 10 questions from each category
    dataset: List[Dict[str, object]] = []
    dataset_categories: List[str] = []

    random.seed(args.seed)  # For reproducibility

    for category, questions in all_datasets.items():
        if len(questions) >= 10:
            selected = random.sample(questions, 10)
        else:
            selected = questions  # Use all if less than 10

        dataset.extend(selected)
        dataset_categories.extend([category] * len(selected))
        print(f"🎲 Randomly selected {len(selected)} questions from {category}")

    print()

    if args.output_path:
        result_path = Path(args.output_path)
    else:
        suffix = reranker_label.replace("/", "-")
        result_path = Path(f"eval_results_{suffix}_random30.json")

    total_questions = len(dataset)
    print(f"📊 Total evaluation questions: {total_questions}\n")

    results: List[Dict[str, object]] = []
    latencies: List[float] = []
    latency_breakdown: Dict[str, List[float]] = defaultdict(list)
    reranker_usage: Counter = Counter()
    category_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"top5": 0, "top1": 0, "total": 0})
    category_latency: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"embed": [], "vector": [], "rerank": [], "total": []})

    top5_correct = top1_correct = 0
    hit_ranks: Counter = Counter()

    for idx, (item, category) in enumerate(zip(dataset, dataset_categories), 1):
        question = item["question"]
        keywords = item["answer_keywords"]
        source_hints = item.get("source_hints", [])
        answer_aliases = item.get("answer_aliases", [])

        print(f"[{idx}/{total_questions}] ({category}) {question}")

        try:
            semantic_mode = category == "semantic"
            chunks, _, timings = await retrieve_chunks(
                question,
                top_k=5,
                search_limit=10,
                include_timings=True,
                semantic_mode=semantic_mode,
            )

            retrieval_ms = timings.get("total_ms", 0.0)
            latencies.append(retrieval_ms)

            embed_ms = timings.get("embed_ms", 0.0)
            vector_ms = timings.get("vector_ms", 0.0)
            rerank_ms = timings.get("rerank_ms", 0.0)
            reranker_model = timings.get("reranker_model_path")
            if reranker_model:
                reranker_usage[Path(reranker_model).name] += 1

            latency_breakdown["embed"].append(embed_ms)
            latency_breakdown["vector"].append(vector_ms)
            latency_breakdown["rerank"].append(rerank_ms)
            category_latency[category]["embed"].append(embed_ms)
            category_latency[category]["vector"].append(vector_ms)
            category_latency[category]["rerank"].append(rerank_ms)
            category_latency[category]["total"].append(retrieval_ms)

            top5_hit, top1_hit, best_rank = evaluate_answer(
                chunks,
                keywords,
                source_hints,
                answer_aliases,
                rouge_threshold=args.rouge_threshold,
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
                f"Retrieval {retrieval_ms:.1f}ms (embed {embed_ms:.1f}ms, vector {vector_ms:.1f}ms, rerank {rerank_ms:.1f}ms)"
            )

            results.append(
                {
                    "question": question,
                    "top5_correct": top5_hit,
                    "top1_correct": top1_hit,
                    "category": category,
                    "best_rank": best_rank,
                    "retrieval_ms": retrieval_ms,
                    "embed_ms": embed_ms,
                    "vector_ms": vector_ms,
                    "rerank_ms": rerank_ms,
                    "reranker_model": reranker_model,
                }
            )
        except Exception as exc:
            print(f"  ❌ ERROR: {exc}")
            results.append(
                {
                    "question": question,
                    "error": str(exc),
                    "category": category,
                }
            )
            category_stats[category]["total"] += 1

        print()

    successful_latencies = [lat for lat in latencies if lat > 0]

    category_breakdown = {
        cat: {
            "total": stats["total"],
            "top5_accuracy_percent": (stats["top5"] / stats["total"] * 100) if stats["total"] else 0.0,
            "top1_accuracy_percent": (stats["top1"] / stats["total"] * 100) if stats["total"] else 0.0,
            "latency": {
                stage: (
                    mean(values) if values else 0.0
                )
                for stage, values in category_latency[cat].items()
            },
        }
        for cat, stats in category_stats.items()
    }

    summary = {
        "total_questions": total_questions,
        "top5_correct": top5_correct,
        "top1_correct": top1_correct,
        "top5_accuracy_percent": (top5_correct / total_questions * 100) if total_questions else 0.0,
        "top1_accuracy_percent": (top1_correct / total_questions * 100) if total_questions else 0.0,
        "random_seed": args.seed,
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
        "reranker_usage": dict(reranker_usage),
        "hit_rank_distribution": dict(hit_ranks),
        "category_breakdown": category_breakdown,
    }

    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"🎯 Top-5 Accuracy: {summary['top5_accuracy_percent']:.1f}%")
    print(f"🎯 Top-1 Accuracy: {summary['top1_accuracy_percent']:.1f}%")
    print(f"⏱️  Mean Latency:  {summary['latency']['mean_ms']:.1f}ms")
    print(f"⏱️  Median Latency:{summary['latency']['median_ms']:.1f}ms")
    print(f"⏱️  P95 Latency:   {summary['latency']['p95_ms']:.1f}ms")
    print(f"🎲 Random Seed:   {summary['random_seed']}")
    print("⏱️  LLM Disabled (retrieval-only).")
    print(f"🔁 Reranker Usage: {summary['reranker_usage']}")
    print("📚 Category Breakdown:")
    for cat, stats in summary["category_breakdown"].items():
        latency = stats.get("latency", {})
        print(
            f"   - {cat}: total={stats['total']}, top-5={stats['top5_accuracy_percent']:.1f}%, "
            f"top-1={stats['top1_accuracy_percent']:.1f}%, "
            f"latency(ms) total={latency.get('total', 0.0):.1f}, embed={latency.get('embed', 0.0):.1f}, "
            f"vector={latency.get('vector', 0.0):.1f}, rerank={latency.get('rerank', 0.0):.1f}"
        )
    print()

    result_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"📁 Results saved to {result_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG evaluation with random 30 questions (10 per category)")
    parser.add_argument(
        "--reranker",
        dest="reranker_choice",
        choices=["bge", "bge-fp32", "minilm", "minilm-fp32"],
        help="Choose a preset reranker model",
    )
    parser.add_argument(
        "--reranker-path",
        dest="reranker_path",
        help="Explicit path to reranker ONNX model",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Path to save evaluation results (defaults to eval_results_<reranker>_random30.json)",
    )
    parser.add_argument(
        "--rouge-threshold",
        dest="rouge_threshold",
        type=float,
        default=ROUGE_THRESHOLD_DEFAULT,
        help="ROUGE-L F1 threshold for semantic match (default from ROUGE_THRESHOLD env or 0.6)",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(args))
