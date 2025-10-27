#!/usr/bin/env python3
"""
CLI helper to populate Qdrant with the bundled seed dataset.

Usage:
    python scripts/bootstrap_qdrant_seed.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.backend.services.qdrant_seed import ensure_seed_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Qdrant with seed vectors.")
    parser.add_argument(
        "--seed-path",
        type=Path,
        default=Path("data/qdrant_seed/assessment_docs_minilm.jsonl"),
        help="Path to the JSONL file containing vector seed data.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=13000,
        help="Expected number of vectors after seeding.",
    )
    args = parser.parse_args()

    summary = ensure_seed_collection(seed_path=args.seed_path, target_count=args.target_count)
    print("Seed summary:", summary)


if __name__ == "__main__":
    main()
