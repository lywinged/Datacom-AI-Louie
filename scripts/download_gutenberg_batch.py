#!/usr/bin/env python3
"""
Download a random batch of Project Gutenberg books until a target corpus size is reached.

Usage:
    python scripts/download_gutenberg_batch.py --out data/gutenberg_corpus --target-mb 200

The script:
  1. Fetches the Project Gutenberg catalogue (pg_catalog.csv)
  2. Filters to English titles
  3. Randomly samples titles and downloads their plain-text files
  4. Stops once the cumulative size of saved .txt files meets the requested target

Note: Project Gutenberg books are in the public domain. Please respect their terms of use:
https://www.gutenberg.org/policy/license.html
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import requests
from tqdm import tqdm

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
USER_AGENT = "ai-assessment-project/1.0 (https://www.gutenberg.org)"

# Candidate URL patterns for plain-text downloads
TEXT_URL_PATTERNS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt.utf8",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt.utf-8",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}-8.txt",
]


def slugify(value: str, max_length: int = 60) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value[:max_length] or "book"


def load_catalog() -> list[dict[str, str]]:
    resp = requests.get(CATALOG_URL, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    decoded = resp.content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(decoded.splitlines())
    return list(reader)


def iter_english_books(rows: Iterable[dict[str, str]]) -> Iterable[dict[str, str]]:
    english_rows = [row for row in rows if "en" in (row.get("Language", "") or "").split(",")]
    random.shuffle(english_rows)
    return english_rows


def try_download_book(book_id: str) -> Optional[bytes]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for pattern in TEXT_URL_PATTERNS:
        url = pattern.format(id=book_id)
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 200 and len(resp.content) > 50_000:
                return resp.content
        except requests.RequestException:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Project Gutenberg books up to a target corpus size.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for downloaded .txt files.")
    parser.add_argument("--target-mb", type=int, default=200, help="Target corpus size in megabytes (default: 200).")
    parser.add_argument("--max-books", type=int, default=2000, help="Optional cap on number of books to attempt.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between downloads to avoid hammering the site.")
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024 * 1024
    args.out.mkdir(parents=True, exist_ok=True)

    print("Fetching Project Gutenberg catalog…")
    catalog = load_catalog()
    books_iter = iter(iter_english_books(catalog))

    total_bytes = 0
    downloaded = 0
    attempted = 0

    progress = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="Corpus size")

    try:
        while total_bytes < target_bytes and attempted < args.max_books:
            try:
                row = next(books_iter)
            except StopIteration:
                print("Ran out of catalog entries before reaching the target size.")
                break

            attempted += 1
            book_id = row.get("Text#", "").strip()
            title = row.get("Title", "Unknown Title")
            if not book_id.isdigit():
                continue

            content = try_download_book(book_id)
            if not content:
                continue

            filename = f"{downloaded:05d}_{slugify(title)}_{book_id}.txt"
            output_path = args.out / filename
            output_path.write_bytes(content)

            file_size = len(content)
            total_bytes += file_size
            downloaded += 1
            progress.update(file_size)
            progress.set_postfix({"books": downloaded})

            if args.sleep > 0:
                time.sleep(args.sleep)

    finally:
        progress.close()

    print(f"Downloaded {downloaded} books totaling {total_bytes / (1024 * 1024):.2f} MB.")
    if total_bytes < target_bytes:
        print("Warning: target corpus size not reached. Try increasing --max-books or using additional sources.")


if __name__ == "__main__":
    main()
