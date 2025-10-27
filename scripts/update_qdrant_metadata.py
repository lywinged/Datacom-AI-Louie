#!/usr/bin/env python3
"""
Update Qdrant metadata without re-chunking.

This script:
1. Loads pg_catalog_150_complete.csv
2. For each book, finds all its chunks in Qdrant
3. Updates the payload with authors, subjects, etc.
4. Much faster than re-ingesting (no chunking/embedding needed)
"""

import csv
import sys
import re
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SetPayloadOperation
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_catalog(catalog_path: Path) -> Dict[int, Dict]:
    """Load catalog and create mapping from Text# to metadata."""
    books = {}

    with open(catalog_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_num = row.get('Text#', '')
            if text_num:
                try:
                    books[int(text_num)] = {
                        'type': row.get('Type', ''),
                        'issued': row.get('Issued', ''),
                        'title': row.get('Title', ''),
                        'language': row.get('Language', 'en'),
                        'authors': row.get('Authors', ''),
                        'subjects': row.get('Subjects', ''),
                        'locc': row.get('LoCC', ''),
                        'bookshelves': row.get('Bookshelves', '')
                    }
                except ValueError:
                    continue

    return books


def extract_text_number_from_source(source: str) -> int:
    """Extract Text# from source like '00116_roderick-hudson_176.txt' -> 176"""
    match = re.search(r'_(\d+)\.txt$', source)
    if match:
        return int(match.group(1))
    return -1


def update_metadata_batch(client: QdrantClient, collection: str, points_to_update: List):
    """Update metadata for a batch of points."""
    if not points_to_update:
        return

    for point_id, metadata in points_to_update:
        try:
            client.set_payload(
                collection_name=collection,
                payload=metadata,
                points=[point_id]
            )
        except Exception as e:
            print(f"      ⚠️  Error updating point {point_id}: {e}")


def main():
    print("=" * 80)
    print("Update Qdrant Metadata (No Re-chunking)")
    print("=" * 80)
    print()

    # Configuration
    catalog_path = Path("data/gutenberg_corpus/pg_catalog_150_complete.csv")
    collection_name = "assessment_docs_minilm"
    qdrant_host = "localhost"
    qdrant_port = 6333

    # Step 1: Load catalog
    print(f"📚 Loading catalog from {catalog_path}")
    books_metadata = load_catalog(catalog_path)
    print(f"   Loaded metadata for {len(books_metadata)} books")
    print()

    # Step 2: Connect to Qdrant
    print(f"🔌 Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Check collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name not in collection_names:
        print(f"❌ Collection '{collection_name}' not found!")
        print(f"   Available: {collection_names}")
        return

    print(f"   ✅ Connected to collection: {collection_name}")

    # Get collection info
    info = client.get_collection(collection_name)
    total_points = info.points_count
    print(f"   Total points in collection: {total_points:,}")
    print()

    # Step 3: Scroll through all points and update
    print("🔄 Updating metadata for all points...")
    print()

    offset = None
    batch_size = 100
    processed = 0
    updated = 0
    books_updated = set()

    while True:
        # Scroll batch
        result = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        points, next_offset = result

        if not points:
            break

        # Process batch
        updates = []

        for point in points:
            processed += 1

            # Get source from payload
            payload = point.payload or {}
            source = payload.get('source', '')

            if not source:
                continue

            # Extract Text#
            text_num = extract_text_number_from_source(source)

            if text_num < 0 or text_num not in books_metadata:
                continue

            # Get metadata from catalog
            book_meta = books_metadata[text_num]

            # Prepare update payload
            update_payload = {
                'type': book_meta['type'],
                'issued': book_meta['issued'],
                'language': book_meta['language'],
                'authors': book_meta['authors'],
                'subjects': book_meta['subjects'],
                'bookshelves': book_meta['bookshelves']
            }

            # Update point
            updates.append((point.id, update_payload))
            books_updated.add(text_num)

        # Batch update
        if updates:
            update_metadata_batch(client, collection_name, updates)
            updated += len(updates)

        # Progress
        if processed % 1000 == 0:
            print(f"   Processed: {processed:,}/{total_points:,} points ({updated:,} updated)")

        # Check if done
        if next_offset is None:
            break

        offset = next_offset

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Processed:      {processed:,} points")
    print(f"✅ Updated:        {updated:,} points")
    print(f"✅ Books updated:  {len(books_updated)} unique books")
    print()

    # Show sample
    print("📋 Verifying sample point...")
    sample = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False
    )[0]

    if sample:
        point = sample[0]
        payload = point.payload
        print(f"   Source:   {payload.get('source', 'N/A')}")
        print(f"   Authors:  {payload.get('authors', 'N/A')}")
        print(f"   Subjects: {payload.get('subjects', 'N/A')[:60]}...")
        print(f"   Language: {payload.get('language', 'N/A')}")

    print()
    print("=" * 80)
    print("✅ Metadata update complete!")
    print("   You can now search by authors/subjects without re-chunking")
    print("=" * 80)


if __name__ == "__main__":
    main()
