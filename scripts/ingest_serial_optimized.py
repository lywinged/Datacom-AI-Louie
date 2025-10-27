#!/usr/bin/env python3
"""
Optimized serial ingestion script for the ONNX + Qdrant RAG pipeline.

Improvements:
1. Batched Qdrant uploads (500 points per batch) to avoid timeouts
2. Progress tracking with detailed stats
3. Optimized batch_size for embeddings
4. Performance timing per file

Usage:
    PYTHONPATH=. python scripts/ingest_serial_optimized.py \
        --docs-dir data/gutenberg_corpus_5mb \
        --embed-model models/bge-m3-embed-int8 \
        --catalog data/gutenberg_corpus_5mb/pg_catalog.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from qdrant_client.http import models as qdrant_models
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config.settings import settings
from backend.services.onnx_inference import ONNXEmbeddingModel
from backend.services.qdrant_client import ensure_collection, get_qdrant_client
from backend.utils.file_loader import load_document_from_path
from backend.utils.text_splitter import split_text


# Optimization constants
EMBEDDING_BATCH_SIZE = 64  # Increased from default 16
QDRANT_UPLOAD_BATCH = 500  # Upload in batches to avoid timeouts


def load_catalog(catalog_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not catalog_path or not catalog_path.exists():
        return {}
    index: Dict[str, Dict[str, str]] = {}
    with catalog_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            book_id = row.get("Text#", "").strip()
            if book_id:
                index[book_id] = {
                    "type": row.get("Type", ""),
                    "issued": row.get("Issued", ""),
                    "title": row.get("Title", ""),
                    "language": row.get("Language", ""),
                    "authors": row.get("Authors", ""),
                    "subjects": row.get("Subjects", ""),
                    "bookshelves": row.get("Bookshelves", ""),
                }
    return index


def extract_book_id(filename: str) -> Optional[str]:
    import re

    match = re.search(r"_(\d+)\.txt$", filename)
    if match:
        return match.group(1)
    return None


def upload_points_batched(client, collection_name: str, points: List, batch_size: int = QDRANT_UPLOAD_BATCH):
    """Upload points in batches to avoid timeout."""
    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
    return total


def ingest_directory(
    docs_dir: Path,
    embed_model_path: str,
    catalog_index: Dict[str, Dict[str, str]],
    max_mb: Optional[int] = None,
    export_json: Optional[Path] = None,
    collection_name: Optional[str] = None,
) -> Dict:
    print("="*60)
    print("🚀 Optimized Serial Ingestion")
    print("="*60)

    # Initialize
    print(f"\n1. Loading embedding model: {embed_model_path}")
    model_start = time.time()

    # Auto-detect GPU
    device = "cpu"
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        device = "cuda"
        print(f"   🚀 GPU detected! Using CUDA for acceleration")
    else:
        print(f"   ℹ️  Using CPU (GPU not available)")

    embedding_model = ONNXEmbeddingModel(embed_model_path, device=device)
    model_time = time.time() - model_start
    print(f"   ✅ Model loaded in {model_time:.2f}s (device: {device})")

    print(f"\n2. Initializing Qdrant collection...")
    target_collection = collection_name or settings.QDRANT_COLLECTION
    ensure_collection(embedding_model.vector_size, collection=target_collection)
    client = get_qdrant_client()
    print(f"   ✅ Collection '{target_collection}' ready (vector_size={embedding_model.vector_size})")

    # Select files
    files = sorted(p for p in docs_dir.glob("*.txt") if p.is_file())
    if max_mb:
        limit_bytes = max_mb * 1024 * 1024
        selected: List[Path] = []
        total = 0
        for path in files:
            size = path.stat().st_size
            if selected and total + size > limit_bytes:
                break
            selected.append(path)
            total += size
        files = selected

    print(f"\n3. Processing {len(files)} files...")
    print("="*60)

    total_chunks = 0
    total_embed_time = 0.0
    total_upload_time = 0.0
    start_time = time.time()

    json_writer = None
    if export_json:
        export_json.parent.mkdir(parents=True, exist_ok=True)
        json_writer = export_json.open("w", encoding="utf-8")

    for file_idx, path in enumerate(files, 1):
        file_start = time.time()

        print(f"\n[{file_idx}/{len(files)}] {path.name}")

        documents = load_document_from_path(str(path))
        book_id = extract_book_id(path.name)
        catalog_meta = catalog_index.get(book_id or "", {})

        file_chunks = 0
        all_points = []

        for doc_idx, doc in enumerate(documents):
            chunks = split_text(
                doc.content,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            if not chunks:
                continue

            # Embedding with optimized batch size
            embed_start = time.time()
            vectors = embedding_model.encode(chunks, batch_size=EMBEDDING_BATCH_SIZE).astype("float32")
            embed_time = time.time() - embed_start
            total_embed_time += embed_time

            document_id = (
                int(book_id) if book_id else hash(f"{path.name}-{doc_idx}") & 0xFFFFFFFFFFFF
            )

            for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                point_id = document_id * 10000 + idx
                vector_list = vector.tolist()
                payload = {
                    "document_id": int(document_id),
                    "book_id": book_id,
                    "chunk_index": idx,
                    "title": catalog_meta.get("title") or doc.title or path.stem,
                    "source": doc.source or path.name,
                    "type": catalog_meta.get("type"),
                    "issued": catalog_meta.get("issued"),
                    "language": catalog_meta.get("language"),
                    "authors": catalog_meta.get("authors"),
                    "subjects": catalog_meta.get("subjects"),
                    "bookshelves": catalog_meta.get("bookshelves"),
                    "text": chunk,
                    "metadata": {
                        **doc.metadata,
                        "relative_path": str(path),
                    },
                }

                all_points.append(
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector_list,
                        payload=payload,
                    )
                )

                if json_writer:
                    json.dump({"id": point_id, "vector": vector_list, "payload": payload}, json_writer)
                    json_writer.write("\n")

            file_chunks += len(chunks)

        # Batched upload to avoid timeouts
        if all_points:
            upload_start = time.time()
            uploaded = upload_points_batched(client, target_collection, all_points)
            upload_time = time.time() - upload_start
            total_upload_time += upload_time

            file_time = time.time() - file_start
            total_chunks += file_chunks

            # Progress output
            print(f"   - Chunks: {file_chunks}")
            print(f"   - Embed:  {embed_time:.2f}s ({file_chunks/embed_time:.1f} chunks/s)")
            print(f"   - Upload: {upload_time:.2f}s ({uploaded/upload_time:.0f} chunks/s)")
            print(f"   ✅ Total: {file_time:.2f}s")

    if json_writer:
        json_writer.close()

    total_time = time.time() - start_time

    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    print(f"  Files processed:    {len(files)}")
    print(f"  Total chunks:       {total_chunks}")
    print(f"  Total time:         {total_time:.2f}s ({total_time/60:.1f} min)")
    embed_throughput = (total_chunks / total_embed_time) if total_embed_time > 0 else 0.0
    upload_throughput = (total_chunks / total_upload_time) if total_upload_time > 0 else 0.0
    overall_throughput = (total_chunks / total_time) if total_time > 0 else 0.0
    avg_file_time = (total_time / len(files)) if files else 0.0

    print(f"  Embedding time:     {total_embed_time:.2f}s ({embed_throughput:.1f} chunks/s)")
    print(f"  Upload time:        {total_upload_time:.2f}s ({upload_throughput:.1f} chunks/s)")
    print(f"  Avg per file:       {avg_file_time:.2f}s")
    print(f"  Overall throughput: {overall_throughput:.1f} chunks/s")
    print("="*60)
    if export_json:
        print(f"  JSON export:        {export_json.resolve()}")
        print("="*60)

    return {
        "files_ingested": len(files),
        "total_chunks": total_chunks,
        "total_time_seconds": round(total_time, 2),
        "embedding_time_seconds": round(total_embed_time, 2),
        "upload_time_seconds": round(total_upload_time, 2),
        "avg_file_time_seconds": round(avg_file_time, 2),
        "throughput_chunks_per_second": round(overall_throughput, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized serial ingestion into Qdrant.")
    parser.add_argument("--docs-dir", type=Path, required=True, help="Directory containing .txt files.")
    parser.add_argument("--embed-model", type=str, required=True, help="Path to embedding ONNX model.")
    parser.add_argument("--catalog", type=Path, help="Path to pg_catalog.csv for metadata enrichment.")
    parser.add_argument("--max-mb", type=int, help="Optional cap on corpus size in MB.")
    parser.add_argument("--export-json", type=Path, help="Optional path to export embeddings as JSONL.")
    parser.add_argument("--collection", type=str, help="Qdrant collection name (default: from settings)")
    args = parser.parse_args()

    if not args.docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {args.docs_dir}")

    catalog_index = load_catalog(args.catalog)
    stats = ingest_directory(
        docs_dir=args.docs_dir,
        embed_model_path=args.embed_model,
        catalog_index=catalog_index,
        max_mb=args.max_mb,
        export_json=args.export_json,
        collection_name=args.collection,
    )

    print(f"\n📄 JSON Output:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
