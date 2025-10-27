#!/usr/bin/env python3
# scripts/ingest_ray_parallel.py
from __future__ import annotations
import argparse, csv, json, os, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
from tqdm import tqdm
from qdrant_client.http import models as qmodels

# Project modules (keep as-is)
from backend.config.settings import settings
from backend.services.onnx_inference import ONNXEmbeddingModel
from backend.services.qdrant_client import ensure_collection, get_qdrant_client
from backend.utils.file_loader import load_document_from_path
from backend.utils.text_splitter import split_text

# ---------- Helper utilities ----------

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
    m = re.search(r"_(\d+)\.txt$", filename)
    return m.group(1) if m else None

# ---------- Progress actor ----------

@ray.remote
class ProgressActor:
    def __init__(self, total_files:int):
        self.total_files = total_files
        self.files_done = 0
        self.points_done = 0
        self.embed_time = 0.0
        self.upload_time = 0.0
        self.chunks_done = 0

    def inc_file(self):
        self.files_done += 1

    def add_points(self, n:int):
        self.points_done += n

    def add_embed_time(self, t:float):
        self.embed_time += t

    def add_upload_time(self, t:float):
        self.upload_time += t

    def add_chunks(self, n:int):
        self.chunks_done += n

    def snapshot(self):
        return {
            "files_done": self.files_done,
            "total_files": self.total_files,
            "points_done": self.points_done,
            "embed_time": self.embed_time,
            "upload_time": self.upload_time,
            "chunks_done": self.chunks_done,
        }

# ---------- Worker (each Ray process owns one ONNX session + one Qdrant client) ----------

@ray.remote(num_cpus=1)
class IngestWorker:
    def __init__(self, embed_model_path:str, qdrant_url:str, collection:str,
                 embed_batch:int, upload_batch:int):
        # Local ONNX inference session (parallelism is handled by Ray)
        self.embedding_model = ONNXEmbeddingModel(embed_model_path, device="cpu")
        # Qdrant client reused per worker
        self.client = get_qdrant_client()
        self.collection = collection
        self.embed_batch = embed_batch
        self.upload_batch = upload_batch

    def process_file(self, file_path:str, catalog_meta:Dict[str,str]) -> Dict:
        # Summary for the current file
        from qdrant_client.http import models as qmodels
        path = Path(file_path)
        documents = load_document_from_path(str(path))
        book_id = extract_book_id(path.name)
        total_chunks = 0
        file_start = time.time()
        total_embed_time = 0.0
        total_upload_time = 0.0

        # Split each document independently to avoid keeping too much in memory
        for doc_idx, doc in enumerate(documents):
            chunks = split_text(
                doc.content,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            if not chunks:
                continue
            total_chunks += len(chunks)

            # Batch embeddings according to embed_batch
            vectors_all: List[np.ndarray] = []
            embed_t0 = time.time()
            for i in range(0, len(chunks), self.embed_batch):
                sub = chunks[i : i + self.embed_batch]
                vecs = self.embedding_model.encode(sub, batch_size=self.embed_batch).astype("float32")
                vectors_all.append(vecs)
            embed_time = time.time() - embed_t0
            total_embed_time += embed_time

            vectors = np.vstack(vectors_all)
            # Construct IDs (document_id * 10000 + idx, same as original logic)
            document_id = int(book_id) if book_id else (hash(f"{path.name}-{doc_idx}") & 0xFFFFFFFFFFFF)

            # Build points and upload in batches to avoid huge buffers in memory
            upload_t0 = time.time()
            buf_points: List[qmodels.PointStruct] = []
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                point_id = document_id * 10000 + idx
                buf_points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
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
                            "metadata": { **doc.metadata, "relative_path": str(path) },
                        },
                    )
                )
                # Flush batched upserts
                if len(buf_points) >= self.upload_batch:
                    self.client.upsert(collection_name=self.collection, points=buf_points)
                    buf_points.clear()
            # Flush any remaining points
            if buf_points:
                self.client.upsert(collection_name=self.collection, points=buf_points)
                buf_points.clear()
            upload_time = time.time() - upload_t0
            total_upload_time += upload_time

        return {
            "file": path.name,
            "chunks": total_chunks,
            "embed_time": total_embed_time,
            "upload_time": total_upload_time,
            "file_time": time.time() - file_start,
        }

# ---------- Main workflow ----------

def main():
    ap = argparse.ArgumentParser(description="Ray-parallel ingestion into Qdrant.")
    ap.add_argument("--docs-dir", type=Path, required=True)
    ap.add_argument("--embed-model", type=str, required=True)
    ap.add_argument("--catalog", type=Path)
    ap.add_argument("--max-mb", type=int)
    ap.add_argument("--num-workers", type=int, default=max(os.cpu_count()-2, 1))
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--upload-batch", type=int, default=1000)
    ap.add_argument("--qdrant-url", type=str, default="http://127.0.0.1:6333")
    ap.add_argument("--collection", type=str, required=True)
    ap.add_argument("--ray-address", type=str, default="")  # Empty string => start Ray locally
    ap.add_argument("--dashboard-port", type=int, default=8265)
    args = ap.parse_args()

    if not args.docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {args.docs_dir}")

    # Preselect candidate files
    files = sorted(p for p in args.docs_dir.glob("*.txt") if p.is_file())
    if args.max_mb:
        limit_bytes = args.max_mb * 1024 * 1024
        sel, acc = [], 0
        for p in files:
            sz = p.stat().st_size
            if sel and acc + sz > limit_bytes: break
            sel.append(p); acc += sz
        files = sel

    print("="*60)
    print(f"🧩 Files to ingest: {len(files)}")
    print("="*60)
    if not files:
        print("No .txt files found."); return

    # Start Ray locally or connect to an existing cluster
    if args.ray_address:
        print(f"🔗 Connecting Ray at {args.ray_address}")
        ray.init(address=args.ray_address)
    else:
        # Ensure Ray spills to a controlled directory to avoid /tmp exhaustion
        os.environ.setdefault("RAY_TMPDIR", str(Path.cwd() / "ray_tmp"))
        spill_dir = Path.cwd() / "ray_spill"
        spill_dir.mkdir(parents=True, exist_ok=True)
        ray.init(
            num_cpus=args.num_workers,
            include_dashboard=True,
            dashboard_port=args.dashboard_port,
            _system_config={
                "object_spilling_config": json.dumps({
                    "type": "filesystem",
                    "params": {"directory_path": str(spill_dir)}
                })
            },
        )
    print("Ray resources:", ray.cluster_resources())

    # Ensure the collection exists before ingesting
    print("\n📦 Ensuring collection ...")
    ensure_collection(vector_size=ONNXEmbeddingModel(args.embed_model, device="cpu").vector_size,
                      client=get_qdrant_client(),
                      collection_name=args.collection)
    print("✅ Collection ready:", args.collection)

    # catalog
    catalog_index = load_catalog(args.catalog)

    # Create progress actor
    progress = ProgressActor.remote(total_files=len(files))

    # Dispatch one task per file
    workers = [
        IngestWorker.remote(
            embed_model_path=args.embed_model,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            embed_batch=args.embed_batch,
            upload_batch=args.upload_batch,
        )
        for _ in range(args.num_workers)
    ]

    # Crude round-robin scheduling
    obj_refs = []
    for i, f in enumerate(files):
        w = workers[i % len(workers)]
        meta = catalog_index.get(extract_book_id(f.name) or "", {})
        obj_refs.append((f.name, w.process_file.remote(str(f), meta)))

    # Poll results and show progress per file
    start = time.time()
    pbar = tqdm(total=len(files), desc="files")
    total_chunks = 0
    total_embed_time = 0.0
    total_upload_time = 0.0

    remaining: List[Tuple[str, "ObjectRef"]] = obj_refs[:]
    while remaining:
        done, remaining_refs = ray.wait([r for (_, r) in remaining], num_returns=1, timeout=1.0)
        if not done:
            # Optional: poll Qdrant counts here to estimate ETA
            time.sleep(0.2)
            continue
        # Reconcile file stats from worker
        idx = [r for (_, r) in remaining].index(done[0])
        fname, _ = remaining.pop(idx)

        result = ray.get(done[0])
        total_chunks += result["chunks"]
        total_embed_time += result["embed_time"]
        total_upload_time += result["upload_time"]

        # Update progress actor state
        progress.inc_file.remote()
        progress.add_chunks.remote(result["chunks"])
        progress.add_embed_time.remote(result["embed_time"])
        progress.add_upload_time.remote(result["upload_time"])

        pbar.set_postfix({
            "chunks": total_chunks,
            "emb/s": f"{(result['chunks']/result['embed_time']) if result['embed_time']>0 else 0:.1f}",
            "up/s": f"{(result['chunks']/result['upload_time']) if result['upload_time']>0 else 0:.1f}",
        })
        pbar.update(1)

    pbar.close()
    total_time = time.time() - start

    snap = ray.get(progress.snapshot.remote())
    print("\n" + "="*60)
    print("📊 Summary (Ray parallel)")
    print("="*60)
    print(f"  Files processed:    {snap['files_done']}/{snap['total_files']}")
    print(f"  Total chunks:       {total_chunks}")
    print(f"  Total time:         {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"  Embedding time:     {total_embed_time:.2f}s ({(total_chunks/total_embed_time) if total_embed_time>0 else 0:.1f} chunks/s)")
    print(f"  Upload time:        {total_upload_time:.2f}s ({(total_chunks/total_upload_time) if total_upload_time>0 else 0:.1f} chunks/s)")
    print(f"  Overall throughput: {(total_chunks/total_time) if total_time>0 else 0:.1f} chunks/s")
    print("="*60)

    # Optional JSON summary
    stats = {
        "files_ingested": snap["files_done"],
        "total_chunks": total_chunks,
        "total_time_seconds": round(total_time, 2),
        "embedding_time_seconds": round(total_embed_time, 2),
        "upload_time_seconds": round(total_upload_time, 2),
        "throughput_chunks_per_second": round((total_chunks/total_time), 2) if total_time>0 else 0,
    }
    print("\n📄 JSON Output:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
