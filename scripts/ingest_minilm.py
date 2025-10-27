#!/usr/bin/env python3
"""
MiniLM embedding ingestion script with real-time progress updates.

Usage (inside Docker container):
    docker exec -it backend-api python3 /app/ingest_minilm.py
"""

import sys
import time
from pathlib import Path

# Ensure unbuffered output for real-time progress
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from backend.services.onnx_inference import ONNXEmbeddingModel
from backend.services.qdrant_client import get_qdrant_client
from backend.utils.file_loader import load_document_from_path
from backend.utils.text_splitter import split_text
from backend.config.settings import settings
from qdrant_client.http import models as qdrant_models


def main():
    print('=' * 70, flush=True)
    print('🚀 MiniLM 入库 - 实时进度显示', flush=True)
    print('=' * 70, flush=True)

    # 1. Load model
    print('\n📦 步骤 1/3: 加载 MiniLM embedding 模型...', flush=True)
    model_start = time.time()
    model = ONNXEmbeddingModel('/app/models/minilm-embed-int8')
    model_time = time.time() - model_start
    print(f'   ✅ 模型加载完成 ({model_time:.1f}秒, 向量维度={model.vector_size})', flush=True)

    # 2. Get client
    print('\n🔌 步骤 2/3: 连接 Qdrant...', flush=True)
    client = get_qdrant_client()
    collection_name = 'assessment_docs_minilm'
    print(f'   ✅ 已连接到 collection: {collection_name}', flush=True)

    # 3. Get files
    docs_dir = Path('/app/data/assessment_docs')
    files = sorted(docs_dir.glob('*.txt'))
    total_files = len(files)

    print(f'\n📚 步骤 3/3: 处理文档并入库...', flush=True)
    print(f'   文件总数: {total_files}', flush=True)
    print('=' * 70, flush=True)
    print(f'{"进度":<8} {"文件":<6} {"向量数":<10} {"速度":<12} {"预计剩余"}', flush=True)
    print('-' * 70, flush=True)

    total_chunks = 0
    start_time = time.time()
    last_print_time = start_time

    for file_idx, path in enumerate(files, 1):
        file_start = time.time()

        # Load and process
        documents = load_document_from_path(str(path))
        all_points = []

        for doc_idx, doc in enumerate(documents):
            chunks = split_text(
                doc.content,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            if not chunks:
                continue

            vectors = model.encode(chunks, batch_size=64).astype('float32')

            for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                point_id = hash(f'{path.name}-{doc_idx}-{idx}') & 0xFFFFFFFFFFFFFFFF
                all_points.append(
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            'document_id': hash(path.name) & 0xFFFFFFFFFFFF,
                            'chunk_index': idx,
                            'title': doc.title or path.stem,
                            'source': doc.source or path.name,
                            'text': chunk,
                        }
                    )
                )

        # Upload to Qdrant
        if all_points:
            client.upsert(collection_name=collection_name, points=all_points)
            total_chunks += len(all_points)

        # Print progress (every file, with time throttling for readability)
        current_time = time.time()
        elapsed = current_time - start_time

        # Print every 2 seconds or at key milestones
        should_print = (
            current_time - last_print_time >= 2.0 or
            file_idx == 1 or
            file_idx % 10 == 0 or
            file_idx in [25, 50, 75, 100, 125, 140] or
            file_idx == total_files
        )

        if should_print:
            rate = total_chunks / elapsed if elapsed > 0 else 0
            percent = (file_idx / total_files) * 100
            eta_sec = (total_files - file_idx) / (file_idx / elapsed) if file_idx > 0 and elapsed > 0 else 0

            progress_bar = '█' * int(percent / 5) + '░' * (20 - int(percent / 5))

            print(
                f'{progress_bar} '
                f'{percent:>5.1f}% '
                f'{file_idx:>3}/{total_files:<3} '
                f'{total_chunks:>8,} '
                f'{rate:>6.0f} c/s   '
                f'{eta_sec/60:>5.1f} 分钟',
                flush=True
            )
            last_print_time = current_time

    # Final summary
    total_time = time.time() - start_time
    print('=' * 70, flush=True)
    print(f'\n✅ 入库完成！', flush=True)
    print(f'   文件数:   {total_files}', flush=True)
    print(f'   向量数:   {total_chunks:,}', flush=True)
    print(f'   总耗时:   {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)', flush=True)
    print(f'   平均速度: {total_chunks/total_time:.0f} chunks/秒', flush=True)
    print('=' * 70, flush=True)


if __name__ == '__main__':
    main()
