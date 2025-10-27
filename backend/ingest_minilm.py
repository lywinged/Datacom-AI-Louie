#!/usr/bin/env python3
"""MiniLM embedding ingestion with real-time progress."""
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from backend.services.onnx_inference import ONNXEmbeddingModel
from backend.services.qdrant_client import get_qdrant_client
from backend.utils.file_loader import load_document_from_path
from backend.utils.text_splitter import split_text
from backend.config.settings import settings
from qdrant_client.http import models as qdrant_models

print('='*70, flush=True)
print('🚀 MiniLM 入库 - 实时进度', flush=True)
print('='*70, flush=True)

print('\n📦 加载模型...', flush=True)
model_start = time.time()
model = ONNXEmbeddingModel('/app/models/minilm-embed-int8')
print(f'   ✅ 完成 ({time.time()-model_start:.1f}秒, 维度={model.vector_size})\n', flush=True)

client = get_qdrant_client()
docs_dir = Path('/app/data/assessment_docs_minilm')
files = sorted(docs_dir.glob('*.txt'))
total_files = len(files)
start_index = int(os.getenv("INGEST_START_INDEX", "0"))

print(f'📚 文件总数: {total_files}', flush=True)
if start_index > 0:
    print(f'⏭️  跳过前 {start_index} 个文件（基于 INGEST_START_INDEX）', flush=True)
print('='*70, flush=True)
print(f'{"进度条":<22} {"完成%":<8} {"文件":<10} {"向量数":<10} {"速度":<10} {"剩余"}', flush=True)
print('-'*70, flush=True)

total_chunks = 0
start_time = time.time()
last_print = start_time

for file_idx, path in enumerate(files, 1):
    if file_idx <= start_index:
        continue

    documents = load_document_from_path(str(path))
    all_points = []
    
    for doc_idx, doc in enumerate(documents):
        chunks = split_text(doc.content, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        if not chunks:
            continue
        vectors = model.encode(chunks, batch_size=64).astype('float32')
        
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = hash(f'{path.name}-{doc_idx}-{idx}') & 0xFFFFFFFFFFFFFFFF
            all_points.append(qdrant_models.PointStruct(
                id=point_id, vector=vector.tolist(),
                payload={'document_id': hash(path.name) & 0xFFFFFFFFFFFF, 'chunk_index': idx,
                        'title': doc.title or path.stem, 'source': doc.source or path.name, 'text': chunk}
            ))
    
    if all_points:
        client.upsert(collection_name='assessment_docs_minilm', points=all_points)
        total_chunks += len(all_points)
    
    current = time.time()
    should_print = (current - last_print >= 3.0 or file_idx == 1 or file_idx % 10 == 0 or 
                   file_idx in [25, 50, 75, 100, 125] or file_idx == total_files)
    
    if should_print:
        elapsed = current - start_time
        rate = total_chunks / elapsed if elapsed > 0 else 0
        pct = (file_idx / total_files) * 100
        eta = (total_files - file_idx) / (file_idx / elapsed) if file_idx > 0 and elapsed > 0 else 0
        bar = '█' * int(pct/5) + '░' * (20-int(pct/5))
        print(f'{bar} {pct:>6.1f}% {file_idx:>3}/{total_files:<3} {total_chunks:>9,} {rate:>7.0f}/s  {eta/60:>5.1f}分', flush=True)
        last_print = current

elapsed = time.time() - start_time
print('='*70, flush=True)
print(f'\n✅ 入库完成！', flush=True)
print(f'   文件数: {total_files} | 向量数: {total_chunks:,} | 耗时: {elapsed/60:.1f}分 | 速度: {total_chunks/elapsed:.0f}/s', flush=True)
print('='*70, flush=True)
