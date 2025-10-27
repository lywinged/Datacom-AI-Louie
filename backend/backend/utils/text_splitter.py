"""
Simple text splitter used for RAG ingestion.
"""
from typing import List


def split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into overlapping chunks based on character count.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size // 4, 0)

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = end - chunk_overlap

    return chunks
