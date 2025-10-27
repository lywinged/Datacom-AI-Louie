"""
No-op metric helpers used to satisfy legacy imports from utils.rag.
"""
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def track_pgvector_query(*args, **kwargs) -> Iterator[None]:
    yield


@contextmanager
def track_embedding(*args, **kwargs) -> Iterator[None]:
    yield


@contextmanager
def track_rerank(*args, **kwargs) -> Iterator[None]:
    yield


def register_model_version(*args, **kwargs) -> None:  # pragma: no cover
    return


def track_embedding_tokens(*args, **kwargs) -> None:  # pragma: no cover
    return
