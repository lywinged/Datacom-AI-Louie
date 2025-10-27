"""
Utilities to build lightweight metadata indexes for author/title lookups.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

from backend.config.settings import settings
from backend.services.qdrant_client import get_qdrant_client

_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    return _TOKEN_PATTERN.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class MetadataEntry:
    point_id: str
    content: str
    source: str
    authors: str | None
    title: str | None
    score: float = 1.0

    @property
    def normalized_title(self) -> str:
        return _normalize(self.title or "")


def _build_metadata_entries() -> List[MetadataEntry]:
    client = get_qdrant_client()
    entries: List[MetadataEntry] = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1024,
            with_payload=[
                "text",
                "content",
                "source",
                "title",
                "authors",
                "metadata",
            ],
            offset=offset,
        )

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            content = (
                payload.get("text")
                or payload.get("content")
                or metadata.get("content")
                or ""
            )
            if not content:
                continue

            entry = MetadataEntry(
                point_id=str(point.id),
                content=content,
                source=payload.get("source") or metadata.get("source") or "",
                authors=payload.get("authors") or metadata.get("authors"),
                title=payload.get("title") or metadata.get("title"),
            )
            entries.append(entry)

        if offset is None:
            break

    return entries


@lru_cache(maxsize=1)
def _title_index() -> Dict[str, List[MetadataEntry]]:
    index: Dict[str, List[MetadataEntry]] = {}
    for entry in _build_metadata_entries():
        key = entry.normalized_title
        if not key:
            continue
        index.setdefault(key, []).append(entry)
    return index


def _title_candidates(query: str) -> Iterable[Tuple[MetadataEntry, float]]:
    normalized_query = _normalize(query)
    if not normalized_query:
        return []

    index = _title_index()
    scored: List[Tuple[MetadataEntry, float]] = []

    for title_key, entries in index.items():
        if not title_key:
            continue

        if normalized_query in title_key or title_key in normalized_query:
            score = 1.0
        else:
            score = SequenceMatcher(None, normalized_query, title_key).ratio()

        if score <= 0.2:
            continue

        for entry in entries:
            scored.append((entry, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def search_by_title(query: str, limit: int = 5) -> List[MetadataEntry]:
    candidates = _title_candidates(query)
    return [entry for entry, _ in candidates][:limit]


def warm_metadata_index() -> None:
    _ = _title_index()
