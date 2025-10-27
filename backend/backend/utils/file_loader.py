"""
Helper utilities to load documents from the data directory for ingestion.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


@dataclass
class LoadedDocument:
    title: str
    content: str
    source: str
    metadata: Dict[str, str]


def _strip_gutenberg_header(text: str) -> str:
    """Remove common Project Gutenberg license header/footer if present."""
    lower = text.lower()

    start_idx = 0
    start_marker = "*** start of"
    marker_pos = lower.find(start_marker)
    if marker_pos != -1:
        newline_pos = text.find("\n", marker_pos)
        start_idx = newline_pos + 1 if newline_pos != -1 else marker_pos

    end_idx = len(text)
    end_marker = "*** end of"
    marker_pos = lower.find(end_marker)
    if marker_pos != -1:
        end_idx = marker_pos

    cleaned = text[start_idx:end_idx].lstrip()

    # Remove additional catalog metadata lines (Title, Author, etc.)
    lines = cleaned.splitlines()
    filtered: List[str] = []
    skipping = True
    gutter_prefixes = (
        "the project gutenberg",
        "project gutenberg",
        "title:",
        "author:",
        "editor:",
        "release date:",
        "language:",
        "character set encoding:",
        "produced by",
        "etext prepared by",
        "credits:",
        "illustrator:",
    )
    for line in lines:
        stripped = line.strip()
        if skipping:
            if not stripped:
                continue
            lower_line = stripped.lower()
            if any(lower_line.startswith(prefix) for prefix in gutter_prefixes):
                continue
            skipping = False
        filtered.append(line)

    cleaned = "\n".join(filtered)
    return cleaned


def load_document_from_path(path: str) -> List[LoadedDocument]:
    """Load a text or PDF document and return metadata suitable for ingestion."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if file_path.suffix.lower() == ".txt":
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        text = _strip_gutenberg_header(raw_text)
        return [
            LoadedDocument(
                title=file_path.stem,
                content=text,
                source=file_path.name,
                metadata={"filename": file_path.name},
            )
        ]

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n".join(pages)
        text = _strip_gutenberg_header(text)
        return [
            LoadedDocument(
                title=file_path.stem,
                content=text,
                source=file_path.name,
                metadata={
                    "filename": file_path.name,
                    "pages": str(len(reader.pages)),
                },
            )
        ]

    raise ValueError(f"Unsupported document type: {path}")
