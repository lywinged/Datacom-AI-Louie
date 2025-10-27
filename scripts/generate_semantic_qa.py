#!/usr/bin/env python3
"""Generate semantic QA questions from corpus sentences."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_catalog(catalog_path: Path) -> Dict[int, Dict[str, str]]:
    mapping: Dict[int, Dict[str, str]] = {}
    if not catalog_path.exists():
        return mapping
    import csv

    with catalog_path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text_num = row.get("Text#")
            if not text_num:
                continue
            try:
                num = int(text_num)
            except ValueError:
                continue
            mapping[num] = {
                "title": row.get("Title", ""),
                "authors": row.get("Authors", ""),
            }
    return mapping


def iter_books(limit: int | None = None) -> Iterable[Tuple[int | None, Path]]:
    corpus_dir = Path("data/gutenberg_corpus")
    files = sorted(corpus_dir.glob("*.txt"))
    if limit:
        files = random.sample(files, min(limit, len(files)))

    for file_path in files:
        match = re.search(r"_(\\d+)\\.txt$", file_path.name)
        text_num = int(match.group(1)) if match else None
        yield text_num, file_path


def read_book_text(path: Path, max_chars: int = 15000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read(max_chars * 2)
    except Exception:
        return ""

    # Remove Gutenberg header/footer heuristically
    start_idx = 0
    header_match = re.search(r"\*\*\*\s*START[^\n]*\n", text, re.IGNORECASE)
    if header_match:
        start_idx = header_match.end()

    text = text[start_idx:start_idx + max_chars]
    return text


def split_sentences(text: str) -> List[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 40]
    return sentences


def _split_on_keyword(sentence: str, keyword: str) -> Tuple[str, str] | None:
    pattern = re.compile(rf"\b{keyword}\b", re.IGNORECASE)
    match = pattern.search(sentence)
    if not match:
        return None
    before = sentence[: match.start()].strip()
    after = sentence[match.end() :].strip()
    if before and after:
        return before, after
    return None


def build_question(title: str, sentence: str) -> Tuple[str, str] | None:
    split = _split_on_keyword(sentence, "because")
    if split:
        cause, reason = split
        question = f"In '{title}', why {cause[0].lower() + cause[1:]}?"
        answer = reason.rstrip(". ")
        return question, answer

    split = _split_on_keyword(sentence, "so that")
    if split:
        before, after = split
        question = f"In '{title}', for what purpose {before[0].lower() + before[1:]}?"
        answer = after.rstrip(". ")
        return question, answer

    split = _split_on_keyword(sentence, "when")
    if split:
        before, after = split
        question = f"According to '{title}', what happens when {after.rstrip('. ')}?"
        answer = before.rstrip(". ")
        return question, answer

    return None


# ---------------- Keyword extraction for answers -----------------

# Minimal English stopwords; avoids heavy NLP deps.
STOPWORDS = {
    "the","a","an","and","or","but","if","when","while","so","that","as","of","to","in","on","for","by","with",
    "it","its","is","are","was","were","be","been","being","this","these","those","at","from","into","than",
    "not","no","nor","too","very","also","there","their","them","they","he","she","we","you","i","his","her",
    "him","us","our","your","my","mine","yours","ours","hers","theirs","who","whom","which","what","why","how",
}


def _tokens(text: str) -> List[str]:
    norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return [t for t in norm.split() if t]


def extract_keywords(answer_text: str, question_text: str = "", max_k: int = 4) -> List[str]:
    """Extract short, content-bearing keywords from the answer clause.

    - Lowercase, strip punctuation
    - Remove stopwords and tokens shorter than 3 chars
    - Exclude tokens that appear in the question to avoid trivial overlap
    - Return up to max_k distinct keywords, preferring longer tokens
    """
    a_toks = _tokens(answer_text)
    q_toks = set(_tokens(question_text)) if question_text else set()

    keep = [t for t in a_toks if len(t) >= 3 and t not in STOPWORDS and t not in q_toks]
    # Deduplicate but prefer longer tokens first
    keep = sorted(set(keep), key=lambda x: (-len(x), x))
    return keep[:max_k] if keep else a_toks[: min(max_k, len(a_toks))]


def make_aliases(answer: str) -> List[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()
    tokens = normalized.split()
    aliases = {answer.strip(), normalized}
    aliases.update(" ".join(tokens[i:j]) for i in range(len(tokens)) for j in range(i + 2, min(len(tokens) + 1, i + 5)))
    return sorted({a for a in aliases if a})


def generate_semantic_questions(target_count: int = 100) -> List[Dict[str, object]]:
    catalog = load_catalog(Path("data/gutenberg_corpus/pg_catalog_150_complete.csv"))
    questions: List[Dict[str, object]] = []

    for text_num, path in iter_books(limit=None):
        if len(questions) >= target_count:
            break

        title = catalog.get(text_num, {}).get("title") or path.stem
        text = read_book_text(path)
        sentences = split_sentences(text)

        random.shuffle(sentences)

        for sentence in sentences:
            if len(sentence) > 300:
                continue
            qa = build_question(title, sentence)
            if not qa:
                continue
            question, answer = qa
            if len(answer.split()) < 4:
                continue

            # Use shorter content keywords instead of the full clause, to avoid
            # trivial string contains and reward semantic ranking.
            kw = extract_keywords(answer, question_text=question, max_k=4)

            questions.append(
                {
                    "question": question,
                    "answer_keywords": kw,
                    "answer_aliases": make_aliases(answer),  # keep full clause in aliases
                    "source_hints": [title, sentence.strip()],
                }
            )
            break

    return questions[:target_count]


def main() -> None:
    random.seed(42)
    questions = generate_semantic_questions(100)
    output_path = Path("data/rag_eval_semantic.json")
    output_path.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generated {len(questions)} semantic questions -> {output_path}")

    for item in questions[:5]:
        print("-", item["question"])
        print("  Answer:", item["answer_keywords"][0])


if __name__ == "__main__":
    main()
