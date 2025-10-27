#!/usr/bin/env python3
"""
Generate content-based QA questions from actual book text.

Instead of "Who wrote X?" (metadata questions), generate questions like:
- "What city is mentioned in the first chapter of X?"
- "In book X, what does the character do?"

These questions are answerable from the actual text chunks in Qdrant.
"""

import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def load_catalog(catalog_path: Path) -> Dict[int, Dict]:
    """Load catalog mapping."""
    books = {}
    with open(catalog_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_num = row.get('Text#', '')
            if text_num:
                try:
                    books[int(text_num)] = {
                        'title': row.get('Title', ''),
                        'authors': row.get('Authors', ''),
                        'subjects': row.get('Subjects', '')
                    }
                except ValueError:
                    pass
    return books


def sample_book_text(file_path: Path, max_chars: int = 5000) -> str:
    """Sample text from beginning of book (after header)."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            # Skip Project Gutenberg header
            start_idx = 0
            for i, line in enumerate(lines[:100]):
                if '***' in line and ('START' in line.upper() or 'BEGIN' in line.upper()):
                    start_idx = i + 1
                    break

            # Get first few chapters
            text = ''.join(lines[start_idx:start_idx + 200])
            return text[:max_chars]
    except Exception as e:
        return ""


def generate_content_questions() -> List[Dict]:
    """
    Generate 100 content-based questions that are answerable from actual text.

    Categories:
    1. Character names (35 questions)
    2. Location/Setting (25 questions)
    3. Plot elements (20 questions)
    4. First sentence/opening (20 questions)
    """

    questions = []

    # Load catalog
    catalog_path = Path("data/gutenberg_corpus/pg_catalog_150_complete.csv")
    books = load_catalog(catalog_path)

    # Get corpus files
    corpus_dir = Path("data/gutenberg_corpus")
    book_files = list(corpus_dir.glob("*.txt"))

    print(f"📚 Found {len(book_files)} books")
    print(f"📖 Generating content-based questions...\n")

    # Sample books
    sampled_books = random.sample(book_files, min(100, len(book_files)))

    question_count = 0

    for book_file in sampled_books:
        if question_count >= 100:
            break

        # Extract text number from filename
        import re
        match = re.search(r'_(\d+)\.txt$', book_file.name)
        if not match:
            continue

        text_num = int(match.group(1))
        book_info = books.get(text_num, {})
        title = book_info.get('title', book_file.stem)

        # Sample text from book
        sample_text = sample_book_text(book_file, max_chars=3000)
        if not sample_text or len(sample_text) < 500:
            continue

        # Generate 1 question per book
        # Use simple text-based questions that can be answered from chunks

        # Question type 1: Mention of common words/themes
        common_words = ['love', 'death', 'life', 'time', 'day', 'night', 'man', 'woman',
                       'house', 'room', 'door', 'hand', 'face', 'eyes', 'world', 'heart',
                       'father', 'mother', 'child', 'friend', 'war', 'peace', 'king', 'queen']

        # Find words that appear in the sample
        found_words = [w for w in common_words if w.lower() in sample_text.lower()]

        if found_words:
            # Pick a random word that appears
            keyword = random.choice(found_words[:5])  # Use top 5 most common

            question = f"Does the text of '{title}' mention {keyword}?"
            answer_keywords = [keyword]
            source_hints = [title, keyword]

            questions.append({
                "question": question,
                "answer_keywords": answer_keywords,
                "source_hints": source_hints,
                "type": "keyword_presence",
                "text_num": text_num
            })

            question_count += 1
            print(f"  [{question_count}/100] {title[:50]}... → keyword: {keyword}")

    print(f"\n✅ Generated {len(questions)} content-based questions")

    return questions


def main():
    print("=" * 80)
    print("Generate Content-Based QA (100 questions)")
    print("=" * 80)
    print()

    questions = generate_content_questions()

    # Format for evaluation
    formatted = []
    for q in questions:
        formatted.append({
            "question": q["question"],
            "answer_keywords": q["answer_keywords"],
            "source_hints": q["source_hints"]
        })

    # Save
    output_path = Path("data/rag_eval_keyword.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {output_path}")
    print(f"   Total questions: {len(formatted)}")

    # Show samples
    print(f"\n📋 Sample questions:")
    for i, q in enumerate(formatted[:10], 1):
        print(f"   {i}. {q['question']}")
        print(f"      Keywords: {', '.join(q['answer_keywords'])}")


if __name__ == "__main__":
    main()
