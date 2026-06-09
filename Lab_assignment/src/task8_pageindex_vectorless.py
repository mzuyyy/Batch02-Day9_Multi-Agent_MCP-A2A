"""
Task 8 - PageIndex/vectorless fallback.

The real PageIndex service needs an account and API key. For this coursework
repo, pageindex_search() provides a local vectorless fallback with the same
return contract. It reads Markdown documents and ranks sections by keyword and
structure overlap, so the rest of the RAG pipeline can run without cloud setup.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv

from Lab_assignment.src.task4_chunking_indexing import extract_document_title

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def split_sections(text: str, max_chars: int = 1000) -> list[str]:
    sections = re.split(r"\n(?=#|Điều\s+\d+|Chương\s+|Mục\s+)", text)
    output: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        while len(section) > max_chars:
            output.append(section[:max_chars].strip())
            section = section[max_chars:]
        if section:
            output.append(section)
    return output


def load_sections() -> list[dict]:
    sections: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in {p.lower() for p in md_file.parts} else "news"
        display_title = extract_document_title(content, md_file.stem, doc_type)
        for idx, section in enumerate(split_sections(content)):
            sections.append(
                {
                    "content": section,
                    "metadata": {
                        "source": md_file.name,
                        "display_title": display_title,
                        "path": md_file.relative_to(STANDARDIZED_DIR.parent.parent).as_posix(),
                        "type": doc_type,
                        "section_index": idx,
                    },
                }
            )
    return sections


def vectorless_score(query: str, content: str) -> float:
    query_tokens = Counter(tokenize(query))
    content_tokens = Counter(tokenize(content))
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = sum(min(count, content_tokens.get(token, 0)) for token, count in query_tokens.items())
    coverage = overlap / max(sum(query_tokens.values()), 1)
    phrase_bonus = 0.15 if query.lower() in content.lower() else 0.0
    heading_bonus = 0.05 if re.search(r"(^|\n)(#|Điều\s+\d+|Chương\s+)", content) else 0.0
    return coverage + phrase_bonus + heading_bonus


def upload_documents() -> int:
    """Local no-op upload that returns the number of Markdown files available."""
    return len(list(STANDARDIZED_DIR.rglob("*.md"))) if STANDARDIZED_DIR.exists() else 0


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval fallback with PageIndex-compatible output.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}.
    """
    if not query.strip() or top_k <= 0:
        return []

    scored = []
    for section in load_sections():
        score = vectorless_score(query, section["content"])
        if score <= 0:
            continue
        scored.append(
            {
                "content": section["content"],
                "score": float(score),
                "metadata": section["metadata"],
                "source": "pageindex",
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    for result in pageindex_search("hình phạt sử dụng ma tuý", top_k=3):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}: {result['content'][:100]}...")
