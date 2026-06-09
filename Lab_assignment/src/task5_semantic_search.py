"""
Task 5 - Semantic search over the local vector store created in Task 4.

The search module reads data/index/vector_store.jsonl, embeds the query with the
same Task 4 embedding helper, then ranks chunks by cosine similarity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from Lab_assignment.src.task4_chunking_indexing import (
    INDEX_DIR,
    VECTOR_STORE_PATH,
    embed_texts,
    run_pipeline,
)


def load_vector_store() -> list[dict]:
    """Load embedded chunks from the Task 4 local JSONL index."""
    if not VECTOR_STORE_PATH.exists():
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        run_pipeline()

    chunks: list[dict] = []
    with VECTOR_STORE_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("content") and record.get("embedding"):
                chunks.append(record)
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity for two dense vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search semantically similar chunks using vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    chunks = load_vector_store()
    if not chunks:
        return []

    query_embedding, backend = embed_texts([query])
    query_vector = query_embedding[0]

    results: list[dict] = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk["embedding"])
        metadata = dict(chunk.get("metadata", {}))
        metadata["query_embedding_backend"] = backend
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": metadata,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}: {result['content'][:100]}...")
