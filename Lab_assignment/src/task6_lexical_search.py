"""
Task 6 - Lexical search with TF-IDF cosine similarity.

Why TF-IDF instead of BM25:
    TF-IDF is a strong, lightweight lexical baseline that runs fully local and
    does not need an external search engine. It is appropriate for this project
    because users often ask legal/news queries containing exact Vietnamese terms
    such as "ma túy", "tàng trữ", "tiền chất", or article numbers. Exact term
    overlap should therefore strongly affect ranking.

How TF-IDF works:
    - TF (term frequency): words appearing in a chunk increase that chunk's
      vector weight for those words.
    - IDF (inverse document frequency): rare words across the corpus get higher
      weight than common words.
    - Each chunk and query is represented as a sparse TF-IDF vector.
    - Ranking uses cosine similarity between the query vector and each chunk
      vector. Higher cosine means stronger lexical overlap after IDF weighting.

Tradeoff vs BM25:
    BM25 usually handles document length saturation better. TF-IDF is simpler,
    deterministic, easy to inspect, and works well here because Task 4 already
    normalizes document length into similarly sized chunks.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer

from Lab_assignment.src.task4_chunking_indexing import (
    VECTOR_STORE_PATH,
    chunk_documents,
    load_documents,
    run_pipeline,
)


SEARCH_METHOD = "tfidf_cosine"


def vietnamese_tokenizer(text: str) -> list[str]:
    """Simple Unicode-aware tokenizer for Vietnamese lexical search."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def load_corpus() -> list[dict]:
    """
    Load retrieval chunks from Task 4's local vector store.

    Falls back to chunking standardized documents if the vector store has not
    been created yet.
    """
    if not VECTOR_STORE_PATH.exists():
        run_pipeline()

    corpus: list[dict] = []
    if VECTOR_STORE_PATH.exists():
        with VECTOR_STORE_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("content"):
                    corpus.append(
                        {
                            "content": record["content"],
                            "metadata": record.get("metadata", {}),
                        }
                    )

    if corpus:
        return corpus

    return chunk_documents(load_documents())


@lru_cache(maxsize=1)
def build_tfidf_index() -> tuple[list[dict], TfidfVectorizer, object]:
    """Build and cache the TF-IDF matrix for all chunks."""
    corpus = load_corpus()
    texts = [item["content"] for item in corpus]
    vectorizer = TfidfVectorizer(
        tokenizer=vietnamese_tokenizer,
        token_pattern=None,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts) if texts else None
    return corpus, vectorizer, matrix


def build_bm25_index(corpus: list[dict]):
    """
    Compatibility wrapper for older notebook/test code.

    This project intentionally uses TF-IDF cosine rather than BM25. The returned
    value is the same TF-IDF index object used by lexical_search().
    """
    texts = [item["content"] for item in corpus]
    vectorizer = TfidfVectorizer(
        tokenizer=vietnamese_tokenizer,
        token_pattern=None,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts) if texts else None
    return vectorizer, matrix


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks lexically using TF-IDF cosine similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    corpus, vectorizer, matrix = build_tfidf_index()
    if not corpus or matrix is None:
        return []

    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    ranked_indices = scores.argsort()[::-1]

    results: list[dict] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            break
        item = corpus[index]
        metadata = dict(item.get("metadata", {}))
        metadata["lexical_method"] = SEARCH_METHOD
        results.append(
            {
                "content": item["content"],
                "score": score,
                "metadata": metadata,
            }
        )
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}: {result['content'][:100]}...")
