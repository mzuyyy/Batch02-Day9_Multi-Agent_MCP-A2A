"""Task 9 - Complete retrieval pipeline via Supervisor - Workers."""

from __future__ import annotations

from .supervisor_workers import (
    DEFAULT_TOP_K,
    RERANK_METHOD,
    SCORE_THRESHOLD,
    normalize_results,
    retrieve_with_supervisor,
)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Run hybrid retrieval with PageIndex/vectorless fallback.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
    """
    return retrieve_with_supervisor(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
    )


if __name__ == "__main__":
    for q in [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]:
        print(f"\nQuery: {q}")
        for i, result in enumerate(retrieve(q, top_k=3), 1):
            print(f"{i}. [{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
