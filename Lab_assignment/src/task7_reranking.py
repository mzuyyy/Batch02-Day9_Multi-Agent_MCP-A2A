"""
Task 7 - Reranking module.

This implementation uses local, deterministic reranking methods:
    - RRF (Reciprocal Rank Fusion) for merging ranked lists.
    - A lightweight lexical reranker for the main rerank() interface.

RRF mechanism:
    RRF(d) = sum(1 / (k + rank_r(d))) across rankers. Documents that rank well
    in multiple systems get boosted without needing calibrated scores.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Local replacement for a cross-encoder API.

    It combines the incoming retrieval score with query-content lexical cosine.
    This keeps the interface API-compatible while avoiding external credentials.
    """
    query_counter = Counter(tokenize(query))
    reranked: list[dict] = []
    max_base = max((float(item.get("score", 0.0)) for item in candidates), default=1.0) or 1.0

    for item in candidates:
        content_counter = Counter(tokenize(item.get("content", "")))
        lexical_score = cosine_counter(query_counter, content_counter)
        base_score = float(item.get("score", 0.0)) / max_base
        rerank_score = 0.65 * lexical_score + 0.35 * base_score
        updated = dict(item)
        updated["score"] = float(rerank_score)
        metadata = dict(updated.get("metadata", {}))
        metadata["rerank_method"] = "local_lexical_cross_encoder_fallback"
        metadata["original_score"] = item.get("score", 0.0)
        updated["metadata"] = metadata
        reranked.append(updated)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance: relevance to query minus redundancy."""
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx = remaining[0]
        best_score = float("-inf")
        for idx in remaining:
            embedding = candidates[idx].get("embedding", [])
            relevance = cosine_sim(query_embedding, embedding)
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    cosine_sim(embedding, candidates[sel].get("embedding", []))
                    for sel in selected
                )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = dict(candidates[idx])
        item["score"] = float(item.get("score", 0.0))
        results.append(item)
    return results


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion for combining multiple ranked result lists."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("metadata", {}).get("chunk_id") or item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in items:
                items[key] = dict(item)

    results = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = dict(items[key])
        item["score"] = float(score)
        metadata = dict(item.get("metadata", {}))
        metadata["rerank_method"] = "rrf"
        item["metadata"] = metadata
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if not candidates or top_k <= 0:
        return []
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        return rerank_cross_encoder(query, candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    for result in rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
