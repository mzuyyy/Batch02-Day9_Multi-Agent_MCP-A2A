"""
Supervisor - Workers orchestration for the lab RAG pipeline.

The supervisor owns routing and quality-control decisions. Workers remain small
and focused: each worker wraps one capability from the original task modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = DEFAULT_TOP_K
    score_threshold: float = SCORE_THRESHOLD
    use_reranking: bool = True


@dataclass
class WorkerResult:
    worker_name: str
    results: list[dict]
    metadata: dict = field(default_factory=dict)


class RetrievalWorker(Protocol):
    name: str

    def run(self, request: RetrievalRequest, candidates: list[dict] | None = None) -> WorkerResult:
        """Execute the worker for a retrieval request."""


def normalize_results(results: list[dict], source: str) -> list[dict]:
    normalized = []
    for item in results:
        updated = dict(item)
        updated["metadata"] = dict(updated.get("metadata", {}))
        updated["metadata"]["retrieval_source"] = source
        normalized.append(updated)
    return normalized


class SemanticSearchWorker:
    name = "semantic_worker"

    def run(self, request: RetrievalRequest, candidates: list[dict] | None = None) -> WorkerResult:
        results = semantic_search(request.query, top_k=request.top_k * 3)
        return WorkerResult(
            worker_name=self.name,
            results=normalize_results(results, "semantic"),
            metadata={"strategy": "dense_vector_similarity"},
        )


class LexicalSearchWorker:
    name = "lexical_worker"

    def run(self, request: RetrievalRequest, candidates: list[dict] | None = None) -> WorkerResult:
        results = lexical_search(request.query, top_k=request.top_k * 3)
        return WorkerResult(
            worker_name=self.name,
            results=normalize_results(results, "lexical"),
            metadata={"strategy": "tfidf_cosine"},
        )


class RerankWorker:
    name = "rerank_worker"

    def run(self, request: RetrievalRequest, candidates: list[dict] | None = None) -> WorkerResult:
        candidates = candidates or []
        if not candidates:
            return WorkerResult(worker_name=self.name, results=[], metadata={"method": RERANK_METHOD})

        if request.use_reranking:
            results = rerank(request.query, candidates, top_k=request.top_k, method=RERANK_METHOD)
        else:
            results = candidates[: request.top_k]

        for item in results:
            item["source"] = "hybrid"

        return WorkerResult(
            worker_name=self.name,
            results=results,
            metadata={"method": RERANK_METHOD if request.use_reranking else "disabled"},
        )


class PageIndexFallbackWorker:
    name = "pageindex_fallback_worker"

    def run(self, request: RetrievalRequest, candidates: list[dict] | None = None) -> WorkerResult:
        return WorkerResult(
            worker_name=self.name,
            results=pageindex_search(request.query, top_k=request.top_k),
            metadata={"strategy": "vectorless_keyword_structure"},
        )


class RetrievalSupervisor:
    """
    Coordinates search workers and chooses the final retrieval path.

    Flow:
        1. Run semantic and lexical workers.
        2. Merge their ranked lists with RRF.
        3. Ask the rerank worker for final ordering.
        4. If confidence is too low, delegate to the fallback worker.
    """

    def __init__(
        self,
        search_workers: list[RetrievalWorker] | None = None,
        rerank_worker: RetrievalWorker | None = None,
        fallback_worker: RetrievalWorker | None = None,
    ) -> None:
        self.search_workers = search_workers or [SemanticSearchWorker(), LexicalSearchWorker()]
        self.rerank_worker = rerank_worker or RerankWorker()
        self.fallback_worker = fallback_worker or PageIndexFallbackWorker()

    def retrieve(self, request: RetrievalRequest) -> list[dict]:
        if not request.query.strip() or request.top_k <= 0:
            return []

        worker_outputs = [worker.run(request) for worker in self.search_workers]
        ranked_lists = [output.results for output in worker_outputs if output.results]
        merged = rerank_rrf(ranked_lists, top_k=request.top_k * 3) if ranked_lists else []
        for item in merged:
            item["source"] = "hybrid"

        final_output = self.rerank_worker.run(request, candidates=merged)
        final_results = final_output.results[: request.top_k]

        best_score = final_results[0]["score"] if final_results else 0.0
        if not final_results or best_score < request.score_threshold:
            return self.fallback_worker.run(request).results[: request.top_k]

        return final_results


DEFAULT_SUPERVISOR = RetrievalSupervisor()


def retrieve_with_supervisor(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    request = RetrievalRequest(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
    )
    return DEFAULT_SUPERVISOR.retrieve(request)
