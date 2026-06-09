import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.supervisor_workers import RetrievalRequest, RetrievalSupervisor, WorkerResult


class FakeSearchWorker:
    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self.score = score

    def run(self, request: RetrievalRequest, candidates: list[dict] | None = None) -> WorkerResult:
        return WorkerResult(
            worker_name=self.name,
            results=[
                {
                    "content": f"{self.name} result",
                    "score": self.score,
                    "metadata": {"chunk_id": self.name},
                }
            ],
        )


class FakeRerankWorker:
    name = "fake_rerank_worker"

    def __init__(self, score: float) -> None:
        self.score = score

    def run(self, request: RetrievalRequest, candidates: list[dict] | None = None) -> WorkerResult:
        results = []
        for item in candidates or []:
            updated = dict(item)
            updated["score"] = self.score
            updated["source"] = "hybrid"
            results.append(updated)
        return WorkerResult(worker_name=self.name, results=results[: request.top_k])


class FakeFallbackWorker:
    name = "fake_fallback_worker"

    def run(self, request: RetrievalRequest, candidates: list[dict] | None = None) -> WorkerResult:
        return WorkerResult(
            worker_name=self.name,
            results=[
                {
                    "content": "fallback result",
                    "score": 1.0,
                    "metadata": {},
                    "source": "pageindex",
                }
            ],
        )


class TestSupervisorWorkers(unittest.TestCase):
    def test_supervisor_returns_hybrid_when_confident(self):
        supervisor = RetrievalSupervisor(
            search_workers=[FakeSearchWorker("semantic", 0.9), FakeSearchWorker("lexical", 0.7)],
            rerank_worker=FakeRerankWorker(score=0.8),
            fallback_worker=FakeFallbackWorker(),
        )

        results = supervisor.retrieve(RetrievalRequest("ma tuy", top_k=2, score_threshold=0.3))

        self.assertEqual(len(results), 2)
        self.assertTrue(all(item["source"] == "hybrid" for item in results))

    def test_supervisor_uses_fallback_when_confidence_is_low(self):
        supervisor = RetrievalSupervisor(
            search_workers=[FakeSearchWorker("semantic", 0.1)],
            rerank_worker=FakeRerankWorker(score=0.05),
            fallback_worker=FakeFallbackWorker(),
        )

        results = supervisor.retrieve(RetrievalRequest("unknown", top_k=1, score_threshold=0.3))

        self.assertEqual(results[0]["source"], "pageindex")
        self.assertEqual(results[0]["content"], "fallback result")


if __name__ == "__main__":
    unittest.main()
