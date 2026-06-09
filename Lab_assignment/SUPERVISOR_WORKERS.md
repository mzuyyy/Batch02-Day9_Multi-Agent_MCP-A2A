# Supervisor - Workers Pattern

`Lab_assignment/src/supervisor_workers.py` adds a Supervisor - Workers layer on
top of the existing RAG tasks.

## Workers

- `SemanticSearchWorker`: wraps Task 5 dense semantic search.
- `LexicalSearchWorker`: wraps Task 6 TF-IDF lexical search.
- `RerankWorker`: wraps Task 7 reranking and final top-k selection.
- `PageIndexFallbackWorker`: wraps Task 8 vectorless fallback.

## Supervisor Flow

1. `RetrievalSupervisor` receives a `RetrievalRequest`.
2. It calls semantic and lexical workers.
3. It merges ranked lists with reciprocal rank fusion.
4. It asks the rerank worker for final ordering.
5. If no result is found or the best score is below `score_threshold`, it routes
   to the PageIndex fallback worker.

`Task 9` now delegates to this supervisor through `retrieve_with_supervisor()`,
so old code can still call `retrieve(...)` without changing imports.
