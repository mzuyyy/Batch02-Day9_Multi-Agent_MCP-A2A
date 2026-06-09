"""
Task 10 - Generation with citation.

This file provides a local extractive generator so the RAG pipeline works even
without an LLM API key. It retrieves evidence, reorders chunks to reduce
"lost in the middle", formats sources, and composes a concise cited answer.
"""

from __future__ import annotations

import os
import re
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9  # If using an LLM, keep nucleus sampling focused but not deterministic.
TEMPERATURE = 0.3  # RAG answers should be factual, so low temperature is preferred.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
GREETING_RESPONSE = (
    "Chào bạn. Bạn có thể hỏi mình về Luật Phòng, chống ma túy, "
    "các nghị định/thông tư liên quan, danh mục chất ma túy hoặc nguồn tin đã crawl."
)


SYSTEM_PROMPT = """Answer the question in Vietnamese using only provided context.
Every factual claim must include a citation. If evidence is insufficient, say
'Tôi không thể xác minh thông tin này từ nguồn hiện có'."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to place high-value evidence at both the beginning and end.

    Example: [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2].
    """
    if len(chunks) <= 2:
        return list(chunks)

    reordered = list(chunks[0::2])
    reordered.extend(reversed(chunks[1::2]))
    return reordered


def source_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("display_title") or metadata.get("source") or f"Source {index}"
    return source


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with source labels for citation."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("display_title") or metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        score = chunk.get("score", 0.0)
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '').strip()}"
        )
    return "\n\n---\n\n".join(parts)


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40]


def query_terms(query: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", query.lower(), flags=re.UNICODE))


def is_greeting_or_smalltalk(text: str) -> bool:
    normalized = re.sub(r"[^\wÀ-ỹ\s]", " ", text.lower(), flags=re.UNICODE)
    tokens = set(normalized.split())
    compact = " ".join(normalized.split())
    greeting_phrases = {
        "chào",
        "xin chào",
        "hello",
        "hi",
        "hey",
        "alo",
        "chào bạn",
        "chào chatbot",
        "cảm ơn",
        "thank",
        "thanks",
    }
    domain_terms = {
        "ma",
        "túy",
        "tuý",
        "luật",
        "nghị",
        "định",
        "thông",
        "tư",
        "chất",
        "cấm",
        "hình",
        "phạt",
        "cai",
        "nghiện",
    }
    if tokens & domain_terms:
        return False
    return compact in greeting_phrases or (len(tokens) <= 3 and bool(tokens & greeting_phrases))


def best_evidence_sentences(query: str, chunks: list[dict], limit: int = 3) -> list[tuple[str, dict]]:
    terms = query_terms(query)
    evidence = []
    for chunk in chunks:
        for sentence in split_sentences(chunk.get("content", "")):
            tokens = query_terms(sentence)
            overlap = len(terms & tokens)
            if overlap:
                evidence.append((overlap + float(chunk.get("score", 0.0)), sentence, chunk))
    evidence.sort(key=lambda item: item[0], reverse=True)
    return [(sentence, chunk) for _, sentence, chunk in evidence[:limit]]


def generate_extractive_answer(query: str, chunks: list[dict]) -> str:
    evidence = best_evidence_sentences(query, chunks, limit=3)
    if not evidence:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_parts = []
    for idx, (sentence, chunk) in enumerate(evidence, start=1):
        citation = source_label(chunk, idx)
        answer_parts.append(f"{sentence} [{citation}]")
    return " ".join(answer_parts)


def generate_with_nvidia(query: str, chunks: list[dict]) -> str | None:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key or api_key in {"nvapi-xxx", "xxx"}:
        return None

    from openai import OpenAI

    context = format_context(chunks)
    user_message = f"""Context:
{context}

---

Question: {query}

Return the answer in Vietnamese. Use citations exactly from the Source labels
shown in the context, for example [Nghị định 57/2022/NĐ-CP]."""

    client = OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL)
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=900,
    )
    return response.choices[0].message.content or ""


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citations.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    if is_greeting_or_smalltalk(query):
        return {
            "answer": GREETING_RESPONSE,
            "sources": [],
            "retrieval_source": "none",
            "generation_model": "smalltalk_rule",
        }

    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    try:
        answer = generate_with_nvidia(query, reordered)
    except Exception as exc:
        answer = None
        print(f"NVIDIA generation unavailable, using extractive fallback: {exc}")

    if not answer:
        answer = generate_extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        "context": format_context(reordered),
        "generation_model": NVIDIA_MODEL
        if os.getenv("NVIDIA_API_KEY", "").strip() not in {"", "nvapi-xxx", "xxx"}
        else "extractive_fallback",
    }


if __name__ == "__main__":
    for q in [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]:
        result = generate_with_citation(q)
        print(f"\nQ: {q}\nA: {result['answer']}")
