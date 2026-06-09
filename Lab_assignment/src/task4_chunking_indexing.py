"""
Task 4 - Chunking and indexing for the RAG corpus.

Design choice:
    - Chunking: RecursiveCharacterTextSplitter with Vietnamese legal separators.
      This is robust for mixed legal PDFs and crawled news where Markdown headings
      are not always clean after conversion.
    - Embedding: BAAI/bge-m3 by default because it is multilingual and strong for
      Vietnamese retrieval. If the model is not downloaded or the environment is
      offline, the script falls back to deterministic hashing embeddings so the
      local vector store and tests remain runnable.
    - Vector store: local JSONL files under data/index/. This avoids requiring a
      running Weaviate server while keeping a simple dense index for Task 5.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "index"
VECTOR_STORE_PATH = INDEX_DIR / "vector_store.jsonl"
MANIFEST_PATH = INDEX_DIR / "manifest.json"


# =============================================================================
# CONFIGURATION
# =============================================================================

# 800 chars keeps legal clauses focused; 150 chars overlap preserves context when
# article/section boundaries are split across chunks.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "recursive"

# bge-m3 is the best quality default for Vietnamese + legal/news multilingual
# retrieval. It outputs 1024-dimensional vectors.
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

# A local JSONL vector store is used for this coursework environment. It is easy
# to inspect and can be queried directly without Docker or cloud credentials.
VECTOR_STORE = "local_jsonl"

LEGAL_SEPARATORS = [
    "\n## ",
    "\n# ",
    "\nĐiều ",
    "\nKhoản ",
    "\nMục ",
    "\nChương ",
    "\n\n",
    "\n",
    ". ",
    "; ",
    ", ",
    " ",
    "",
]


def detect_doc_type(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_document_title(content: str, fallback: str, doc_type: str) -> str:
    """
    Extract a human-readable source title for citations and UI.

    Legal files often start with conversion metadata before the actual title, so
    prefer explicit legal document patterns over the Markdown filename heading.
    """
    lines = [compact_text(line.strip("# ").strip()) for line in content.splitlines()]
    lines = [line for line in lines if line and not line.startswith("**") and line != "---"]

    if doc_type == "news":
        return lines[0] if lines else fallback

    fallback_key = fallback.lower()
    stem_patterns = [
        (r"nghidinh-(\d+)-(\d+)", "Nghị định {number}/{year}/NĐ-CP"),
        (r"nghi-dinh-(\d+)-(\d+)", "Nghị định {number}/{year}/NĐ-CP"),
        (r"thongtu-(\d+)-(\d+)", "Thông tư {number}/{year}/TT-BYT"),
        (r"thong-tu-(\d+)-(\d+)", "Thông tư {number}/{year}/TT-BYT"),
        (r"luat73-2021|luat-73-2021", "Luật Phòng, chống ma túy 2021"),
    ]
    for pattern, template in stem_patterns:
        match = re.search(pattern, fallback_key)
        if match:
            if "{number}" in template:
                return template.format(number=match.group(1), year=match.group(2))
            return template

    text = compact_text("\n".join(lines[:80]))
    legal_patterns = [
        (r"Nghị\s*định\s+số\s+(\d+/\d+/NĐ-CP)", "Nghị định {number}"),
        (r"Thông\s*tư\s+số\s+(\d+/\d+/TT-[A-ZĐ]+)", "Thông tư {number}"),
        (r"Nghị\s*định.*?(\d+/\d+/NĐ-CP)", "Nghị định {number}"),
        (r"Thông\s*tư.*?(\d+/\d+/TT-[A-ZĐ]+)", "Thông tư {number}"),
        (r"(Luật\s+Phòng,\s*chống\s+ma\s+túy).*?(?:số\s+)?(73/2021/QH\d+)?", "Luật Phòng, chống ma túy 2021"),
    ]
    for pattern, template in legal_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            if "{number}" in template:
                return template.format(number=match.group(1))
            return template

    for line in lines:
        upper = line.upper()
        if upper.startswith(("LUẬT", "NGHỊ ĐỊNH", "THÔNG TƯ")):
            return line
    return fallback


def load_documents() -> list[dict]:
    """
    Read all Markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, ...}}
    """
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(PROJECT_DIR).as_posix()
        doc_type = detect_doc_type(md_file)
        display_title = extract_document_title(content, md_file.stem, doc_type)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "display_title": display_title,
                    "path": relative_path,
                    "type": doc_type,
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into retrieval chunks with legal-aware separators.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=LEGAL_SEPARATORS,
        length_function=len,
    )

    chunks: list[dict] = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for chunk_index, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_index,
                        "chunk_id": f"{doc['metadata']['path']}#{chunk_index}",
                        "chunking_method": CHUNKING_METHOD,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                }
            )

    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def hashing_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic fallback embedding used when bge-m3 is unavailable."""
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return normalize(vector)


def _embed_with_sentence_transformers(texts: list[str]) -> tuple[list[list[float]], str]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [embedding.tolist() for embedding in embeddings], "sentence_transformers"


def embed_texts(texts: list[str]) -> tuple[list[list[float]], str]:
    """
    Embed texts with bge-m3 by default, falling back to local hashing vectors.

    Set RAG_FORCE_HASH_EMBEDDINGS=1 to skip model loading for quick local tests.
    """
    if os.getenv("RAG_FORCE_HASH_EMBEDDINGS") == "1":
        return [hashing_embedding(text) for text in texts], "hashing_fallback"

    try:
        return _embed_with_sentence_transformers(texts)
    except Exception as exc:
        print(f"Embedding model unavailable, using hashing fallback: {exc}")
        return [hashing_embedding(text) for text in texts], "hashing_fallback"


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' vector to each chunk.

    Returns:
        The input chunks with embedding metadata attached.
    """
    if not chunks:
        return []

    embeddings, backend = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        chunk["metadata"] = {
            **chunk["metadata"],
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": len(embedding),
            "embedding_backend": backend,
        }
    return chunks


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """
    Save embedded chunks into a local JSONL vector store.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with VECTOR_STORE_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    manifest = {
        "vector_store": VECTOR_STORE,
        "vector_store_path": VECTOR_STORE_PATH.relative_to(PROJECT_DIR).as_posix(),
        "num_chunks": len(chunks),
        "chunking_method": CHUNKING_METHOD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return VECTOR_STORE_PATH


def run_pipeline() -> list[dict]:
    """Run the full load -> chunk -> embed -> index pipeline."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_path = index_to_vectorstore(chunks)
    print(f"Indexed to {index_path}")
    return chunks


if __name__ == "__main__":
    run_pipeline()
