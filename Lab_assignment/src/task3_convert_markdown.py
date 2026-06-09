"""
Task 3 - Convert files in data/landing/ to Markdown.

Outputs are written to data/standardized/ while preserving subdirectories such
as legal/ and news/. MarkItDown is used for binary/legal documents; crawled news
JSON is converted directly from its stored Markdown content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from markitdown import MarkItDown

try:
    import olefile
except Exception:  # pragma: no cover - optional .doc fallback dependency
    olefile = None


PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"

LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}
NEWS_EXTENSIONS = {".json", ".html", ".md", ".txt"}


def markdown_header(**metadata: str) -> str:
    lines = []
    title = metadata.pop("title", None)
    if title:
        lines.append(f"# {title}")
        lines.append("")

    for key, value in metadata.items():
        if value:
            label = key.replace("_", " ").title()
            lines.append(f"**{label}:** {value}")

    if lines:
        lines.extend(["", "---", ""])
    return "\n".join(lines)


def fallback_markdown(source_path: Path, error: Exception) -> str:
    return (
        markdown_header(
            title=source_path.stem,
            source_file=str(source_path.relative_to(PROJECT_DIR)),
            conversion_status="fallback",
        )
        + "MarkItDown could not extract this file in the current environment.\n\n"
        + f"Conversion error: `{type(error).__name__}: {error}`\n\n"
        + "This placeholder preserves the original document path for later manual "
        + "inspection. The original legal file remains in data/landing/legal/ and "
        + "can be reprocessed after installing any missing PDF/DOCX extraction "
        + "dependencies or OCR tooling.\n"
    )


def iter_files(directory: Path, extensions: set[str]) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_legacy_doc_text(source_path: Path) -> str:
    """Extract text from old OLE Word .doc files when MarkItDown cannot."""
    if source_path.suffix.lower() != ".doc" or olefile is None:
        return ""

    try:
        with olefile.OleFileIO(str(source_path)) as ole:
            if not ole.exists("WordDocument"):
                return ""
            data = ole.openstream("WordDocument").read()
    except Exception:
        return ""

    decoded = data.decode("utf-16le", errors="ignore")
    chunks = re.findall(
        r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9\s,.;:()/%\-–—“”\"'Đđ]+",
        decoded,
        flags=re.UNICODE,
    )
    useful_chunks = []
    for chunk in chunks:
        chunk = clean_extracted_text(chunk)
        if len(chunk) >= 20 and len(re.findall(r"[A-Za-zÀ-ỹ]", chunk)) >= 10:
            useful_chunks.append(chunk)

    return clean_extracted_text("\n\n".join(useful_chunks))


def convert_with_markitdown(md: MarkItDown, source_path: Path) -> str:
    try:
        result = md.convert(str(source_path))
        text = (getattr(result, "text_content", "") or "").strip()
        if not text:
            text = extract_legacy_doc_text(source_path)
        if not text:
            raise ValueError("MarkItDown returned empty text_content")
        return markdown_header(
            title=source_path.stem,
            source_file=str(source_path.relative_to(PROJECT_DIR)),
            conversion_status="converted",
        ) + text
    except Exception as exc:
        text = extract_legacy_doc_text(source_path)
        if text:
            return markdown_header(
                title=source_path.stem,
                source_file=str(source_path.relative_to(PROJECT_DIR)),
                conversion_status="converted_with_olefile_fallback",
            ) + text
        return fallback_markdown(source_path, exc)


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    saved: list[Path] = []

    source_files = list(iter_files(legal_dir, LEGAL_EXTENSIONS))
    expected_outputs = {f"{source_path.stem}.md" for source_path in source_files}
    for old_output in output_dir.glob("*.md"):
        if old_output.name not in expected_outputs:
            old_output.unlink()

    for source_path in source_files:
        print(f"Converting legal: {source_path.name}")
        content = convert_with_markitdown(md, source_path)
        output_path = output_dir / f"{source_path.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"  Saved: {output_path}")
        saved.append(output_path)

    return saved


def convert_news_json(source_path: Path) -> str:
    data = json.loads(source_path.read_text(encoding="utf-8"))
    header = markdown_header(
        title=data.get("title") or source_path.stem,
        url=data.get("url", ""),
        source=data.get("source", ""),
        published_date=data.get("published_date", ""),
        date_crawled=data.get("date_crawled", ""),
    )
    body = (
        data.get("content_markdown")
        or data.get("markdown")
        or data.get("content")
        or json.dumps(data, ensure_ascii=False, indent=2)
    )
    return header + body.strip() + "\n"


def convert_news_articles() -> list[Path]:
    """Convert crawled news files in data/landing/news/ to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    saved: list[Path] = []

    for source_path in iter_files(news_dir, NEWS_EXTENSIONS):
        print(f"Converting news: {source_path.name}")
        if source_path.suffix.lower() == ".json":
            content = convert_news_json(source_path)
        elif source_path.suffix.lower() == ".md":
            content = source_path.read_text(encoding="utf-8")
        else:
            content = convert_with_markitdown(md, source_path)

        output_path = output_dir / f"{source_path.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"  Saved: {output_path}")
        saved.append(output_path)

    return saved


def convert_all() -> list[Path]:
    """Convert all supported landing files to Markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_files = convert_legal_docs()

    print("\n--- News Articles ---")
    news_files = convert_news_articles()

    saved = legal_files + news_files
    print(f"\nDone. Saved {len(saved)} markdown files to {OUTPUT_DIR}")
    return saved


if __name__ == "__main__":
    convert_all()
