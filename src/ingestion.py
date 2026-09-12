"""Document loading, text extraction and chunking.

Handles PDF, TXT and Markdown. The key responsibility beyond extraction is
attaching source metadata (filename + page) to every chunk, since that
metadata is what makes citations possible downstream.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension is outside SUPPORTED_EXTENSIONS."""


def load_document(path: Path) -> list[Document]:
    """Extract text from a single file into one Document per page/file.

    PDFs yield one Document per page, which preserves page numbers for
    citation. Text and Markdown files have no pages, so they yield a single
    Document and cite by filename alone.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"{path.name}: '{suffix}' is not supported. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        # Fall back through encodings rather than crashing on a stray byte.
        try:
            docs = TextLoader(str(path), encoding="utf-8").load()
        except UnicodeDecodeError:
            logger.warning("%s: not valid UTF-8, retrying as latin-1", path.name)
            docs = TextLoader(str(path), encoding="latin-1").load()

    # Normalise metadata so downstream code has one shape to deal with.
    for doc in docs:
        doc.metadata["source"] = path.name
        if suffix == ".pdf":
            # PyPDFLoader pages are 0-indexed; humans count from 1.
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        else:
            doc.metadata["page"] = None

    non_empty = [d for d in docs if d.page_content.strip()]
    if not non_empty:
        logger.warning(
            "%s: no extractable text found (scanned image PDF?)", path.name
        )

    logger.info("Loaded %s -> %d page(s) with text", path.name, len(non_empty))
    return non_empty


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks, preserving source metadata."""
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    # A stable id per chunk helps with debugging and deduplication.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    logger.info("Split %d document(s) into %d chunk(s)", len(docs), len(chunks))
    return chunks


def ingest_paths(paths: list[Path]) -> list[Document]:
    """Load and chunk several files, skipping any that fail.

    One unreadable file should not abandon an otherwise good batch, so
    failures are logged and collected rather than raised.
    """
    all_docs: list[Document] = []
    failures: list[str] = []

    for path in paths:
        try:
            all_docs.extend(load_document(Path(path)))
        except (UnsupportedFileTypeError, FileNotFoundError) as exc:
            logger.error("Skipping %s: %s", path, exc)
            failures.append(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface any loader failure
            logger.exception("Unexpected error loading %s", path)
            failures.append(f"{Path(path).name}: {exc}")

    if failures and not all_docs:
        raise RuntimeError(
            "No documents could be loaded:\n" + "\n".join(failures)
        )

    return chunk_documents(all_docs)
