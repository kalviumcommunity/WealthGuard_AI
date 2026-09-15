"""Create source-traceable chunks for retrieval and citation."""

import argparse
import re
from pathlib import Path
from typing import Iterable, Iterator

from src.document_loader import Document, load_documents


ChunkPart = tuple[str, int]


def split_text(text: str, chunk_size: int = 240, overlap: int = 40) -> list[ChunkPart]:
    """Split text into overlapping character windows with their source offsets."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    chunks: list[ChunkPart] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            leading_whitespace = len(text[start:end]) - len(text[start:end].lstrip())
            chunks.append((chunk, start + leading_whitespace))
        if end == len(text):
            break
        start += step

    return chunks


def _section_at(text: str, position: int) -> str | None:
    """Return the latest Markdown heading before a chunk, if one exists."""
    section: str | None = None
    for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", text):
        if match.start() > position:
            break
        section = match.group(1).strip()
    return section


def tag_chunks(
    source: str,
    chunks: Iterable[ChunkPart],
    *,
    full_text: str = "",
    page: int | None = None,
) -> list[dict[str, object]]:
    """Attach consistent citation metadata to chunk text and offsets."""
    tagged: list[dict[str, object]] = []

    for chunk_index, (text, char_start) in enumerate(chunks):
        tagged.append(
            {
                "text": text,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index,
                    "char_start": char_start,
                    "char_end": char_start + len(text),
                    "section": _section_at(full_text, char_start) if full_text else None,
                    "page": page,
                },
            }
        )

    return tagged


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 240,
    overlap: int = 40,
    page: int | None = None,
) -> list[dict[str, object]]:
    """Chunk a loaded document and preserve metadata needed for citations."""
    chunks = split_text(document.text, chunk_size=chunk_size, overlap=overlap)
    return tag_chunks(document.source, chunks, full_text=document.text, page=page)


def trace_chunk(chunk: dict[str, object]) -> str:
    """Format the source location carried by a retrieved chunk."""
    metadata = chunk["metadata"]
    return (
        f"Answer from: {metadata['source']} | "
        f"chunk {metadata['chunk_index']} | "
        f"characters {metadata['char_start']}-{metadata['char_end']} | "
        f"section: {metadata['section'] or 'not available'}"
    )


def _iter_chunks(directory: Path) -> Iterator[dict[str, object]]:
    for document in load_documents(directory):
        yield from chunk_document(document)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create traceable metadata chunks")
    parser.add_argument("directory", type=Path, help="directory containing corpus files")
    args = parser.parse_args()

    chunks = list(_iter_chunks(args.directory))
    print(f"Created {len(chunks)} chunks")
    if chunks:
        print(chunks[0])
        print(trace_chunk(chunks[0]))
