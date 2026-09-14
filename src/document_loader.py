"""Load mixed corpus files into source-tagged plain-text documents."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm"}


@dataclass(frozen=True)
class Document:
    """Text extracted from one source file."""

    source: str
    text: str


def load_text(path: Path) -> str:
    """Extract plain text from a supported file."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

    raise ValueError(f"unsupported format: {suffix or '[no extension]'}")


def load_documents(directory: Path) -> list[Document]:
    """Load supported files below *directory*, skipping files that fail."""
    documents: list[Document] = []

    if not directory.exists():
        print(f"SKIP {directory}: directory does not exist")
        return documents
    if not directory.is_dir():
        print(f"SKIP {directory}: not a directory")
        return documents

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        try:
            text = load_text(path)
        except Exception as error:
            print(f"SKIP {path.name}: {error}")
            continue

        document = Document(source=path.name, text=text)
        documents.append(document)
        print(f"OK {document.source}: {len(document.text)} chars | {document.text[:60]!r}")

    return documents


def load_paths(paths: Iterable[Path]) -> list[Document]:
    """Load individual paths using the same failure handling as directory intake."""
    documents: list[Document] = []

    for path in paths:
        try:
            text = load_text(path)
        except Exception as error:
            print(f"SKIP {path}: {error}")
            continue

        document = Document(source=path.name, text=text)
        documents.append(document)
        print(f"OK {document.source}: {len(document.text)} chars | {document.text[:60]!r}")

    return documents


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a mixed document corpus")
    parser.add_argument("directory", type=Path, help="directory containing corpus files")
    args = parser.parse_args()
    load_documents(args.directory)
