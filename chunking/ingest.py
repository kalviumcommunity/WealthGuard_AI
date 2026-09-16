from pathlib import Path
import re
import json
from datetime import datetime


# -----------------------------
# 1. LOAD
# -----------------------------
def load_text(path):
    """
    Load text from a file.
    Supports .txt, .md, and common text-based files.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


# -----------------------------
# 2. CLEAN
# -----------------------------
def clean(text):
    """
    Basic text cleaning:
    - Normalize whitespace
    - Remove excessive blank lines
    """
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# -----------------------------
# 3. TOKEN-AWARE CHUNKING
# -----------------------------
def token_chunks(text, chunk_size=300, overlap=50):
    """
    Simple token-based chunking.
    chunk_size = maximum tokens per chunk
    overlap = tokens shared between consecutive chunks
    """

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


# -----------------------------
# 4. TAG CHUNKS
# -----------------------------
def tag_chunks(filename, chunks):
    """
    Add metadata to every chunk.
    """

    tagged = []

    for position, chunk in enumerate(chunks):
        tagged.append({
            "text": chunk,
            "metadata": {
                "source": filename,
                "chunk_position": position,
                "chunk_index": position + 1,
                "total_chunks": len(chunks)
            }
        })

    return tagged


# -----------------------------
# 5. FULL INGESTION PIPELINE
# -----------------------------
def ingest(folder):
    """
    Run load -> clean -> chunk -> tag
    over the complete corpus.
    """

    docs = 0
    chunks = []
    failures = []

    folder_path = Path(folder)

    files = [
        path
        for path in folder_path.rglob("*")
        if path.is_file()
    ]

    print("Starting ingestion...")
    print(f"Total source files found: {len(files)}")
    print("-" * 60)

    for number, path in enumerate(files, start=1):

        try:
            # Load
            text = load_text(path)

            # Clean
            cleaned_text = clean(text)

            # Chunk
            raw_chunks = token_chunks(
                cleaned_text,
                chunk_size=300,
                overlap=50
            )

            # Tag
            tagged = tag_chunks(
                path.name,
                raw_chunks
            )

            chunks.extend(tagged)

            docs += 1

            print(
                f"[{number}/{len(files)}] "
                f"SUCCESS: {path.name} "
                f"-> {len(tagged)} chunks"
            )

        except Exception as e:

            failures.append({
                "file": str(path),
                "error": str(e)
            })

            print(
                f"[{number}/{len(files)}] "
                f"FAILED: {path.name}"
            )

    return files, docs, chunks, failures


# -----------------------------
# 6. SAVE SAMPLE CHUNKS
# -----------------------------
def save_sample_chunks(chunks, filename="sample_chunks.json"):
    """
    Save first few chunks so they can be inspected.
    """

    samples = chunks[:5]

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            samples,
            file,
            indent=4,
            ensure_ascii=False
        )


# -----------------------------
# 7. SAVE FAILURES
# -----------------------------
def save_failures(failures, filename="ingestion_failures.json"):
    """
    Save failed documents and their errors.
    """

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            failures,
            file,
            indent=4,
            ensure_ascii=False
        )


# -----------------------------
# 8. SAVE MANIFEST
# -----------------------------
def save_manifest(files, docs, chunks, failures,
                  filename="ingestion_manifest.json"):

    manifest = {
        "run_time": datetime.now().isoformat(),
        "total_source_documents": len(files),
        "successfully_ingested_documents": docs,
        "total_chunks": len(chunks),
        "failed_documents": len(failures),
        "complete": docs + len(failures) == len(files)
    }

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=4
        )


# -----------------------------
# 9. VALIDATION
# -----------------------------
def validate_ingestion(files, docs, chunks, failures):

    total_files = len(files)
    successful = docs
    failed = len(failures)

    print("\n")
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    print(f"Total source documents : {total_files}")
    print(f"Successfully ingested : {successful}")
    print(f"Total chunks created   : {len(chunks)}")
    print(f"Failed documents       : {failed}")

    print("-" * 60)

    # Completeness validation
    accounted_for = successful + failed

    print(f"Documents accounted for: {accounted_for}")
    print(f"Expected documents     : {total_files}")

    assert (
        accounted_for == total_files
    ), "ERROR: A document was silently dropped!"

    print("VALIDATION PASSED")
    print("Every source document is accounted for.")

    # Failure details
    if failures:
        print("\nFAILED DOCUMENTS")
        print("-" * 60)

        for failure in failures:
            print("FILE :", failure["file"])
            print("ERROR:", failure["error"])
            print("-" * 60)

    else:
        print("\nNo ingestion failures detected.")


# -----------------------------
# 10. INSPECT SAMPLE CHUNKS
# -----------------------------
def inspect_samples(chunks):

    print("\n")
    print("=" * 60)
    print("SAMPLE CHUNKS")
    print("=" * 60)

    if not chunks:
        print("No chunks were created.")
        return

    for index, chunk in enumerate(chunks[:5], start=1):

        print(f"\nSample {index}")
        print("-" * 60)

        print("Text:")
        print(chunk["text"][:300])

        print("\nMetadata:")
        print(chunk["metadata"])


# -----------------------------
# 11. MAIN
# -----------------------------
if __name__ == "__main__":

    # Change this only if your corpus folder has another name.
    CORPUS_FOLDER = "data"

    files, docs, chunks, failures = ingest(
        CORPUS_FOLDER
    )

    # Validate that no document disappeared
    validate_ingestion(
        files,
        docs,
        chunks,
        failures
    )

    # Inspect sample chunks
    inspect_samples(chunks)

    # Save outputs
    save_sample_chunks(chunks)
    save_failures(failures)
    save_manifest(
        files,
        docs,
        chunks,
        failures
    )

    print("\n")
    print("=" * 60)
    print("OUTPUT FILES CREATED")
    print("=" * 60)

    print("1. sample_chunks.json")
    print("2. ingestion_failures.json")
    print("3. ingestion_manifest.json")

    print("\nIngestion pipeline completed successfully.")