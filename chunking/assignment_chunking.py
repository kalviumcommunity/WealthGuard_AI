"""
Document Chunking Strategies Assignment
========================================

This script demonstrates and compares different chunking strategies
for the WealthGuard AI corpus. It implements fixed-size chunking with
overlap and paragraph-based chunking, then compares their effectiveness.

Assignment Tasks:
1. Split using a defined strategy
2. Compare two strategies
3. Report chunk stats
4. Justify choice
5. Commit with sample chunks
"""

import os
from pathlib import Path


def fixed_chunks(text, size=200, overlap=30):
    """
    Fixed-size chunking with overlap.

    Args:
        text: Input text to chunk
        size: Target chunk size in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    i = 0

    while i < len(text):
        chunk = text[i:i + size]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        i += size - overlap

    return chunks


def paragraph_chunks(text):
    """
    Paragraph-based chunking.

    Splits text by double newlines (paragraph boundaries) and
    returns non-empty paragraphs.

    Args:
        text: Input text to chunk

    Returns:
        List of paragraph chunks
    """
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def sentence_chunks(text):
    """
    Sentence-based chunking.

    Splits text by sentence boundaries (period, question mark, exclamation).
    Simple approach that works for basic text.

    Args:
        text: Input text to chunk

    Returns:
        List of sentence chunks
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def average_chunk_size(chunks):
    """Calculate average character count of chunks."""
    if not chunks:
        return 0
    return sum(len(chunk) for chunk in chunks) / len(chunks)


def load_corpus(corpus_dir):
    """Load all text files from the corpus directory."""
    corpus_path = Path(corpus_dir)
    documents = {}

    for file_path in corpus_path.glob("*"):
        if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.html']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents[file_path.name] = content
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

    return documents


def print_separator(char="=", length=70):
    print(char * length)


def main():
    # Load corpus documents
    corpus_dir = Path(__file__).parent.parent / "data" / "sample_corpus"
    documents = load_corpus(corpus_dir)

    if not documents:
        print("No documents found in corpus directory.")
        return

    print_separator()
    print("DOCUMENT CHUNKING STRATEGIES ASSIGNMENT")
    print_separator()
    print(f"\nLoaded {len(documents)} documents from corpus:")
    for filename in documents.keys():
        print(f"  - {filename}")
    print()

    # Combine all documents for analysis
    combined_text = "\n\n".join(documents.values())
    print(f"Total corpus size: {len(combined_text)} characters")
    print()

    # -------------------------------------------------
    # TASK 1 & 2: Apply two chunking strategies
    # -------------------------------------------------

    print_separator()
    print("TASK 1 & 2: COMPARING CHUNKING STRATEGIES")
    print_separator()
    print()

    # Strategy 1: Fixed-size with overlap
    FIXED_SIZE = 200
    OVERLAP = 30

    fixed_chunks_list = fixed_chunks(combined_text, size=FIXED_SIZE, overlap=OVERLAP)

    # Strategy 2: Paragraph-based
    paragraph_chunks_list = paragraph_chunks(combined_text)

    # Strategy 3: Sentence-based (bonus comparison)
    sentence_chunks_list = sentence_chunks(combined_text)

    # -------------------------------------------------
    # TASK 3: Report chunk statistics
    # -------------------------------------------------

    print_separator()
    print("TASK 3: CHUNK STATISTICS")
    print_separator()
    print()

    strategies = [
        ("Fixed-size (200 chars, 30 overlap)", fixed_chunks_list),
        ("Paragraph-based", paragraph_chunks_list),
        ("Sentence-based", sentence_chunks_list)
    ]

    for name, chunks in strategies:
        chunk_count = len(chunks)
        avg_size = average_chunk_size(chunks)
        min_size = min(len(c) for c in chunks) if chunks else 0
        max_size = max(len(c) for c in chunks) if chunks else 0

        print(f"{name}:")
        print(f"  Chunk count: {chunk_count}")
        print(f"  Average size: {avg_size:.1f} characters")
        print(f"  Size range: {min_size} - {max_size} characters")
        print()

    # -------------------------------------------------
    # TASK 4: Justify chosen strategy
    # -------------------------------------------------

    print_separator()
    print("TASK 4: STRATEGY JUSTIFICATION")
    print_separator()
    print()

    print("CHOSEN STRATEGY: Paragraph-based chunking")
    print()
    print("JUSTIFICATION:")
    print("-" * 70)
    print()
    print("1. DOCUMENT STRUCTURE:")
    print("   The WealthGuard corpus consists of structured documents:")
    print("   - Policy documents with clear section breaks")
    print("   - Compliance notes as standalone paragraphs")
    print("   - Product information with distinct information blocks")
    print()
    print("2. MEANING PRESERVATION:")
    print("   - Paragraph boundaries naturally align with complete ideas")
    print("   - Each paragraph is a self-contained unit of information")
    print("   - Reduces risk of cutting mid-sentence or mid-concept")
    print()
    print("3. RETRIEVAL PRECISION:")
    print("   - Queries often target specific information blocks")
    print("   - Paragraph chunks provide sufficient context without noise")
    print("   - Clear boundaries make source attribution straightforward")
    print()
    print("4. CONTEXT WINDOW EFFICIENCY:")
    print("   - Average paragraph size (~40-60 chars for this corpus)")
    print("   - Allows multiple chunks in retrieval without overwhelming context")
    print("   - Enables combining top-k results effectively")
    print()
    print("5. TRADE-OFFS CONSIDERED:")
    print("   - Fixed-size: Uniform but may cut ideas in half")
    print("   - Sentence-based: Too granular, loses surrounding context")
    print("   - Paragraph: Balanced - preserves meaning with reasonable size")
    print()
    print("6. CORPUS-SUITABILITY:")
    print("   - Small corpus with well-structured documents")
    print("   - Each document type has natural paragraph structure")
    print("   - Paragraph chunking respects the document organization")
    print()

    # -------------------------------------------------
    # TASK 5: Sample chunks for review
    # -------------------------------------------------

    print_separator()
    print("TASK 5: SAMPLE CHUNKS FOR REVIEW")
    print_separator()
    print()

    print("FIXED-SIZE CHUNKS (first 3):")
    print("-" * 70)
    for i, chunk in enumerate(fixed_chunks_list[:3], 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)

    print("\n\n")
    print("PARAGRAPH CHUNKS (all):")
    print("-" * 70)
    for i, chunk in enumerate(paragraph_chunks_list, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk)

    print("\n\n")
    print("SENTENCE CHUNKS (first 3):")
    print("-" * 70)
    for i, chunk in enumerate(sentence_chunks_list[:3], 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk)

    # Save sample chunks to file
    output_file = Path(__file__).parent / "chunking_assignment_samples.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DOCUMENT CHUNKING ASSIGNMENT - SAMPLE CHUNKS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total corpus size: {len(combined_text)} characters\n")
        f.write(f"Documents: {list(documents.keys())}\n\n")

        f.write("STATISTICS:\n")
        f.write("-" * 70 + "\n")
        for name, chunks in strategies:
            chunk_count = len(chunks)
            avg_size = average_chunk_size(chunks)
            f.write(f"{name}:\n")
            f.write(f"  Chunk count: {chunk_count}\n")
            f.write(f"  Average size: {avg_size:.1f} characters\n\n")

        f.write("\nFIXED-SIZE CHUNKS:\n")
        f.write("-" * 70 + "\n")
        for i, chunk in enumerate(fixed_chunks_list, 1):
            f.write(f"\nChunk {i} ({len(chunk)} chars):\n")
            f.write(chunk + "\n")

        f.write("\n\nPARAGRAPH CHUNKS:\n")
        f.write("-" * 70 + "\n")
        for i, chunk in enumerate(paragraph_chunks_list, 1):
            f.write(f"\nChunk {i} ({len(chunk)} chars):\n")
            f.write(chunk + "\n")

        f.write("\n\nSENTENCE CHUNKS:\n")
        f.write("-" * 70 + "\n")
        for i, chunk in enumerate(sentence_chunks_list, 1):
            f.write(f"\nChunk {i} ({len(chunk)} chars):\n")
            f.write(chunk + "\n")

        f.write("\n\nJUSTIFICATION:\n")
        f.write("-" * 70 + "\n")
        f.write("Paragraph-based chunking is chosen because:\n")
        f.write("1. WealthGuard documents have clear paragraph structure\n")
        f.write("2. Each paragraph represents a complete idea\n")
        f.write("3. Preserves meaning without cutting concepts in half\n")
        f.write("4. Suitable size for retrieval within context window\n")
        f.write("5. Aligns with document organization and retrieval needs\n")

    print(f"\n\nSample chunks saved to: {output_file}")
    print()

    # -------------------------------------------------
    # Additional analysis: Chunk boundaries
    # -------------------------------------------------

    print_separator()
    print("BOUNDARY ANALYSIS")
    print_separator()
    print()

    print("How chunk boundaries affect answer quality:")
    print("-" * 70)
    print()
    print("SCENARIO: Query about 'management fee'")
    print()
    print("With fixed-size chunking:")
    print("  - If 'management fee' appears at chunk boundary, it may be split")
    print("  - Neither chunk contains the complete information")
    print("  - Retrieval may miss the relevant chunk entirely")
    print()
    print("With paragraph chunking:")
    print("  - 'The annual management fee is 1.2 percent' is a complete unit")
    print("  - The entire statement is preserved in one chunk")
    print("  - Retrieval can match and return the complete answer")
    print()
    print("SCENARIO: Query about 'lock-in period'")
    print()
    print("With fixed-size chunking:")
    print("  - HTML content may be chunked mid-tag or mid-sentence")
    print("  - Context about three-year period may be separated")
    print()
    print("With paragraph chunking:")
    print("  - Each <p> element becomes a complete chunk")
    print("  - 'The lock-in period is three years' is preserved intact")
    print("  - Clear boundary ensures complete information retrieval")
    print()

    # -------------------------------------------------
    # Context window relationship
    # -------------------------------------------------

    print_separator()
    print("CHUNK SIZE VS CONTEXT WINDOW")
    print_separator()
    print()

    print("Relationship between chunk size and context window:")
    print("-" * 70)
    print()
    print("1. FORMULA:")
    print("   (chunk_size × top_k) + query_tokens + response_tokens ≤ context_window")
    print()
    print("2. EXAMPLE:")
    print("   - Context window: 4096 tokens (typical for many models)")
    print("   - Chunk size: ~50 chars ≈ 12-15 tokens (paragraph)")
    print("   - Top-k: 5 chunks")
    print("   - Query: ~50 tokens")
    print("   - Response: ~200 tokens")
    print("   - Total: (15 × 5) + 50 + 200 = 325 tokens ✓ Well within budget")
    print()
    print("3. IF CHUNKS WERE LARGER (fixed-size 500 chars):")
    print("   - Chunk size: ~120 tokens")
    print("   - Top-k: 5 chunks")
    print("   - Total: (120 × 5) + 50 + 200 = 850 tokens")
    print("   - Still feasible, but fewer chunks can be retrieved")
    print()
    print("4. IMPLICATIONS:")
    print("   - Smaller chunks → more chunks in context, more diverse context")
    print("   - Larger chunks → fewer chunks in context, deeper but narrower")
    print("   - Optimal size balances precision with context richness")
    print("   - Must be tuned based on retrieval tests and query patterns")
    print()

    print_separator()
    print("ASSIGNMENT COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()
