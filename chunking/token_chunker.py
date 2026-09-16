import tiktoken


# Tokenizer used for token-aware chunking.
ENCODING = tiktoken.get_encoding("o200k_base")

# Chosen RAG chunk settings.
CHUNK_SIZE = 256
OVERLAP = 40


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into token-sized chunks with controlled overlap."""

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    tokens = ENCODING.encode(text)
    chunks = []

    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(ENCODING.decode(chunk_tokens))

        if end == len(tokens):
            break

        start = end - overlap

    return chunks


def count_tokens(text):
    """Return the number of tokens in text."""
    return len(ENCODING.encode(text))


if __name__ == "__main__":
    sample_text = """
WealthGuard AI provides evidence-based information for relationship
managers. Approved documents contain product features, applicable
fees, eligibility requirements, risk information, and compliance
guidelines. Relationship managers should rely on verified information
and avoid unsupported claims.

For the Secure Growth Plan, the annual management fee is 1.2 percent.
The minimum investment is fifty thousand rupees and the lock-in period
is three years. Capital protection is not guaranteed. Early withdrawal
is subject to applicable terms and conditions.

Tax treatment may depend on applicable regulations and customer
circumstances. If verified information is unavailable, the assistant
should clearly state that it does not have enough information to
answer reliably.
"""

    chunks = chunk_text(sample_text)

    print("TOKEN-AWARE CHUNKER")
    print("=" * 60)
    print(f"Tokenizer: o200k_base")
    print(f"Chunk size: {CHUNK_SIZE} tokens")
    print(f"Overlap: {OVERLAP} tokens")
    print(f"Total tokens: {count_tokens(sample_text)}")
    print(f"Number of chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print("\n" + "-" * 60)
        print(f"Chunk {index}")
        print(f"Token count: {count_tokens(chunk)}")
        print(chunk.strip())