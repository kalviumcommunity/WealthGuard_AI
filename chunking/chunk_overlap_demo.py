import tiktoken

from token_chunker import chunk_text, count_tokens


ENCODING = tiktoken.get_encoding("o200k_base")


def split_without_overlap(text, chunk_size):
    tokens = ENCODING.encode(text)
    chunks = []

    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start:start + chunk_size]
        chunks.append(ENCODING.decode(chunk))

    return chunks


def main():
    boundary_text = (
        "The Secure Growth Plan requires careful review of its "
        "withdrawal conditions. Early withdrawal is subject to "
        "applicable terms and conditions, and customers should "
        "review the approved product documentation before making "
        "decisions."
    )

    chunk_size = 20

    without_overlap = split_without_overlap(
        boundary_text,
        chunk_size,
    )

    with_overlap = chunk_text(
        boundary_text,
        chunk_size=chunk_size,
        overlap=8,
    )

    print("BOUNDARY OVERLAP DEMONSTRATION")
    print("=" * 60)

    print("\nWITHOUT OVERLAP")
    print("-" * 60)

    for i, chunk in enumerate(without_overlap, 1):
        print(f"Chunk {i} ({count_tokens(chunk)} tokens):")
        print(chunk.strip())

    print("\nWITH 8-TOKEN OVERLAP")
    print("-" * 60)

    for i, chunk in enumerate(with_overlap, 1):
        print(f"Chunk {i} ({count_tokens(chunk)} tokens):")
        print(chunk.strip())


if __name__ == "__main__":
    main()