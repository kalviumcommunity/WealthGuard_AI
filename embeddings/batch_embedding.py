import time

# -----------------------------
# Configuration
# -----------------------------

BATCH_SIZE = 4
MAX_ATTEMPTS = 5
PRICE_PER_1K_TOKENS = 0.00002


# -----------------------------
# Sample chunks
# -----------------------------

all_chunks = [
    {"id": 1, "text": "Artificial intelligence helps computers perform intelligent tasks."},
    {"id": 2, "text": "Machine learning allows systems to learn from data."},
    {"id": 3, "text": "Embeddings convert text into numerical vectors."},
    {"id": 4, "text": "Vector databases are useful for semantic search."},
    {"id": 5, "text": "Batch processing sends multiple texts in one request."},
    {"id": 6, "text": "Retry mechanisms help handle temporary API failures."},
    {"id": 7, "text": "Exponential backoff increases the waiting time between retries."},
    {"id": 8, "text": "Cost estimation helps control embedding expenses."},
    {"id": 9, "text": "Existing embeddings should be skipped during reruns."},
    {"id": 10, "text": "Large datasets can be processed using multiple durable jobs."}
]


# -----------------------------
# Existing embeddings
# -----------------------------

# These chunks are treated as already embedded.
existing_embedding_ids = {2, 5, 8}


# -----------------------------
# Create batches
# -----------------------------

def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


# -----------------------------
# Estimate tokens
# -----------------------------

def estimate_tokens(texts):
    total_words = 0

    for text in texts:
        total_words += len(text.split())

    # Simple approximation:
    # 1 token ≈ 0.75 words
    return int(total_words / 0.75)


# -----------------------------
# Fake embedding API
# -----------------------------

def embedding_api(texts):
    """
    Simulates an embedding API.

    In a real project, replace this function
    with the actual embedding API call.
    """

    embeddings = []

    for text in texts:
        # Simple fake vector for demonstration
        vector = [
            len(text),
            len(text.split()),
            sum(ord(char) for char in text) % 100
        ]

        embeddings.append(vector)

    return embeddings


# -----------------------------
# Retry with exponential backoff
# -----------------------------

def embed_with_retry(texts, max_attempts=5):

    for attempt in range(max_attempts):

        try:
            response = embedding_api(texts)

            return response

        except Exception as error:

            if attempt == max_attempts - 1:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"Retrying after error: {error} "
                f"| wait={wait_seconds}s"
            )

            time.sleep(wait_seconds)


# -----------------------------
# Save embeddings
# -----------------------------

def save_embeddings(batch, embeddings):

    for chunk, embedding in zip(batch, embeddings):

        print(
            f"Saved embedding for chunk {chunk['id']}: "
            f"{embedding}"
        )


# -----------------------------
# Main pipeline
# -----------------------------

def main():

    # Find chunks that do not have embeddings yet
    pending_chunks = [
        chunk
        for chunk in all_chunks
        if chunk["id"] not in existing_embedding_ids
    ]

    # Run summary
    summary = {
        "total_chunks": len(all_chunks),
        "skipped_existing": len(all_chunks) - len(pending_chunks),
        "embedded": 0,
        "failed": 0,
        "input_tokens": 0
    }

    print("Starting batch embedding...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Skipped chunks: {summary['skipped_existing']}")
    print()

    # Process chunks in batches
    for batch_number, batch in enumerate(
        batches(pending_chunks, BATCH_SIZE),
        start=1
    ):

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        print(
            f"Processing batch {batch_number} "
            f"with {len(batch)} chunks..."
        )

        # Estimate tokens
        summary["input_tokens"] += estimate_tokens(texts)

        try:

            embeddings = embed_with_retry(
                texts,
                MAX_ATTEMPTS
            )

            save_embeddings(
                batch,
                embeddings
            )

            summary["embedded"] += len(embeddings)

            print(
                f"Batch {batch_number} completed successfully."
            )

        except Exception as error:

            summary["failed"] += len(batch)

            print(
                f"Batch {batch_number} failed: {error}"
            )

        print()

    # Calculate approximate cost
    estimated_cost = (
        summary["input_tokens"] / 1000
    ) * PRICE_PER_1K_TOKENS

    # Final summary
    print("=" * 40)
    print("RUN SUMMARY")
    print("=" * 40)

    print(
        f"Total chunks       : "
        f"{summary['total_chunks']}"
    )

    print(
        f"Skipped existing   : "
        f"{summary['skipped_existing']}"
    )

    print(
        f"Embeddings created : "
        f"{summary['embedded']}"
    )

    print(
        f"Failed chunks      : "
        f"{summary['failed']}"
    )

    print(
        f"Input tokens       : "
        f"{summary['input_tokens']}"
    )

    print(
        f"Estimated cost USD : "
        f"${estimated_cost:.6f}"
    )

    print("=" * 40)


# -----------------------------
# Run program
# -----------------------------

if __name__ == "__main__":
    main()