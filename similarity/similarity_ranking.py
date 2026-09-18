from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (norm_a * norm_b)


def main():
    # Small sample corpus representing retrieved RAG chunks.
    chunks = [
        {
            "id": "chunk_1",
            "source": "product_brochure.txt",
            "page": 2,
            "text": (
                "The Secure Growth Plan has an annual management fee "
                "of 1.2 percent and a minimum investment of fifty "
                "thousand rupees."
            ),
        },
        {
            "id": "chunk_2",
            "source": "product_brochure.txt",
            "page": 3,
            "text": (
                "The Secure Growth Plan has a three-year lock-in period. "
                "Early withdrawal is subject to applicable terms and conditions."
            ),
        },
        {
            "id": "chunk_3",
            "source": "compliance.txt",
            "page": 1,
            "text": (
                "Capital protection is not guaranteed. Customers should "
                "review the applicable product terms before investing."
            ),
        },
        {
            "id": "chunk_4",
            "source": "tax_rules.txt",
            "page": 4,
            "text": (
                "Tax treatment may depend on applicable regulations "
                "and customer circumstances."
            ),
        },
        {
            "id": "chunk_5",
            "source": "weather.txt",
            "page": 1,
            "text": (
                "The mountain weather forecast predicts heavy snowfall "
                "tomorrow."
            ),
        },
    ]

    query = (
        "What is the annual management fee for the Secure Growth Plan?"
    )

    print("SIMILARITY RANKING DEMONSTRATION")
    print("=" * 70)

    model = SentenceTransformer(MODEL_NAME)

    chunk_texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings for query and chunks.
    query_vector = model.encode(query)
    chunk_vectors = model.encode(chunk_texts)

    # Compare query embedding against every chunk embedding.
    results = []

    for chunk, vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(query_vector, vector)

        results.append(
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": score,
            }
        )

    # Highest similarity first.
    results.sort(key=lambda item: item["score"], reverse=True)

    print(f"Model: {MODEL_NAME}")
    print(f"Query: {query}")

    print("\nRANKED RESULTS")
    print("-" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Chunk ID: {result['id']}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Cosine similarity: {result['score']:.4f}")
        print(f"Text: {result['text']}")

    print("\nMOST SIMILAR")
    print("-" * 70)
    print(f"{results[0]['id']} - score: {results[0]['score']:.4f}")
    print(results[0]["text"])

    print("\nLEAST SIMILAR")
    print("-" * 70)
    print(f"{results[-1]['id']} - score: {results[-1]['score']:.4f}")
    print(results[-1]["text"])

    print("\nMETRIC JUSTIFICATION")
    print("-" * 70)
    print(
        "Cosine similarity compares the direction of embedding vectors. "
        "It is useful for semantic retrieval because texts with similar "
        "meaning tend to have more similar vector directions."
    )


if __name__ == "__main__":
    main()