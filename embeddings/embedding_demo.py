from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (norm_a * norm_b)


def main():
    texts = [
        "The Secure Growth Plan has an annual management fee of 1.2 percent.",
        "The Secure Growth Plan charges a yearly management fee of 1.2 percent.",
        "The mountain weather forecast predicts heavy snowfall tomorrow.",
    ]

    print("LOADING EMBEDDING MODEL")
    print("=" * 60)

    model = SentenceTransformer(MODEL_NAME)

    vectors = model.encode(texts)

    print(f"Model: {MODEL_NAME}")
    print(f"Number of texts: {len(texts)}")

    dimension = len(vectors[0])
    print(f"Vector dimension: {dimension}")

    all_same_length = all(len(vector) == dimension for vector in vectors)
    print(f"All vectors same length: {all_same_length}")

    print("\nSAMPLE VECTOR VALUES")
    print("-" * 60)

    for i, vector in enumerate(vectors, start=1):
        print(f"Text {i} first 8 values: {vector[:8]}")

    similar_score = cosine_similarity(vectors[0], vectors[1])
    unrelated_score = cosine_similarity(vectors[0], vectors[2])

    print("\nCOSINE SIMILARITY")
    print("-" * 60)

    print(f"Similar texts:   {similar_score:.4f}")
    print(f"Unrelated text:  {unrelated_score:.4f}")

    print("\nINTERPRETATION")
    print("-" * 60)
    print(
        "The similar texts have a higher cosine similarity because "
        "they express the same meaning."
    )
    print(
        "The unrelated text has a lower similarity because it discusses "
        "a different topic."
    )

    print("\nEMBEDDING EXPLANATION")
    print("-" * 60)
    print(
        "Embeddings are numerical vector representations of text meaning. "
        "They are not random IDs or simple keyword counts. "
        "Texts with similar meanings tend to have vectors that are "
        "closer together in vector space."
    )


if __name__ == "__main__":
    main()