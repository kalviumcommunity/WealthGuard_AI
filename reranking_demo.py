import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "wealthguard_chunks"
CANDIDATE_COUNT = 10
FINAL_K = 3

QUERY = "What is the annual management fee for the Secure Growth Plan?"


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def main():
    print("=" * 80)
    print("WEALTHGUARD AI - RETRIEVAL + RE-RANKING")
    print("=" * 80)

    print(f"\nQuery: {QUERY}")
    print(f"Candidate count: {CANDIDATE_COUNT}")
    print(f"Final results: {FINAL_K}")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path="./vector_db/chroma_data"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Load the same embedding model used for indexing
    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------------
    # Make sure the collection contains enough candidate chunks
    # ---------------------------------------------------------

    chunks = [
        {
            "id": "chunk_1",
            "text": (
                "The Secure Growth Plan has an annual management "
                "fee of 1.2 percent."
            ),
            "metadata": {
                "source_document": "product_brochure.txt",
                "chunk_index": 0,
                "section": "Fees",
                "page": 2,
            },
        },
        {
            "id": "chunk_2",
            "text": (
                "The Secure Growth Plan has a three-year lock-in "
                "period and early withdrawal is subject to applicable terms."
            ),
            "metadata": {
                "source_document": "product_brochure.txt",
                "chunk_index": 1,
                "section": "Withdrawal Conditions",
                "page": 3,
            },
        },
        {
            "id": "chunk_3",
            "text": (
                "Capital protection is not guaranteed and investment "
                "value may fluctuate."
            ),
            "metadata": {
                "source_document": "compliance.txt",
                "chunk_index": 0,
                "section": "Risk",
                "page": 1,
            },
        },
        {
            "id": "chunk_4",
            "text": (
                "Tax treatment depends on applicable regulations and "
                "the customer's circumstances."
            ),
            "metadata": {
                "source_document": "tax_rules.txt",
                "chunk_index": 0,
                "section": "Tax Treatment",
                "page": 4,
            },
        },
        {
            "id": "chunk_5",
            "text": (
                "The mountain weather forecast predicts heavy "
                "snowfall tomorrow."
            ),
            "metadata": {
                "source_document": "weather.txt",
                "chunk_index": 0,
                "section": "Weather",
                "page": 1,
            },
        },
    ]

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    embeddings = model.encode(texts).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    # ---------------------------------------------------------
    # Task 1: Initial retrieval - larger candidate set
    # ---------------------------------------------------------

    query_embedding = model.encode(QUERY).tolist()

    candidate_count = min(CANDIDATE_COUNT, collection.count())

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances",
            "embeddings",
        ],
    )

    candidates = []

    for i in range(len(result["ids"][0])):
        distance = result["distances"][0][i]

        # Chroma cosine distance -> cosine similarity
        vector_score = 1 - distance

        candidates.append(
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "vector_score": vector_score,
                "embedding": result["embeddings"][0][i],
            }
        )

    # ---------------------------------------------------------
    # Task 4: BEFORE re-ranking
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("BEFORE RE-RANKING")
    print("=" * 80)

    for rank, candidate in enumerate(candidates, start=1):
        print(f"\nRank {rank}")
        print(f"ID: {candidate['id']}")
        print(f"Vector score: {candidate['vector_score']:.4f}")
        print(f"Source: {candidate['metadata']['source_document']}")
        print(f"Section: {candidate['metadata']['section']}")
        print(f"Page: {candidate['metadata']['page']}")
        print(f"Text: {candidate['text']}")

    # ---------------------------------------------------------
    # Task 2: Re-ranking
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("RE-RANKING CANDIDATES")
    print("=" * 80)

    for candidate in candidates:
        rerank_score = cosine_similarity(
            query_embedding,
            candidate["embedding"],
        )

        candidate["rerank_score"] = rerank_score

    reranked = sorted(
        candidates,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    # ---------------------------------------------------------
    # Task 3 + Task 4: AFTER re-ranking
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("AFTER RE-RANKING")
    print("=" * 80)

    for rank, candidate in enumerate(reranked, start=1):
        print(f"\nRank {rank}")
        print(f"ID: {candidate['id']}")
        print(f"Original vector score: {candidate['vector_score']:.4f}")
        print(f"Re-rank score: {candidate['rerank_score']:.4f}")
        print(f"Source: {candidate['metadata']['source_document']}")
        print(f"Section: {candidate['metadata']['section']}")
        print(f"Page: {candidate['metadata']['page']}")
        print(f"Text: {candidate['text']}")

    # ---------------------------------------------------------
    # Task 3: Final top-k
    # ---------------------------------------------------------

    final_results = reranked[:FINAL_K]

    print("\n" + "=" * 80)
    print(f"FINAL SELECTED CHUNKS (TOP {FINAL_K})")
    print("=" * 80)

    for rank, candidate in enumerate(final_results, start=1):
        print(f"\nFinal Rank {rank}")
        print(f"ID: {candidate['id']}")
        print(f"Re-rank score: {candidate['rerank_score']:.4f}")
        print(f"Source: {candidate['metadata']['source_document']}")
        print(f"Section: {candidate['metadata']['section']}")
        print(f"Page: {candidate['metadata']['page']}")
        print(f"Text: {candidate['text']}")

    print("\n" + "=" * 80)
    print("RE-RANKING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()