import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "wealthguard_chunks"
VECTOR_DIMENSION = 384


def main():
    print("WEALTHGUARD RETRIEVAL DEMONSTRATION")
    print("=" * 70)

    # Connect to the same local ChromaDB used in the previous task.
    client = chromadb.PersistentClient(
        path="./vector_db/chroma_data"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Existing records: {collection.count()}")

    # Use the SAME embedding model used when creating the chunks.
    model = SentenceTransformer(MODEL_NAME)

    query = (
        "What is the annual management fee for the Secure Growth Plan?"
    )

    print(f"\nQuery: {query}")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Expected vector dimension: {VECTOR_DIMENSION}")

    # ---------------------------------------------------------
    # Make sure the collection has several chunks for retrieval.
    # ---------------------------------------------------------

    chunks = [
        {
            "id": "chunk_1",
            "text": (
                "The Secure Growth Plan has an annual management fee "
                "of 1.2 percent and a minimum investment of fifty "
                "thousand rupees."
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
                "The Secure Growth Plan has a three-year lock-in period. "
                "Early withdrawal is subject to applicable terms and "
                "conditions."
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
                "Capital protection is not guaranteed. Customers should "
                "review the applicable product terms before investing."
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
                "Tax treatment may depend on applicable regulations "
                "and customer circumstances."
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
                "The mountain weather forecast predicts heavy snowfall "
                "tomorrow."
            ),
            "metadata": {
                "source_document": "weather.txt",
                "chunk_index": 0,
                "section": "Weather",
                "page": 1,
            },
        },
    ]

    # Generate embeddings using the same model.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts).tolist()

    # Store the chunks in ChromaDB.
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[chunk["metadata"] for chunk in chunks],
    )

    print(f"Records after indexing: {collection.count()}")

    # ---------------------------------------------------------
    # Task 1: Embed the user query.
    # ---------------------------------------------------------

    query_embedding = model.encode(query).tolist()

    print(f"Query vector length: {len(query_embedding)}")

    if len(query_embedding) != VECTOR_DIMENSION:
        raise ValueError(
            f"Query vector dimension mismatch. "
            f"Expected {VECTOR_DIMENSION}, "
            f"got {len(query_embedding)}"
        )

    # ---------------------------------------------------------
    # Task 2 + 3: Top-k similarity search.
    # ---------------------------------------------------------

    def retrieve(top_k):
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        print("\n" + "=" * 70)
        print(f"TOP-{top_k} RETRIEVAL RESULTS")
        print("=" * 70)

        for rank in range(len(result["ids"][0])):
            chunk_id = result["ids"][0][rank]
            text = result["documents"][0][rank]
            metadata = result["metadatas"][0][rank]
            distance = result["distances"][0][rank]

            # Chroma's default distance for this collection is cosine
            # distance, so similarity = 1 - distance.
            similarity = 1 - distance

            print(f"\nRank: {rank + 1}")
            print(f"ID: {chunk_id}")
            print(f"Similarity score: {similarity:.4f}")
            print(f"Source document: {metadata['source_document']}")
            print(f"Chunk index: {metadata['chunk_index']}")
            print(f"Section: {metadata['section']}")
            print(f"Page: {metadata['page']}")
            print(f"Text: {text}")

    # ---------------------------------------------------------
    # Task 4: Demonstrate different k values.
    # ---------------------------------------------------------

    retrieve(1)
    retrieve(3)


if __name__ == "__main__":
    main()