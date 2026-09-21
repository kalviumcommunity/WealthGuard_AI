import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "wealthguard_chunks"
VECTOR_DIMENSION = 384


def main():
    print("WEALTHGUARD VECTOR DATABASE DEMO")
    print("=" * 70)

    # Create a local persistent Chroma database.
    client = chromadb.PersistentClient(path="./vector_db/chroma_data")

    print("Vector database: ChromaDB")
    print("Storage: ./vector_db/chroma_data")
    print("Database reachable: True")

    # Create or reuse the collection.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "WealthGuard AI embedded document chunks",
            "vector_dimension": VECTOR_DIMENSION,
            "embedding_model": MODEL_NAME,
        },
    )

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Expected vector dimension: {VECTOR_DIMENSION}")

    # Load the same embedding model used in the previous assignment.
    model = SentenceTransformer(MODEL_NAME)

    # Test record.
    test_text = (
        "The Secure Growth Plan has an annual management fee "
        "of 1.2 percent."
    )

    test_metadata = {
        "source_document": "product_brochure.txt",
        "chunk_index": 0,
        "section": "Fees",
        "page": 2,
    }

    test_id = "test_chunk_001"

    # Generate the embedding.
    embedding = model.encode(test_text).tolist()

    print(f"Generated vector length: {len(embedding)}")

    if len(embedding) != VECTOR_DIMENSION:
        raise ValueError(
            f"Vector dimension mismatch: expected {VECTOR_DIMENSION}, "
            f"got {len(embedding)}"
        )

    # Insert the test record.
    collection.upsert(
        ids=[test_id],
        embeddings=[embedding],
        documents=[test_text],
        metadatas=[test_metadata],
    )

    print("\nTEST RECORD INSERTED")
    print("-" * 70)
    print(f"ID: {test_id}")

    # Read the record back.
    result = collection.get(
        ids=[test_id],
        include=["embeddings", "documents", "metadatas"],
    )

    print("\nREAD-BACK TEST")
    print("-" * 70)

    returned_id = result["ids"][0]
    returned_vector = result["embeddings"][0]
    returned_text = result["documents"][0]
    returned_metadata = result["metadatas"][0]

    print(f"ID: {returned_id}")
    print(f"Vector length: {len(returned_vector)}")
    print(f"Text: {returned_text}")
    print(f"Metadata: {returned_metadata}")

    # Verify the record.
    assert returned_id == test_id
    assert len(returned_vector) == VECTOR_DIMENSION
    assert returned_text == test_text
    assert returned_metadata == test_metadata

    print("\nVERIFICATION")
    print("-" * 70)
    print("Database reachable: PASS")
    print("Correct vector dimension: PASS")
    print("Embedding stored: PASS")
    print("Source text stored: PASS")
    print("Metadata stored: PASS")
    print("Read-back successful: PASS")


if __name__ == "__main__":
    main()