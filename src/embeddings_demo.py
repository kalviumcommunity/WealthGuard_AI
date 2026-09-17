"""
Embeddings Fundamentals & Vector Representation
=================================================
Demonstrates generating embeddings, inspecting vector dimensions,
and comparing similar vs dissimilar texts using cosine similarity.

Uses chromadb's built-in embedding function (all-MiniLM-L6-v2)
so no API key is required to run.
"""

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ── Embedding helper ───────────────────────────────────────────────
embed_fn = DefaultEmbeddingFunction()          # Sentence-Transformers model

def embed(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a list of texts."""
    return embed_fn(texts)

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── Main demonstration ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Task 1: Generate embeddings for sample texts ───────────────
    texts = [
        "How do I reset my account password?",          # A
        "Steps to recover access to my login",          # B  (similar to A)
        "The cafeteria menu has pasta today",            # C  (unrelated to A)
        "What is the company's annual leave policy?",   # D
        "How many vacation days do employees get?",     # E  (similar to D)
    ]

    print("=" * 60)
    print("EMBEDDINGS FUNDAMENTALS & VECTOR REPRESENTATION")
    print("=" * 60)

    print("\n── Sample Texts ──")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t}")

    embeddings = embed(texts)

    # ── Task 2: Report vector dimension ────────────────────────────
    dims = [len(v) for v in embeddings]
    print(f"\n── Vector Dimensions ──")
    print(f"  Dimension of each embedding: {dims[0]}")
    print(f"  All vectors same length?     {len(set(dims)) == 1}  ({dims})")
    print(f"  First 8 values of text [0]:  {[round(x, 6) for x in embeddings[0][:8]]}")

    # ── Task 3: Compare similar and dissimilar pairs ───────────────
    pairs = [
        (0, 1, "password reset ↔ login recovery  (SIMILAR)"),
        (0, 2, "password reset ↔ cafeteria menu   (DISSIMILAR)"),
        (3, 4, "leave policy   ↔ vacation days    (SIMILAR)"),
        (3, 2, "leave policy   ↔ cafeteria menu   (DISSIMILAR)"),
    ]

    print(f"\n── Cosine Similarity Scores ──")
    for i, j, label in pairs:
        score = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  {label}  →  {score:.4f}")

    # ── Task 4: Explanation note ───────────────────────────────────
    note = """
── What Do Embedding Vectors Represent? ──

An embedding vector is NOT a random ID or a bag-of-words keyword count.
It is a list of numbers produced by a neural network such that the
*position* of the text in high-dimensional space encodes its *meaning*.

• Texts with similar meaning → vectors that point in similar directions
  → high cosine similarity.
• Texts about unrelated topics → vectors that point in different
  directions → low cosine similarity.

No single dimension has a human-readable label. The full pattern across
all dimensions is what captures semantic relationships.

This is what enables SEMANTIC SEARCH in a RAG pipeline:
  1. Every document chunk is embedded and stored in a vector database.
  2. A user's question is also embedded.
  3. Retrieval = nearest-neighbour search in vector space.
  4. Chunks closest to the question vector are returned, even when the
     exact wording differs — because meaning, not keywords, is matched.
"""
    print(note)
