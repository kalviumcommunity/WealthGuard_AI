import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()
# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "wealthguard_chunks"
VECTOR_DB_PATH = "./vector_db/chroma_data"

EMBEDDING_DIMENSION = 384
TOP_K = 3

QUERY = "What is the annual management fee for the Secure Growth Plan?"


# ============================================================
# Stage 1: EMBED
# ============================================================

def embed_query(query, model):
    """Convert the user query into an embedding vector."""

    embedding = model.encode(query).tolist()

    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIMENSION}, "
            f"got {len(embedding)}"
        )

    return embedding


# ============================================================
# Stage 2: RETRIEVE
# ============================================================

def retrieve_chunks(query_embedding, collection, top_k=TOP_K):
    """Retrieve the most relevant document chunks from ChromaDB."""

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    chunks = []

    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]

        # Chroma cosine distance -> similarity score
        similarity = 1 - distance

        chunks.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": similarity,
            }
        )

    return chunks


# ============================================================
# Stage 3: ASSEMBLE CONTEXT
# ============================================================

def assemble_context(chunks):
    """Build grounded context from retrieved chunks."""

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):

        metadata = chunk["metadata"]

        source = metadata.get(
            "source_document",
            "unknown"
        )

        section = metadata.get(
            "section",
            "unknown"
        )

        page = metadata.get(
            "page",
            "unknown"
        )

        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {source}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Text: {chunk['text']}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# Stage 4: GENERATE
# ============================================================

def generate_answer(query, context):
    """Generate a grounded answer using the retrieved context."""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", "openrouter/free")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    if base_url:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    else:
        client = OpenAI(
            api_key=api_key,
        )

    system_prompt = """
You are WealthGuard AI, an evidence-based wealth-management assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not invent facts, numbers, regulations, or product features.
- If the context does not contain enough information, say:
  "I don't have enough verified information to answer this reliably."
- Keep the answer concise and factual.
- Mention the relevant source when possible.
"""

    user_prompt = f"""
Context:

{context}

Question:

{query}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=150,
    )

    return response.choices[0].message.content


# ============================================================
# Complete RAG Pipeline
# ============================================================

def run_rag(query):
    """Run the complete query-to-answer RAG pipeline."""

    print("=" * 80)
    print("WEALTHGUARD AI - END-TO-END RAG PIPELINE")
    print("=" * 80)

    print(f"\nUser query:\n{query}")

    # Load embedding model
    print("\n[1] Loading embedding model...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    # Connect to vector database
    print("[2] Connecting to ChromaDB...")
    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # --------------------------------------------------------
    # EMBED
    # --------------------------------------------------------

    print("[3] Embedding query...")
    query_embedding = embed_query(
        query,
        embedding_model
    )

    print(
        f"    Query vector dimension: "
        f"{len(query_embedding)}"
    )

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    print("[4] Retrieving relevant chunks...")

    chunks = retrieve_chunks(
        query_embedding,
        collection,
        TOP_K
    )

    print(
        f"    Retrieved chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # ASSEMBLE
    # --------------------------------------------------------

    print("[5] Assembling grounded context...")

    context = assemble_context(chunks)

    print(
        f"    Context length: {len(context)} characters"
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print("[6] Generating answer...")

    answer = generate_answer(
        query,
        context
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("GENERATED ANSWER")
    print("=" * 80)

    print(answer)

    print("\n" + "=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for index, chunk in enumerate(chunks, start=1):

        metadata = chunk["metadata"]

        print(f"\nSource {index}")
        print("-" * 40)
        print(
            f"Score: {chunk['score']:.4f}"
        )
        print(
            f"Document: "
            f"{metadata.get('source_document', 'unknown')}"
        )
        print(
            f"Section: "
            f"{metadata.get('section', 'unknown')}"
        )
        print(
            f"Page: "
            f"{metadata.get('page', 'unknown')}"
        )
        print(
            f"Text: {chunk['text']}"
        )

    print("\n" + "=" * 80)
    print("RAG PIPELINE COMPLETED")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_rag(QUERY)