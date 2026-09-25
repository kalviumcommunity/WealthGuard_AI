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

TOP_K = 3

GROUNDED_QUERY = (
    "What is the annual management fee for the Secure Growth Plan?"
)

UNSUPPORTED_QUERY = (
    "What is the expected annual return of the Secure Growth Plan?"
)


# ============================================================
# Stage 1: Embed
# ============================================================

def embed_query(query, model):
    """Convert a query into an embedding."""

    embedding = model.encode(query).tolist()

    if len(embedding) != 384:
        raise ValueError(
            f"Expected 384 dimensions, got {len(embedding)}"
        )

    return embedding


# ============================================================
# Stage 2: Retrieve
# ============================================================

def retrieve_chunks(query_embedding, collection, top_k=TOP_K):
    """Retrieve relevant chunks from ChromaDB."""

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

        # Chroma cosine distance → similarity
        score = 1 - distance

        chunks.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": score,
            }
        )

    return chunks


# ============================================================
# Stage 3: Assemble context
# ============================================================

def assemble_context(chunks):
    """Convert retrieved chunks into grounded context."""

    if not chunks:
        return ""

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
# Stage 4: Generate grounded answer
# ============================================================

def generate_grounded_answer(query, context):
    """
    Generate an answer using ONLY the supplied context.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv(
        "OPENAI_MODEL",
        "openrouter/free"
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    if not context.strip():
        return (
            "I don't have enough verified information "
            "in the provided context to answer this reliably."
        )

    system_prompt = """
You are WealthGuard AI, an evidence-based assistant.

Your answer MUST be grounded only in the supplied context.

Rules:
1. Use only facts explicitly supported by the context.
2. Do not add outside knowledge.
3. Do not invent product features, fees, returns,
   regulations, eligibility requirements, or numbers.
4. If the context does not support the answer, say:
   "I don't have enough verified information in the
   provided context to answer this reliably."
5. Keep the answer concise and factual.
6. Mention the supporting source when appropriate.
"""

    user_prompt = f"""
Retrieved context:

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
# Stage 5: Generate without retrieval
# ============================================================

def generate_without_retrieval(query):
    """
    Generate an answer without providing retrieved context.

    This is used only to compare grounded vs ungrounded generation.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv(
        "OPENAI_MODEL",
        "openrouter/free"
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    system_prompt = """
You are WealthGuard AI.

Answer the question directly without access to
retrieved organizational context.

Do not claim that information is verified from
WealthGuard documents.

Clearly indicate when the available information
is insufficient to provide a verified answer.
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
                "content": query,
            },
        ],
        max_tokens=150,
    )

    return response.choices[0].message.content


# ============================================================
# Display retrieved sources
# ============================================================

def print_sources(chunks):
    """Print the chunks used to generate the answer."""

    print("\nSUPPORTING RETRIEVED CHUNKS")
    print("=" * 70)

    for index, chunk in enumerate(chunks, start=1):

        metadata = chunk["metadata"]

        print(f"\nSource {index}")
        print("-" * 70)

        print(
            f"Similarity score: "
            f"{chunk['score']:.4f}"
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


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 80)
    print("WEALTHGUARD AI - GROUNDED GENERATION EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load components
    # --------------------------------------------------------

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # ========================================================
    # Task 1 + Task 2
    # Generate using retrieved context
    # ========================================================

    print("\n" + "=" * 80)
    print("1. GROUNDED ANSWER WITH RETRIEVAL")
    print("=" * 80)

    print(f"\nQuestion:\n{GROUNDED_QUERY}")

    query_embedding = embed_query(
        GROUNDED_QUERY,
        model
    )

    chunks = retrieve_chunks(
        query_embedding,
        collection,
        TOP_K
    )

    context = assemble_context(chunks)

    print_sources(chunks)

    grounded_answer = generate_grounded_answer(
        GROUNDED_QUERY,
        context
    )

    print("\nGENERATED GROUNDED ANSWER")
    print("-" * 70)
    print(grounded_answer)

    # ========================================================
    # Task 3
    # Missing-context fallback
    # ========================================================

    print("\n" + "=" * 80)
    print("2. MISSING-CONTEXT FALLBACK")
    print("=" * 80)

    print(f"\nQuestion:\n{UNSUPPORTED_QUERY}")

    # Deliberately provide no context.
    fallback_answer = generate_grounded_answer(
        UNSUPPORTED_QUERY,
        ""
    )

    print("\nFALLBACK ANSWER")
    print("-" * 70)
    print(fallback_answer)

    # ========================================================
    # Task 4
    # Compare with and without retrieval
    # ========================================================

    print("\n" + "=" * 80)
    print("3. WITH RETRIEVAL VS WITHOUT RETRIEVAL")
    print("=" * 80)

    print(f"\nQuestion:\n{GROUNDED_QUERY}")

    print("\nWITH RETRIEVAL")
    print("-" * 70)
    print(grounded_answer)

    print("\nWITHOUT RETRIEVAL")
    print("-" * 70)

    ungrounded_answer = generate_without_retrieval(
        GROUNDED_QUERY
    )

    print(ungrounded_answer)

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print("""
Grounding checks:

1. The grounded answer was generated using retrieved context.
2. Supporting chunks are printed with source metadata.
3. Empty context triggers the missing-context fallback.
4. The same query was evaluated with and without retrieval.
5. The comparison demonstrates the role of retrieved evidence.
""")

    print("=" * 80)
    print("GROUNDED GENERATION EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()