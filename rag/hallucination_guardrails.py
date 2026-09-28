import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "wealthguard_chunks"
VECTOR_DB_PATH = "./vector_db/chroma_data"

TOP_K = 3

# Hallucination guardrail:
# A retrieved chunk must have at least this cosine similarity
# with the query to be considered relevant.
RELEVANCE_THRESHOLD = 0.50

MIN_RELEVANT_CHUNKS = 1

REFUSAL_MESSAGE = (
    "I don't have enough verified information in the retrieved "
    "context to answer this reliably."
)

SUCCESS_QUERY = (
    "What is the annual management fee for the Secure Growth Plan?"
)

REFUSAL_QUERY = (
    "What is the expected annual return of the Secure Growth Plan?"
)


# ---------------------------------------------------------
# Embedding
# ---------------------------------------------------------

def embed_query(query, model):
    """Convert the user query into an embedding vector."""
    return model.encode(query).tolist()


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_candidates(query_embedding, collection, top_k=TOP_K):
    """
    Retrieve candidate chunks from Chroma.

    We retrieve the embeddings as well so that relevance
    can be checked using explicit cosine similarity.
    """

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "embeddings"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    embeddings = results.get("embeddings", [[]])[0]

    candidates = []

    for document, metadata, embedding in zip(
        documents, metadatas, embeddings
    ):
        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        candidates.append({
            "document": document,
            "metadata": metadata,
            "embedding": embedding,
            "similarity": similarity,
        })

    # Highest similarity first
    candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return candidates


# ---------------------------------------------------------
# Hallucination guardrail
# ---------------------------------------------------------

def check_retrieval_quality(candidates):
    """
    Decide whether retrieved context is strong enough
    to allow answer generation.
    """

    if not candidates:
        return False, "No retrieval results were returned."

    relevant_chunks = [
        candidate
        for candidate in candidates
        if candidate["similarity"] >= RELEVANCE_THRESHOLD
    ]

    if len(relevant_chunks) < MIN_RELEVANT_CHUNKS:
        highest_score = candidates[0]["similarity"]

        return (
            False,
            (
                "Retrieval is too weak. "
                f"Highest similarity = {highest_score:.4f}, "
                f"required threshold = {RELEVANCE_THRESHOLD:.2f}."
            )
        )

    return True, (
        f"{len(relevant_chunks)} relevant chunk(s) passed "
        f"the {RELEVANCE_THRESHOLD:.2f} threshold."
    )


# ---------------------------------------------------------
# Context assembly
# ---------------------------------------------------------

def assemble_context(relevant_chunks):
    """Build the context injected into the generation prompt."""

    if not relevant_chunks:
        return ""

    context_parts = []

    for index, chunk in enumerate(relevant_chunks, start=1):
        metadata = chunk["metadata"]

        source = metadata.get(
            "source_document",
            "Unknown source"
        )

        section = metadata.get(
            "section",
            "Unknown section"
        )

        page = metadata.get(
            "page",
            "Unknown page"
        )

        context_parts.append(
            f"""
Source {index}
Document: {source}
Section: {section}
Page: {page}
Similarity: {chunk["similarity"]:.4f}

Content:
{chunk["document"]}
"""
        )

    return "\n".join(context_parts)


# ---------------------------------------------------------
# Grounded generation
# ---------------------------------------------------------

def generate_grounded_answer(query, context):
    """
    Generate an answer using only verified retrieved context.
    """

    if not context.strip():
        return REFUSAL_MESSAGE

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    system_prompt = """
You are WealthGuard AI, an evidence-based assistant for
relationship managers.

Answer ONLY using the supplied retrieved context.

Rules:
- Do not use outside knowledge.
- Do not invent facts, numbers, regulations, product features,
  returns, fees, eligibility requirements, or tax treatment.
- Every factual claim must be supported by the supplied context.
- If the context does not support the answer, say:
  "I don't have enough verified information in the retrieved
  context to answer this reliably."
- Keep the answer concise and factual.
"""

    user_prompt = f"""
Retrieved context:

{context}

Question:
{query}

Provide a concise answer based only on the retrieved context.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_tokens=100,
    )

    message = response.choices[0].message

    if message.content:
       return message.content.strip()

# Some OpenAI-compatible providers may return no text content.
# Treat that as a safe refusal instead of generating/returning
# an unsupported answer.
    return REFUSAL_MESSAGE


# ---------------------------------------------------------
# Full guarded RAG pipeline
# ---------------------------------------------------------

def run_guarded_rag(query, model, collection):
    """Run retrieval, guardrail check and grounded generation."""

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    query_embedding = embed_query(query, model)

    candidates = retrieve_candidates(
        query_embedding,
        collection,
        TOP_K
    )

    print("\n" + "=" * 70)
    print("RETRIEVED CANDIDATES")
    print("=" * 70)

    for index, candidate in enumerate(candidates, start=1):
        metadata = candidate["metadata"]

        print(
            f"\n[{index}] "
            f"similarity={candidate['similarity']:.4f}"
        )

        print(
            f"source={metadata.get('source_document', 'unknown')}"
        )

        print(
            f"section={metadata.get('section', 'unknown')}"
        )

        print(
            f"page={metadata.get('page', 'unknown')}"
        )

        print(
            f"text={candidate['document']}"
        )

    allowed, reason = check_retrieval_quality(candidates)

    print("\n" + "=" * 70)
    print("RETRIEVAL QUALITY CHECK")
    print("=" * 70)

    print(f"Threshold: {RELEVANCE_THRESHOLD:.2f}")
    print(f"Minimum relevant chunks: {MIN_RELEVANT_CHUNKS}")
    print(f"Decision: {'ALLOW' if allowed else 'REFUSE'}")
    print(f"Reason: {reason}")

    if not allowed:
        print("\n" + "=" * 70)
        print("SAFE REFUSAL")
        print("=" * 70)
        print(REFUSAL_MESSAGE)

        return {
            "allowed": False,
            "answer": REFUSAL_MESSAGE,
            "candidates": candidates,
        }

    relevant_chunks = [
        candidate
        for candidate in candidates
        if candidate["similarity"] >= RELEVANCE_THRESHOLD
    ]

    context = assemble_context(relevant_chunks)

    answer = generate_grounded_answer(
        query,
        context
    )

    print("\n" + "=" * 70)
    print("GROUNDED ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("SUPPORTING CHUNKS")
    print("=" * 70)

    for index, chunk in enumerate(relevant_chunks, start=1):
        metadata = chunk["metadata"]

        print(
            f"[{index}] "
            f"{metadata.get('source_document', 'unknown')} | "
            f"{metadata.get('section', 'unknown')} | "
            f"page={metadata.get('page', 'unknown')} | "
            f"similarity={chunk['similarity']:.4f}"
        )

    return {
        "allowed": True,
        "answer": answer,
        "candidates": candidates,
        "relevant_chunks": relevant_chunks,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("WEALTHGUARD AI - HALLUCINATION GUARDRAILS")
    print("=" * 70)

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # -----------------------------------------------------
    # SUCCESS CASE
    # -----------------------------------------------------

    success_result = run_guarded_rag(
        SUCCESS_QUERY,
        model,
        collection
    )

    # -----------------------------------------------------
    # REFUSAL CASE
    # -----------------------------------------------------

    refusal_result = run_guarded_rag(
        REFUSAL_QUERY,
        model,
        collection
    )

    # -----------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("GUARDRAIL EVALUATION")
    print("=" * 70)

    print(
        "Strong-context case: "
        f"{'ANSWERED' if success_result['allowed'] else 'REFUSED'}"
    )

    print(
        "Weak/unsupported-context case: "
        f"{'REFUSED' if not refusal_result['allowed'] else 'ANSWERED'}"
    )

    print("\nExpected behaviour:")
    print("- Strong supporting context -> grounded answer")
    print("- Weak or unsupported context -> safe refusal")
    print("- No unsupported facts should be generated")


if __name__ == "__main__":
    main()