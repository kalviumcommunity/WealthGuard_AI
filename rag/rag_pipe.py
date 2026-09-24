# RAG Pipeline Architecture & Flow Design
# Flow:
# User Query
#     ↓
# Embed Query
#     ↓
# Retrieve Top-K Chunks
#     ↓
# Assemble Context
#     ↓
# Generate Grounded Answer
#     ↓
# Return Answer + Sources


# -----------------------------
# Sample knowledge base
# -----------------------------

documents = [
    {
        "text": "Students must submit project evidence including screenshots, code, and a working demonstration.",
        "metadata": {"source": "Project Submission Guide"}
    },
    {
        "text": "A project submission should clearly explain the problem, solution, implementation, and testing process.",
        "metadata": {"source": "Project Documentation Guide"}
    },
    {
        "text": "Students should include relevant GitHub commits and a pull request showing meaningful project work.",
        "metadata": {"source": "GitHub Submission Guide"}
    },
    {
        "text": "The final project should be tested before submission to ensure that the main features work correctly.",
        "metadata": {"source": "Testing Guide"}
    }
]


# -----------------------------
# Stage 1: Embed
# -----------------------------

def embed_query(query):
    """
    Simple demo embedding.
    Converts words into a set so we can compare
    the query with document words.
    """
    return set(query.lower().split())


# -----------------------------
# Stage 2: Retrieve
# -----------------------------

def retrieve_context(query_vector, k=4):
    """
    Retrieve the most relevant documents
    based on the number of matching words.
    """

    results = []

    for document in documents:
        document_vector = set(document["text"].lower().split())

        score = len(query_vector.intersection(document_vector))

        if score > 0:
            results.append({
                "text": document["text"],
                "metadata": document["metadata"],
                "score": score
            })

    # Sort by relevance score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:k]


# -----------------------------
# Stage 3: Assemble Context
# -----------------------------

def assemble_context(chunks):
    """
    Combine retrieved chunks into one context
    with source citations.
    """

    parts = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"]["source"]
        text = chunk["text"]

        parts.append(
            f"[{index}] Source: {source}\n{text}"
        )

    return "\n\n".join(parts)


# -----------------------------
# Stage 4: Generate
# -----------------------------

def generate_answer(query, context):
    """
    Generate a grounded answer using only
    the retrieved context.
    """

    if not context:
        return "I could not find enough relevant context to answer the question."

    # Simple rule-based generation for demonstration
    answer = "Based on the retrieved context:\n\n"

    answer += context

    answer += (
        "\n\nAnswer: "
        "The project submission should include evidence such as "
        "screenshots, code, a working demonstration, documentation, "
        "and relevant GitHub commits or pull requests."
    )

    return answer


# -----------------------------
# Stage 5: Complete RAG Pipeline
# -----------------------------

def answer_query(query, k=4):

    # Step 1: Embed query
    query_vector = embed_query(query)

    # Step 2: Retrieve relevant chunks
    chunks = retrieve_context(query_vector, k)

    # Handle empty retrieval
    if not chunks:
        return {
            "answer": "I could not find relevant context for that question.",
            "sources": []
        }

    # Step 3: Assemble context
    context = assemble_context(chunks)

    # Step 4: Generate answer
    answer = generate_answer(query, context)

    # Step 5: Collect sources
    sources = [
        chunk["metadata"]
        for chunk in chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }


# -----------------------------
# Run End-to-End Pipeline
# -----------------------------

query = "What evidence is required for project submission?"

result = answer_query(query)


# -----------------------------
# Display Output
# -----------------------------

print("=" * 50)
print("RAG PIPELINE")
print("=" * 50)

print("\nUser Query:")
print(query)

print("\nGenerated Answer:")
print(result["answer"])

print("\nRetrieved Sources:")

for source in result["sources"]:
    print("-", source["source"])