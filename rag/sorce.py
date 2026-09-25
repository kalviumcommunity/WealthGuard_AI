# ============================================
# Source Citation & Attribution - Kalvium 3.40
# ============================================

# --------------------------------------------
# Sample retrieved chunks
# In your actual RAG project, these will come
# from your vector database/retriever.
# --------------------------------------------

DOCUMENTS = [
    {
        "text": """
        Students must submit a GitHub pull request containing
        meaningful commits and the required implementation.
        The repository must be public and the pull request
        must be open at the time of submission.
        """,
        "metadata": {
            "source": "assignment.md",
            "chunk_id": "chunk_01",
            "chunk_index": 1,
            "section": "Submission"
        }
    },
    {
        "text": """
        Students must also submit a 3-5 minute video explanation.
        The video should explain why citations matter, how a citation
        maps to its source, how chunk metadata enables citations,
        and the risk of fabricated citations.
        """,
        "metadata": {
            "source": "assignment.md",
            "chunk_id": "chunk_02",
            "chunk_index": 2,
            "section": "Video Explanation"
        }
    },
    {
        "text": """
        Citations should map to the real document and location
        used to generate an answer. Metadata can include the
        document name, chunk ID, chunk index, page, or section.
        """,
        "metadata": {
            "source": "rag_notes.md",
            "chunk_id": "chunk_03",
            "chunk_index": 3,
            "section": "Citation Mapping"
        }
    }
]


# --------------------------------------------
# RETRIEVE
# Simple demonstration retriever.
#
# In your actual project, replace this with
# your vector database retrieval function.
# --------------------------------------------

def retrieve(question, k=4):

    question_words = set(question.lower().split())

    scored_chunks = []

    for chunk in DOCUMENTS:

        text_words = set(chunk["text"].lower().split())

        score = len(question_words.intersection(text_words))

        scored_chunks.append((score, chunk))

    # Highest matching chunks first
    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Return only chunks that have some relevance
    results = [
        chunk
        for score, chunk in scored_chunks[:k]
        if score > 0
    ]

    return results


# --------------------------------------------
# BUILD CITATION MAP
# Creates:
#
# [1] -> source + chunk + section + text
# [2] -> source + chunk + section + text
# --------------------------------------------

def build_citation_map(chunks):

    citation_map = {}

    for index, chunk in enumerate(chunks, start=1):

        metadata = chunk.get("metadata", {})

        citation_map[f"[{index}]"] = {
            "source": metadata.get(
                "source",
                "Unknown source"
            ),

            "chunk_id": metadata.get(
                "chunk_id",
                chunk.get("id", "Unknown chunk")
            ),

            "chunk_index": metadata.get(
                "chunk_index"
            ),

            "page": metadata.get(
                "page"
            ),

            "section": metadata.get(
                "section"
            ),

            "text": chunk.get(
                "text",
                ""
            )
        }

    return citation_map


# --------------------------------------------
# BUILD CONTEXT
# Converts retrieved chunks into:
#
# [1] text...
# [2] text...
# --------------------------------------------

def build_context(chunks):

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):

        context_parts.append(
            f"[{index}]\n"
            f"{chunk['text'].strip()}"
        )

    return "\n\n".join(context_parts)


# --------------------------------------------
# BUILD CITED PROMPT
# --------------------------------------------

def build_cited_prompt(question, chunks):

    context = build_context(chunks)

    prompt = f"""
You are a knowledge assistant.

Answer the user's question using ONLY the
provided context.

Citation rules:

1. Cite factual claims using [1], [2], etc.
2. Only use citation numbers that exist in the context.
3. Never invent a citation.
4. Never create a source that was not provided.
5. If the context does not contain enough information,
   clearly say that there is not enough information.
6. Do not use outside knowledge.

Context:

{context}

Question:

{question}

Answer:
"""

    return prompt


# --------------------------------------------
# SIMPLE LOCAL ANSWER GENERATOR
#
# This is used so the example can run without
# an external API.
#
# In your real RAG project, replace this with
# your LLM/API call.
# --------------------------------------------

def generate_answer(question, chunks):

    question_lower = question.lower()

    available = {
        "github": False,
        "video": False,
        "citation": False
    }

    for chunk in chunks:

        text = chunk["text"].lower()

        if "github" in text:
            available["github"] = True

        if "video" in text:
            available["video"] = True

        if "citation" in text or "metadata" in text:
            available["citation"] = True


    # Question about submission requirements
    if "submission" in question_lower:

        parts = []

        if available["github"]:
            parts.append(
                "A public GitHub repository with an open "
                "pull request is required [1]."
            )

        if available["video"]:
            parts.append(
                "A 3-5 minute video explanation is also required [2]."
            )

        if parts:
            return " ".join(parts)

    # Question about citations
    if "citation" in question_lower:

        if available["citation"]:
            return (
                "Citations can be mapped to the real document, "
                "chunk ID, chunk index, page, or section used to "
                "generate the answer [3]."
            )

    # Fallback
    return (
        "I don't have enough information in the provided "
        "sources to answer this question."
    )


# --------------------------------------------
# MAIN RAG FUNCTION
# --------------------------------------------

def answer_with_citations(question):

    # Step 1: Retrieve chunks
    chunks = retrieve(question, k=4)

    # ----------------------------------------
    # NO SOURCE FALLBACK
    # ----------------------------------------

    if not chunks:

        return {
            "answer": (
                "I don't have enough information in the "
                "provided sources."
            ),
            "citations": {}
        }

    # Step 2: Build prompt
    prompt = build_cited_prompt(
        question,
        chunks
    )

    # Show prompt for demonstration
    print("\n========== PROMPT SENT TO MODEL ==========")
    print(prompt)

    # Step 3: Generate answer
    answer = generate_answer(
        question,
        chunks
    )

    # Step 4: Build citation map
    citations = build_citation_map(
        chunks
    )

    return {
        "answer": answer,
        "citations": citations
    }


# --------------------------------------------
# DISPLAY RESULT
# --------------------------------------------

def display_result(result):

    print("\n==========================================")
    print("ANSWER")
    print("==========================================")

    print(result["answer"])


    print("\n==========================================")
    print("CITATIONS")
    print("==========================================")

    if not result["citations"]:

        print("No citations available.")

        return


    for citation, details in result["citations"].items():

        print("\n" + citation)

        print(
            "Source:",
            details["source"]
        )

        print(
            "Chunk ID:",
            details["chunk_id"]
        )

        print(
            "Chunk Index:",
            details["chunk_index"]
        )

        print(
            "Section:",
            details["section"]
        )

        print(
            "Original Text:",
            details["text"].strip()
        )


# --------------------------------------------
# VERIFY CITATION
# --------------------------------------------

def verify_citation(result, citation):

    print("\n==========================================")
    print("CITATION VERIFICATION")
    print("==========================================")

    if citation not in result["citations"]:

        print(
            f"{citation} does not exist."
        )

        return


    details = result["citations"][citation]

    print(
        "Citation:",
        citation
    )

    print(
        "Source:",
        details["source"]
    )

    print(
        "Chunk ID:",
        details["chunk_id"]
    )

    print(
        "Section:",
        details["section"]
    )

    print("\nOriginal retrieved text:")

    print(
        details["text"].strip()
    )


# ============================================
# TEST 1
# ============================================

print("\n\n########################################")
print("TEST 1 - CITED ANSWER")
print("########################################")

question1 = (
    "What evidence is required for project submission?"
)

result1 = answer_with_citations(
    question1
)

display_result(result1)


# ============================================
# VERIFY CITATION
# ============================================

verify_citation(
    result1,
    "[1]"
)


# ============================================
# TEST 2
# ============================================

print("\n\n########################################")
print("TEST 2 - CITATION QUESTION")
print("########################################")

question2 = (
    "How can citations map back to their sources?"
)

result2 = answer_with_citations(
    question2
)

display_result(result2)


# ============================================
# TEST 3
# NO-SOURCE FALLBACK
# ============================================

print("\n\n########################################")
print("TEST 3 - NO SOURCE FALLBACK")
print("########################################")

question3 = (
    "What is the population of Mars?"
)

result3 = answer_with_citations(
    question3
)

display_result(result3)