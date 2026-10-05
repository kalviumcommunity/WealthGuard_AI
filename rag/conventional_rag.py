
# 3.42 Conversational RAG & Follow-Up Context
# Copy-paste-ready demonstration

# ---------------------------------------------------------
# 1. SAMPLE KNOWLEDGE BASE
# ---------------------------------------------------------

documents = [
    {
        "id": "doc1",
        "content": "Project submission requires a GitHub PR link, sample output, and a 3-5 minute video explanation."
    },
    {
        "id": "doc2",
        "content": "The video explanation should demonstrate the implementation, explain the important concepts, and show the sample dialogue."
    },
    {
        "id": "doc3",
        "content": "The GitHub repository must be public and the pull request must remain open at submission time."
    },
    {
        "id": "doc4",
        "content": "For conversational RAG, follow-up questions should be rewritten into standalone queries before retrieval."
    },
    {
        "id": "doc6",
        "content": "RAG answers should be grounded in retrieved source chunks instead of relying only on conversation memory."
    },
    {
        "id": "doc5",
        "content": "Long conversations should use a short rolling history and summarize older turns to avoid token-limit problems."
    }
]


# ---------------------------------------------------------
# 2. CONVERSATION HISTORY
# ---------------------------------------------------------

history = []


# ---------------------------------------------------------
# 3. ADD MESSAGE TO HISTORY
# ---------------------------------------------------------

def add_to_history(role, content):
    history.append({
        "role": role,
        "content": content
    })


# ---------------------------------------------------------
# 4. GET RECENT HISTORY
# ---------------------------------------------------------

def get_recent_history(limit=4):
    return history[-limit:]


# ---------------------------------------------------------
# 5. REWRITE FOLLOW-UP QUESTION
# ---------------------------------------------------------

def rewrite_followup(history, question):

    # Simple rule-based rewriting for demonstration.
    # In a real RAG system, an LLM would perform this step.

    if not history:
        return question

    previous_question = ""
    previous_answer = ""

    for message in reversed(history):
        if message["role"] == "assistant" and previous_answer == "":
            previous_answer = message["content"]

        if message["role"] == "user" and previous_question == "":
            previous_question = message["content"]

        if previous_question and previous_answer:
            break

    question_lower = question.lower()

    # Handle common follow-up references
    if "video" in question_lower:
        return "What video explanation is required for project submission?"

    if "github" in question_lower or "pr" in question_lower:
        return "What GitHub PR requirements are required for project submission?"

    if "deadline" in question_lower:
        return "What is the project submission deadline?"

    if "long conversation" in question_lower:
        return "How should long conversational RAG conversations handle history and token limits?"

    if "grounded" in question_lower:
        return "How can long conversations remain grounded in retrieved source evidence?"

    # Generic fallback
    return previous_question + " - " + question


# ---------------------------------------------------------
# 6. SIMPLE RETRIEVAL
# ---------------------------------------------------------

def retrieve(query, k=3):

    query_words = set(query.lower().split())

    results = []

    for document in documents:

        document_words = set(document["content"].lower().split())

        score = len(query_words.intersection(document_words))

        if score > 0:
            results.append({
                "score": score,
                "id": document["id"],
                "content": document["content"]
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:k]


# ---------------------------------------------------------
# 7. CHECK RETRIEVAL QUALITY
# ---------------------------------------------------------

def retrieval_is_strong(chunks):

    if not chunks:
        return False

    return chunks[0]["score"] > 0


# ---------------------------------------------------------
# 8. GENERATE GROUNDED ANSWER
# ---------------------------------------------------------

def generate_grounded_answer(question, chunks):

    if not chunks:
        return {
            "answer": "I don't have enough reliable context to answer that.",
            "sources": []
        }

    answer_parts = []

    for chunk in chunks:
        answer_parts.append(chunk["content"])

    answer = " ".join(answer_parts)

    return {
        "answer": answer,
        "sources": [chunk["id"] for chunk in chunks]
    }


# ---------------------------------------------------------
# 9. CONVERSATIONAL RAG
# ---------------------------------------------------------

def conversational_answer(user_question):

    # Keep only recent history for rewriting
    recent_history = get_recent_history(4)

    # Step 1: Rewrite follow-up
    standalone_query = rewrite_followup(
        recent_history,
        user_question
    )

    # Step 2: Retrieve using rewritten query
    chunks = retrieve(
        standalone_query,
        k=3
    )

    # Step 3: Check retrieval
    if not retrieval_is_strong(chunks):

        answer = "I don't have enough reliable context to answer that."

        sources = []

    else:

        result = generate_grounded_answer(
            user_question,
            chunks
        )

        answer = result["answer"]
        sources = result["sources"]

    # Step 4: Store conversation
    add_to_history(
        "user",
        user_question
    )

    add_to_history(
        "assistant",
        answer
    )

    # Step 5: Return complete result
    return {
        "rewritten_query": standalone_query,
        "retrieved_context": chunks,
        "answer": answer,
        "sources": sources
    }


# ---------------------------------------------------------
# 10. DISPLAY RESULT
# ---------------------------------------------------------

def show_result(result):

    print("\n==============================")
    print("REWRITTEN QUERY")
    print("==============================")
    print(result["rewritten_query"])

    print("\n==============================")
    print("RETRIEVED CONTEXT")
    print("==============================")

    for chunk in result["retrieved_context"]:
        print(
            f"[{chunk['id']}] "
            f"Score: {chunk['score']} "
            f"-> {chunk['content']}"
        )

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print(result["answer"])

    print("\n==============================")
    print("SOURCES")
    print("==============================")
    print(result["sources"])


# ---------------------------------------------------------
# 11. MULTI-TURN CONVERSATION
# ---------------------------------------------------------

print("\n\n===== MULTI-TURN CONVERSATIONAL RAG =====")


# Turn 1
question1 = "What evidence is required for project submission?"

result1 = conversational_answer(question1)

print("\nUSER:", question1)
show_result(result1)


# Turn 2 - FOLLOW-UP
question2 = "What about the video?"

result2 = conversational_answer(question2)

print("\n\nUSER:", question2)
show_result(result2)


# Turn 3 - FOLLOW-UP
question3 = "What about the GitHub PR?"

result3 = conversational_answer(question3)

print("\n\nUSER:", question3)
show_result(result3)


# Turn 4 - FOLLOW-UP
question4 = "How should long conversations be handled?"

result4 = conversational_answer(question4)

print("\n\nUSER:", question4)
show_result(result4)


# ---------------------------------------------------------
# 12. SHOW COMPLETE CONVERSATION HISTORY
# ---------------------------------------------------------

print("\n\n===================================")
print("COMPLETE CONVERSATION HISTORY")
print("===================================")

for message in history:

    print(
        message["role"].upper() + ":",
        message["content"]
    )


# ---------------------------------------------------------
# 13. TOKEN / HISTORY MANAGEMENT
# ---------------------------------------------------------

def manage_history(max_turns=4):

    """
    Keeps only the most recent conversation turns.
    This prevents the conversation history from becoming
    too large.
    """

    global history

    max_messages = max_turns * 2

    if len(history) > max_messages:

        old_messages = history[:-max_messages]

        summary = "Earlier conversation contained previous questions and answers."

        history = [
            {
                "role": "system",
                "content": summary
            }
        ] + history[-max_messages:]


# ---------------------------------------------------------
# 14. GROUNDING STRATEGY
# ---------------------------------------------------------

def keep_conversation_grounded(question):

    """
    Long conversations should not rely only on chat memory.

    The current question is rewritten into a standalone query,
    fresh retrieval is performed, and the answer is generated
    from retrieved source chunks.
    """

    recent_history = get_recent_history(4)

    standalone_query = rewrite_followup(
        recent_history,
        question
    )

    chunks = retrieve(
        standalone_query,
        k=3
    )

    if not retrieval_is_strong(chunks):

        return "I don't have enough reliable source context to answer that."

    result = generate_grounded_answer(
        question,
        chunks
    )

    return result["answer"]


# ---------------------------------------------------------
# END
# ---------------------------------------------------------

### What this code demonstrates
