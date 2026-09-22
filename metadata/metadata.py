# ============================================
# 3.33 Metadata Filtering & Hybrid Search
# ============================================

# Task 1: Metadata-filtered retrieval

def retrieve(query, k=3, metadata_filter=None):
    query_vector = embed([query], model=EMBEDDING_MODEL)[0]

    return collection.search(
        vector=query_vector,
        top_k=k,
        filter=metadata_filter,
        include=["text", "metadata", "score"]
    )


# Query
query = "What are the password reset steps?"

# Unfiltered search
unfiltered = retrieve(query, k=3)

# Filtered search
filtered = retrieve(
    query,
    k=3,
    metadata_filter={"section": "Account access"}
)


# ============================================
# Task 2: Show filtered vs unfiltered results
# ============================================

def show_results(label, results):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    for i, item in enumerate(results, 1):
        print("\nResult", i)
        print("Score:", round(item["score"], 4))
        print("Source:", item["metadata"].get("source"))
        print("Section:", item["metadata"].get("section"))
        print("Text:", item["text"][:200])


show_results("UNFILTERED RESULTS", unfiltered)
show_results("FILTERED RESULTS", filtered)


# ============================================
# Task 3: Keyword Search
# ============================================

def keyword_score(text, keywords):
    lowered = text.lower()

    score = 0

    for word in keywords:
        if word.lower() in lowered:
            score += 1

    return score


# ============================================
# Hybrid Search
# ============================================

def hybrid_rank(
    vector_results,
    keywords,
    vector_weight=0.8,
    keyword_weight=0.2
):
    ranked = []

    for item in vector_results:

        lexical = keyword_score(
            item["text"],
            keywords
        )

        combined = (
            vector_weight * item["score"]
            + keyword_weight * lexical
        )

        new_item = item.copy()

        new_item["keyword_score"] = lexical
        new_item["hybrid_score"] = combined

        ranked.append(new_item)

    return sorted(
        ranked,
        key=lambda item: item["hybrid_score"],
        reverse=True
    )


# Run hybrid search
hybrid = hybrid_rank(
    filtered,
    keywords=["password", "reset"]
)


# ============================================
# Task 4: Show improved precision
# ============================================

show_results(
    "HYBRID FILTERED RESULTS",
    hybrid
)


print("\n" + "=" * 60)
print("PRECISION COMPARISON")
print("=" * 60)

print("Query:", query)
print("Filter: section = Account access")
print("Keywords: password, reset")

print("\nUnfiltered results:")
for item in unfiltered:
    print(
        "-",
        item["metadata"].get("section"),
        "|",
        item["text"][:100]
    )

print("\nFiltered results:")
for item in filtered:
    print(
        "-",
        item["metadata"].get("section"),
        "|",
        item["text"][:100]
    )

print("\nHybrid results:")
for item in hybrid:
    print(
        "- Hybrid score:",
        round(item["hybrid_score"], 4),
        "| Keyword score:",
        item["keyword_score"],
        "|",
        item["text"][:100]
    )


# ============================================
# Task 5: Sample filtered-search output
# ============================================

print("\n" + "=" * 60)
print("SAMPLE FILTERED SEARCH RESULTS")
print("=" * 60)

print("Query:", query)
print("Metadata Filter:", {"section": "Account access"})

for i, item in enumerate(filtered, 1):

    print("\n--- Result", i, "---")

    print("Score:", round(item["score"], 4))

    print(
        "Source:",
        item["metadata"].get("source")
    )

    print(
        "Section:",
        item["metadata"].get("section")
    )

    print(
        "Metadata:",
        item["metadata"]
    )

    print(
        "Text:",
        item["text"]
    )


# ============================================
# Final explanation
# ============================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print("""
Metadata filtering restricts vector search to the relevant
part of the database. This can improve precision by removing
irrelevant documents.

Vector search finds results based on semantic meaning, while
keyword search looks for exact words or phrases.

Hybrid search combines semantic similarity with keyword
matching. It is useful for exact names, IDs, error codes,
product names, and policy numbers.

For a password reset query, filtering by the
'Account access' section helps remove unrelated security
or IT documents.

A useful filter for a real RAG application could be:
user role, document type, department, date, product,
region, section, or access level.
""")