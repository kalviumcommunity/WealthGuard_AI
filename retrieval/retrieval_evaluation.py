```python
# Retrieval Evaluation & Recall Testing

# --------------------------------------------------
# Task 1: Labelled query set
# --------------------------------------------------

labelled_queries = [
    {
        "query": "How can a learner reset their password?",
        "relevant_chunk_ids": {
            "account-guide.md:0",
            "account-guide.md:1"
        }
    },
    {
        "query": "What evidence is required for project submission?",
        "relevant_chunk_ids": {
            "submission-rubric.md:2"
        }
    },
    {
        "query": "How can I update my account details?",
        "relevant_chunk_ids": {
            "account-guide.md:3"
        }
    },
    {
        "query": "What should be included in a project submission?",
        "relevant_chunk_ids": {
            "submission-rubric.md:2",
            "submission-rubric.md:3"
        }
    }
]


# --------------------------------------------------
# Example retriever
# Replace this function with your actual retriever
# --------------------------------------------------

def retrieve(query, k=5):
    # Example retrieved chunks.
    # These represent the chunks returned by the retriever.
    
    example_results = {
        "How can a learner reset their password?": [
            {"id": "account-guide.md:0"},
            {"id": "account-guide.md:1"},
            {"id": "account-guide.md:3"},
            {"id": "faq.md:0"},
            {"id": "help.md:1"}
        ],

        "What evidence is required for project submission?": [
            {"id": "submission-rubric.md:2"},
            {"id": "submission-rubric.md:3"},
            {"id": "account-guide.md:0"},
            {"id": "faq.md:1"},
            {"id": "help.md:2"}
        ],

        "How can I update my account details?": [
            {"id": "account-guide.md:0"},
            {"id": "account-guide.md:1"},
            {"id": "faq.md:0"},
            {"id": "help.md:1"},
            {"id": "submission-rubric.md:2"}
        ],

        "What should be included in a project submission?": [
            {"id": "submission-rubric.md:2"},
            {"id": "submission-rubric.md:3"},
            {"id": "submission-rubric.md:4"},
            {"id": "faq.md:0"},
            {"id": "help.md:1"}
        ]
    }

    return example_results.get(query, [])[:k]


# --------------------------------------------------
# Task 2 & 3: Measure recall@k and precision@k
# --------------------------------------------------

def evaluate_query(item, k=5):

    results = retrieve(item["query"], k=k)

    retrieved_ids = [result["id"] for result in results]

    relevant = item["relevant_chunk_ids"]

    hits = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant
    ]

    # Recall = relevant chunks retrieved / total relevant chunks
    recall = len(hits) / len(relevant)

    # Precision = relevant retrieved / total retrieved
    precision = (
        len(hits) / len(retrieved_ids)
        if retrieved_ids
        else 0
    )

    return {
        "query": item["query"],
        "retrieved_ids": retrieved_ids,
        "relevant_chunk_ids": sorted(relevant),
        "hits": hits,
        "recall": recall,
        "precision": precision
    }


# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

rows = [
    evaluate_query(item, k=5)
    for item in labelled_queries
]


# --------------------------------------------------
# Task 4: Aggregate results
# --------------------------------------------------

avg_recall = (
    sum(row["recall"] for row in rows)
    / len(rows)
)

avg_precision = (
    sum(row["precision"] for row in rows)
    / len(rows)
)

failures = [
    row
    for row in rows
    if row["recall"] < 1.0
]


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("===================================")
print("RETRIEVAL EVALUATION RESULTS")
print("===================================")

print("Queries:", len(rows))
print("Recall@5:", round(avg_recall, 3))
print("Precision@5:", round(avg_precision, 3))

print("\nIndividual Results:")

for row in rows:

    print("\nQuery:", row["query"])

    print("Expected:",
          row["relevant_chunk_ids"])

    print("Retrieved:",
          row["retrieved_ids"])

    print("Hits:",
          row["hits"])

    print("Recall:",
          round(row["recall"], 3))

    print("Precision:",
          round(row["precision"], 3))


# --------------------------------------------------
# Failure analysis
# --------------------------------------------------

print("\n===================================")
print("FAILURE ANALYSIS")
print("===================================")

if not failures:

    print("No retrieval failures found.")

else:

    for failure in failures:

        print("\nFailed query:")
        print(failure["query"])

        print("Expected chunks:")
        print(failure["relevant_chunk_ids"])

        print("Retrieved chunks:")
        print(failure["retrieved_ids"])

        print("Likely causes:")
        print("- Chunking may separate important information.")
        print("- Query wording may not match the stored text.")
        print("- Top-k may be too small.")
        print("- Metadata filtering may be missing.")
        print("- Embeddings may not capture the query meaning well.")


# --------------------------------------------------
# Improvement plan
# --------------------------------------------------

print("\n===================================")
print("HOW TO IMPROVE RECALL")
print("===================================")

print("1. Increase k to retrieve more chunks.")
print("2. Improve document chunking.")
print("3. Rewrite queries before retrieval.")
print("4. Add metadata filtering.")
print("5. Use hybrid keyword + vector search.")
print("6. Use better embeddings.")
print("7. Add re-ranking after retrieval.")


{
    "query": "How can a learner reset their password?",
    "relevant_chunk_ids": {
        "account-guide.md:0",
        "account-guide.md:1"
    }
}
