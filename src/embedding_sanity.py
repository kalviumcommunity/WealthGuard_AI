"""Smoke-test embedding relevance before trusting retrieval results."""

import json
from pathlib import Path
from typing import Callable

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNKS = [
    {
        "text": "The Secure Growth Plan has an annual management fee of 1.2 percent.",
        "metadata": {"source": "product_brochure.txt", "chunk_index": 0},
    },
    {
        "text": "The Secure Growth Plan has a three-year lock-in period.",
        "metadata": {"source": "product_brochure.txt", "chunk_index": 1},
    },
    {
        "text": "Capital protection is not guaranteed. Review the applicable product terms before investing.",
        "metadata": {"source": "compliance.txt", "chunk_index": 0},
    },
    {
        "text": "The mountain weather forecast predicts heavy snowfall tomorrow.",
        "metadata": {"source": "weather.txt", "chunk_index": 0},
    },
]

TEST_CASES = [
    {
        "query": "What is the annual management fee for the Secure Growth Plan?",
        "expected_source": "product_brochure.txt",
        "note": "Specific fee query should retrieve the product fee chunk.",
    },
    {
        "query": "How long is the Secure Growth Plan lock-in period?",
        "expected_source": "product_brochure.txt",
        "note": "Specific lock-in query should retrieve the product terms chunk.",
    },
    {
        "query": "Is capital protection guaranteed before investing?",
        "expected_source": "compliance.txt",
        "note": "Risk and guarantee language should retrieve the compliance chunk.",
    },
    {
        "query": "What are the applicable product terms?",
        "expected_source": "product_brochure.txt",
        "note": "Surprising case: broad wording may rank the compliance warning first.",
    },
]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Return cosine similarity for two non-zero vectors."""
    a = np.asarray(vector_a)
    b = np.asarray(vector_b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def rank_chunks(
    query: str,
    chunk_records: list[dict[str, object]],
    embed: Callable[[list[str]], list[list[float]]],
) -> list[dict[str, object]]:
    """Rank chunk records with the same embedding function used for the query."""
    query_embedding = embed([query])[0]
    ranked = []
    for chunk in chunk_records:
        ranked.append(
            {
                **chunk,
                "score": cosine_similarity(query_embedding, chunk["embedding"]),
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def run_sanity_tests(
    test_cases: list[dict[str, str]],
    chunks: list[dict[str, object]],
    embed: Callable[[list[str]], list[list[float]]],
) -> list[dict[str, object]]:
    """Embed the corpus once, rank every test query, and return a report."""
    embeddings = embed([chunk["text"] for chunk in chunks])
    records = [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, embeddings)]
    report = []

    for case in test_cases:
        ranked = rank_chunks(case["query"], records, embed)
        top = ranked[0]
        top_source = top["metadata"]["source"]
        passed = top_source == case["expected_source"]
        report.append(
            {
                "query": case["query"],
                "expected_source": case["expected_source"],
                "top_source": top_source,
                "top_score": round(top["score"], 4),
                "passed": passed,
                "note": case["note"] if not passed else "Expected source ranked first.",
            }
        )

    return report


def print_report(report: list[dict[str, object]]) -> None:
    """Print a compact sanity report for a human review."""
    passed = sum(row["passed"] for row in report)
    failed = len(report) - passed
    print("embedding sanity report")
    print(f"model: {EMBEDDING_MODEL}")
    print(f"tests: {len(report)} passed: {passed} failed: {failed}")
    for row in report:
        print(json.dumps(row, ensure_ascii=True))


def main() -> None:
    embed = DefaultEmbeddingFunction()
    report = run_sanity_tests(TEST_CASES, CHUNKS, embed)
    print_report(report)
    Path("results/embedding_sanity_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("saved: results/embedding_sanity_report.json")


if __name__ == "__main__":
    main()