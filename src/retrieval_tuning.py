"""Compare retrieval settings against a small relevance test set."""

import json
from pathlib import Path
from typing import Callable

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.embedding_sanity import CHUNKS, EMBEDDING_MODEL, TEST_CASES, cosine_similarity


SETTINGS = [
    {"name": "baseline_top1", "k": 1, "min_score": 0.0},
    {"name": "wider_top2", "k": 2, "min_score": 0.0},
    {"name": "strict_score_0.70", "k": 3, "min_score": 0.70},
]


def retrieve(
    query: str,
    records: list[dict[str, object]],
    embed: Callable[[list[str]], list[list[float]]],
    *,
    k: int,
    min_score: float,
) -> list[dict[str, object]]:
    """Return up to k chunks whose cosine score meets the threshold."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    query_vector = embed([query])[0]
    ranked = []
    for record in records:
        ranked.append(
            {
                **record,
                "score": cosine_similarity(query_vector, record["embedding"]),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return [item for item in ranked if item["score"] >= min_score][:k]


def evaluate_setting(
    setting: dict[str, object],
    test_queries: list[dict[str, str]],
    records: list[dict[str, object]],
    embed: Callable[[list[str]], list[list[float]]],
) -> dict[str, object]:
    """Evaluate one retrieval setting and return detailed and aggregate results."""
    details = []
    for test in test_queries:
        retrieved = retrieve(
            test["query"],
            records,
            embed,
            k=setting["k"],
            min_score=setting["min_score"],
        )
        sources = [item["metadata"]["source"] for item in retrieved]
        hit = test["expected_source"] in sources
        details.append(
            {
                "query": test["query"],
                "expected_source": test["expected_source"],
                "returned_sources": sources,
                "top_score": round(retrieved[0]["score"], 4) if retrieved else None,
                "hit": hit,
            }
        )

    hits = sum(row["hit"] for row in details)
    return {
        "setting": setting["name"],
        "k": setting["k"],
        "min_score": setting["min_score"],
        "hit_rate": round(hits / len(details), 4),
        "hits": hits,
        "tests": len(details),
        "average_results": round(
            sum(len(row["returned_sources"]) for row in details) / len(details), 2
        ),
        "details": details,
    }


def choose_best(results: list[dict[str, object]]) -> dict[str, object]:
    """Prefer hit rate, then fewer returned chunks when scores tie."""
    return max(results, key=lambda row: (row["hit_rate"], -row["average_results"]))


def run_experiment(
    test_queries: list[dict[str, str]],
    chunks: list[dict[str, object]],
    embed: Callable[[list[str]], list[list[float]]],
) -> dict[str, object]:
    """Embed chunks once, compare all settings, and select the best one."""
    vectors = embed([chunk["text"] for chunk in chunks])
    records = [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]
    results = [evaluate_setting(setting, test_queries, records, embed) for setting in SETTINGS]
    best = choose_best(results)
    return {
        "model": EMBEDDING_MODEL,
        "test_queries": test_queries,
        "settings": results,
        "chosen_setting": best["setting"],
        "justification": (
            f"{best['setting']} achieved the highest hit rate of "
            f"{best['hit_rate']:.0%} with {best['average_results']:.2f} "
            "results returned per query."
        ),
    }


def main() -> None:
    report = run_experiment(TEST_CASES, CHUNKS, DefaultEmbeddingFunction())
    print("retrieval relevance tuning report")
    for result in report["settings"]:
        print(
            f"{result['setting']}: hit_rate={result['hit_rate']:.0%}, "
            f"average_results={result['average_results']}"
        )
    print(f"chosen: {report['chosen_setting']}")
    print(f"why: {report['justification']}")
    Path("results/retrieval_tuning_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("saved: results/retrieval_tuning_report.json")


if __name__ == "__main__":
    main()
