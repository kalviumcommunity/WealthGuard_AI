import json
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]
TEST_SET_PATH = BASE_DIR / "rag/evaluation/test_set.json"
RESULTS_PATH = BASE_DIR / "rag/evaluation/evaluation_results.json"
CITATION_PATH = BASE_DIR / "rag/evaluation/citation_checks.txt"
SUMMARY_PATH = BASE_DIR / "rag/evaluation/evaluation_summary.md"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "wealthguard_chunks"
VECTOR_DB_PATH = str(BASE_DIR / "vector_db/chroma_data")

TOP_K = 3

REFUSAL_MESSAGE = (
    "I don't have enough verified information "
    "to answer this reliably."
)


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_test_set():
    with open(TEST_SET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def retrieve_chunks(question, model, collection):
    embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i] or {}

        chunks.append({
            "text": document,
            "source": metadata.get("source_document", "unknown"),
            "section": metadata.get("section", "unknown"),
            "page": metadata.get("page", "unknown"),
            "distance": results["distances"][0][i]
        })

    return chunks


def generate_answer(question, chunks, client):
    if not chunks:
        return REFUSAL_MESSAGE

    context_parts = []

    for chunk in chunks:
        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Section: {chunk['section']}\n"
            f"Page: {chunk['page']}\n"
            f"Content: {chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are WealthGuard AI. Answer only using "
                    "the supplied context. Do not invent facts. "
                    "If the context does not support an answer, "
                    "say you do not have enough verified information. "
                    "Cite supporting documents using [Source: filename]. "
                    "Keep the answer concise."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Context:\n{context}"
                )
            }
        ],
        max_tokens=150
    )

    content = response.choices[0].message.content

    if not content:
        return REFUSAL_MESSAGE

    return content.strip()


def score_answer(test_case, answer, chunks):
    normalized_answer = normalize(answer)
    normalized_context = normalize(
        " ".join(chunk["text"] for chunk in chunks)
    )

    expected_keywords = test_case["expected_keywords"]

    if test_case["should_answer"]:
        correct_matches = sum(
            keyword.lower() in normalized_answer
            for keyword in expected_keywords
        )

        correctness = (
            correct_matches / len(expected_keywords)
            if expected_keywords else 0
        )

        supported_matches = sum(
            keyword.lower() in normalized_context
            for keyword in expected_keywords
        )

        grounding = (
            supported_matches / len(expected_keywords)
            if expected_keywords else 0
        )

        answered = not any(
            phrase in normalized_answer
            for phrase in [
                "not enough verified information",
                "cannot determine",
                "insufficient information"
            ]
        )
    else:
        refusal_phrases = [
            "not enough verified information",
            "not available",
            "cannot determine",
            "insufficient information"
        ]

        answered = not any(
            phrase in normalized_answer
            for phrase in refusal_phrases
        )

        correctness = 0 if answered else 1

        grounding = (
            0 if answered and not chunks else 1
        )

    expected_sources = set(test_case["expected_sources"])
    retrieved_sources = {
        chunk["source"] for chunk in chunks
    }

    cited_sources = set(
        re.findall(
            r"\[Source:\s*([^\]]+)\]",
            answer,
            flags=re.IGNORECASE
        )
    )

    if test_case["should_answer"]:
        source_retrieval = (
            len(expected_sources & retrieved_sources)
            / len(expected_sources)
            if expected_sources else 0
        )

        citation_precision = (
            len(cited_sources & expected_sources)
            / len(cited_sources)
            if cited_sources else 0
        )

        citation_recall = (
            len(cited_sources & expected_sources)
            / len(expected_sources)
            if expected_sources else 0
        )

        citation_quality = (
            (citation_precision + citation_recall) / 2
        )
    else:
        source_retrieval = 1 if not chunks else 0
        citation_quality = 1 if not cited_sources else 0

    return {
        "correctness": round(correctness, 2),
        "grounding": round(grounding, 2),
        "citation_quality": round(citation_quality, 2),
        "source_retrieval": round(source_retrieval, 2),
        "cited_sources": sorted(cited_sources),
        "retrieved_sources": sorted(retrieved_sources),
        "expected_sources": sorted(expected_sources),
        "answered": answered
    }


def main():
    load_dotenv(BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not configured.")

    if not os.getenv("OPENAI_MODEL"):
        raise ValueError("OPENAI_MODEL is not configured.")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1"
        )
    )

    model = SentenceTransformer(MODEL_NAME)

    db = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    collection = db.get_collection(name=COLLECTION_NAME)

    test_cases = load_test_set()

    results = []
    citation_lines = []

    for test_case in test_cases:
        print(f"\nEvaluating {test_case['id']}...")

        chunks = retrieve_chunks(
            test_case["question"],
            model,
            collection
        )

        answer = generate_answer(
            test_case["question"],
            chunks,
            client
        )

        scores = score_answer(
            test_case,
            answer,
            chunks
        )

        result = {
            "id": test_case["id"],
            "question": test_case["question"],
            "expected_answer": test_case["expected_answer"],
            "actual_answer": answer,
            "retrieved_chunks": chunks,
            **scores
        }

        results.append(result)

        print("Answer:", answer)
        print("Scores:", scores)

        citation_lines.append(
            f"{test_case['id']} - {test_case['question']}\n"
            f"Expected sources: {scores['expected_sources']}\n"
            f"Retrieved sources: {scores['retrieved_sources']}\n"
            f"Cited sources: {scores['cited_sources']}\n"
            f"Citation quality: {scores['citation_quality']}\n"
            f"{'-' * 60}"
        )

    count = len(results)

    avg_correctness = sum(
        item["correctness"] for item in results
    ) / count

    avg_grounding = sum(
        item["grounding"] for item in results
    ) / count

    avg_citation = sum(
        item["citation_quality"] for item in results
    ) / count

    avg_retrieval = sum(
        item["source_retrieval"] for item in results
    ) / count

    summary = {
        "test_cases": count,
        "average_correctness": round(avg_correctness, 2),
        "average_grounding": round(avg_grounding, 2),
        "average_citation_quality": round(avg_citation, 2),
        "average_source_retrieval": round(avg_retrieval, 2),
        "evaluation_method": (
            "Keyword-based correctness and grounding checks, "
            "plus expected-source citation matching."
        )
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "results": results},
            file,
            indent=2
        )

    with open(CITATION_PATH, "w", encoding="utf-8") as file:
        file.write("\n\n".join(citation_lines))

    failures = [
        item for item in results
        if item["correctness"] < 1
        or item["grounding"] < 1
        or item["citation_quality"] < 1
    ]

    report = f"""# WealthGuard AI — RAG Evaluation Summary

## Evaluation Overview

- Test cases: {count}
- Average correctness: {avg_correctness:.2%}
- Average grounding: {avg_grounding:.2%}
- Average citation quality: {avg_citation:.2%}
- Average source retrieval: {avg_retrieval:.2%}

## Evaluation Method

Correctness and grounding use expected-keyword checks.
Citation quality compares cited filenames with expected source files.
Source retrieval measures whether expected documents were retrieved.

These are automated proxy metrics, not a complete semantic or human evaluation.

## Notable Failures

"""

    if failures:
        for item in failures:
            report += (
                f"- **{item['id']}**: "
                f"Correctness={item['correctness']:.2f}, "
                f"Grounding={item['grounding']:.2f}, "
                f"Citation={item['citation_quality']:.2f}. "
                f"Review retrieved context, answer, and citation output. "
                f"Possible causes include retrieval mismatch, "
                f"missing source metadata, answer wording variation, "
                f"or unsupported generation.\n"
            )
    else:
        report += "No failures detected by the automated checks.\n"

    report += """
## Limitations

- Keyword matching may miss semantically correct paraphrases.
- A cited filename does not prove that every claim is supported.
- Similarity retrieval can return relevant-looking but insufficient context.
- Results depend on the current test set, database, and model response.
- Manual review is required before treating scores as final quality evidence.
"""

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    print("\n========== FINAL EVALUATION ==========")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    print(RESULTS_PATH)
    print(CITATION_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()