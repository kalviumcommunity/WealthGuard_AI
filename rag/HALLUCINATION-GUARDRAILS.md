# WealthGuard AI - Hallucination Guardrails

## Objective

The RAG pipeline should not generate answers when the retrieved
context is weak or unsupported.

A retrieval-quality guardrail was added before the generation step.

---

## Guardrail Design

The system uses cosine similarity between the query embedding
and retrieved document-chunk embeddings.

### Configuration

- Embedding model: `all-MiniLM-L6-v2`
- Vector database: ChromaDB
- Collection: `wealthguard_chunks`
- Top-K retrieval: 3
- Relevance threshold: `0.50`
- Minimum relevant chunks: 1

A retrieved chunk is considered relevant only when:

```text
cosine similarity >= 0.50