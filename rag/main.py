import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "rag_chunks"
)

app = FastAPI(
    title="RAG Backend API",
    description="Backend API for a Retrieval-Augmented Generation system",
    version="1.0.0"
)


# -----------------------------
# Request model
# -----------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000
    )


# -----------------------------
# Source model
# -----------------------------

class Source(BaseModel):
    source: str
    chunk_id: str | None = None
    score: float | None = None


# -----------------------------
# Response model
# -----------------------------

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: str


# -----------------------------
# RAG pipeline
# -----------------------------

def guarded_answer(question: str):
    """
    This function represents the RAG pipeline.

    Replace this function with your existing
    RAG pipeline code if you already have one.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty")

    # Example grounded answer
    answer = (
        "The submission requires a PR link, "
        "sample output, and a video explanation."
    )

    sources = [
        {
            "source": "submission-rubric.md",
            "chunk_id": "submission-rubric.md:2",
            "score": 0.84
        }
    ]

    return {
        "answer": answer,
        "sources": sources,
        "status": "answered"
    }


# -----------------------------
# Root endpoint
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "RAG Backend API is running"
    }


# -----------------------------
# Query endpoint
# -----------------------------

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    try:

        # Call RAG pipeline
        result = guarded_answer(request.question)

        # Return structured JSON
        return {
            "answer": result["answer"],

            "sources": [
                {
                    "source": source.get("source"),
                    "chunk_id": source.get("chunk_id"),
                    "score": source.get("score")
                }
                for source in result.get("sources", [])
            ],

            "status": result.get(
                "status",
                "answered"
            )
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="RAG service failed"
        )