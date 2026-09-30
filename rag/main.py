
import uuid
from pathlib import Path

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from cleaning.clean_text import clean_text
from chunking.token_chunker import chunk_text


# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "vector_db" / "chroma_data"

COLLECTION_NAME = "wealthguard_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".txt", ".md"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="WealthGuard AI Document API",
    description="Upload, index, and search approved documents.",
    version="2.0.0",
)

# Load shared resources once when the API starts
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/")
def home():
    return {
        "message": "WealthGuard AI API is running",
        "collection": COLLECTION_NAME,
        "indexed_chunks": collection.count(),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if not original_name:
        raise HTTPException(400, "Filename is required.")

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            415,
            "Unsupported file type. Upload .txt or .md files.",
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, "File exceeds the 5 MB limit.")

        if not content.strip():
            raise HTTPException(400, "Uploaded file is empty.")

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                400,
                "File must contain valid UTF-8 text.",
            )

        cleaned = clean_text(text)

        if not cleaned:
            raise HTTPException(
                400,
                "No usable text found after cleaning.",
            )

        chunks = chunk_text(cleaned)

        if not chunks:
            raise HTTPException(
                400,
                "Document could not be divided into chunks.",
            )

        upload_id = str(uuid.uuid4())
        stored_name = f"{upload_id}{extension}"
        stored_path = UPLOAD_DIR / stored_name

        # Save the original uploaded bytes under a generated filename.
        stored_path.write_bytes(content)

        try:
            embeddings = model.encode(chunks).tolist()

            ids = [
                f"{upload_id}_{index}"
                for index in range(len(chunks))
            ]

            metadatas = [
                {
                    "source_document": original_name,
                    "upload_id": upload_id,
                    "chunk_index": index,
                    "section": "Uploaded Document",
                }
                for index in range(len(chunks))
            ]

            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        except Exception:
            stored_path.unlink(missing_ok=True)
            raise

        return {
            "status": "success",
            "message": "Document indexed successfully.",
            "filename": original_name,
            "upload_id": upload_id,
            "chunks_created": len(chunks),
            "total_indexed_chunks": collection.count(),
            "searchable_immediately": True,
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            500,
            "Document processing failed. Check server logs.",
        )

    finally:
        await file.close()


@app.post("/query")
def query_document(request: QueryRequest):
    try:
        query_embedding = model.encode(
            request.question
        ).tolist()

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.top_k, collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        matches = []

        if result["ids"] and result["ids"][0]:
            for index, chunk_id in enumerate(result["ids"][0]):
                matches.append({
                    "chunk_id": chunk_id,
                    "text": result["documents"][0][index],
                    "source": result["metadatas"][0][index].get(
                        "source_document",
                        "Unknown",
                    ),
                    "similarity": round(
                        1 - result["distances"][0][index],
                        4,
                    ),
                })

        return {
            "question": request.question,
            "results": matches,
            "status": "success" if matches else "no_results",
        }

    except Exception:
        raise HTTPException(
            500,
            "Document search failed. Check server logs.",
        )