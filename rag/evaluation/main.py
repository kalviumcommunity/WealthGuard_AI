import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


async def rag_pipeline_stream(question: str):

    # Sample retrieved source
    sources = [
        {
            "id": "source-1",
            "label": "[1]",
            "document": "refund-policy.md",
            "chunk_id": "refund-policy-03",
            "text": "Refund requests must be submitted within 14 days."
        },
        {
            "id": "source-2",
            "label": "[2]",
            "document": "support-guide.md",
            "chunk_id": "support-guide-02",
            "text": "Customers can contact support for help with refund requests."
        }
    ]

    # Send citations first
    yield {
        "type": "citations",
        "sources": sources
    }

    # Sample answer
    answer = (
        "According to the refund policy, refund requests must be "
        "submitted within 14 days [1]. If you need help with the "
        "refund process, you can contact customer support [2]."
    )

    # Stream answer word by word
    words = answer.split(" ")

    for i, word in enumerate(words):
        if i == len(words) - 1:
            text = word
        else:
            text = word + " "

        yield {
            "type": "token",
            "text": text
        }

        await asyncio.sleep(0.08)


@app.post("/query/stream")
async def stream_query(request: QueryRequest):

    async def events():

        try:

            async for event in rag_pipeline_stream(request.question):

                yield f"data: {json.dumps(event)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception:

            error_event = {
                "type": "error",
                "message": "The answer stopped streaming. Please retry."
            }

            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream"
    )


@app.get("/")
def home():
    return {
        "message": "Streaming RAG API is running"
    }