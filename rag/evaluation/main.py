```python
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


# ============================================================
# 1. CACHE
# ============================================================

query_cache = {}

CACHE_TTL_SECONDS = 15 * 60


def cache_key(question, filters=None):
    raw = {
        "question": question.strip().lower(),
        "filters": filters or {}
    }

    return hashlib.sha256(
        str(raw).encode("utf-8")
    ).hexdigest()


def get_cached_answer(question, filters=None):

    key = cache_key(question, filters)

    cached = query_cache.get(key)

    if not cached:
        return None

    if time.time() - cached["created_at"] > CACHE_TTL_SECONDS:
        query_cache.pop(key, None)
        return None

    return cached["response"]


def save_cached_answer(question, response, filters=None):

    key = cache_key(question, filters)

    query_cache[key] = {
        "created_at": time.time(),
        "response": response
    }


# ============================================================
# 2. STRUCTURED LOGGING
# ============================================================

logger = logging.getLogger("rag_app")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

log_records = []


def log_rag_request(record):

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": record["request_id"],
        "question": record["question"],
        "answer_preview": record["answer"][:180],
        "sources": record["sources"],
        "cache_hit": record["cache_hit"],
        "input_tokens": record["input_tokens"],
        "output_tokens": record["output_tokens"],
        "estimated_cost": record["estimated_cost"],
        "latency_ms": record["latency_ms"],
        "error": record.get("error")
    }

    log_records.append(log_data)

    logger.info(
        json.dumps(log_data)
    )


# ============================================================
# 3. TOKEN AND COST TRACKING
# ============================================================

MODEL_INPUT_COST_PER_1K = 0.00015
MODEL_OUTPUT_COST_PER_1K = 0.00060


def estimate_tokens(text):

    # Simple approximate token calculation
    return max(1, len(text) // 4)


def estimate_cost(input_tokens, output_tokens):

    input_cost = (
        input_tokens / 1000
    ) * MODEL_INPUT_COST_PER_1K

    output_cost = (
        output_tokens / 1000
    ) * MODEL_OUTPUT_COST_PER_1K

    return round(
        input_cost + output_cost,
        6
    )


# ============================================================
# 4. USAGE SUMMARY
# ============================================================

def summarize_usage(records):

    total_requests = len(records)

    cache_hits = sum(
        1
        for item in records
        if item["cache_hit"]
    )

    total_cost = sum(
        item["estimated_cost"]
        for item in records
    )

    total_input_tokens = sum(
        item["input_tokens"]
        for item in records
    )

    total_output_tokens = sum(
        item["output_tokens"]
        for item in records
    )

    total_latency = sum(
        item["latency_ms"]
        for item in records
    )

    average_latency = (
        total_latency /
        max(total_requests, 1)
    )

    return {
        "total_requests": total_requests,

        "cache_hits": cache_hits,

        "cache_misses": (
            total_requests - cache_hits
        ),

        "cache_hit_rate": round(
            cache_hits /
            max(total_requests, 1),
            2
        ),

        "total_input_tokens":
            total_input_tokens,

        "total_output_tokens":
            total_output_tokens,

        "total_estimated_cost":
            round(total_cost, 6),

        "average_latency_ms":
            round(average_latency, 2)
    }


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    question: str


# ============================================================
# SAMPLE RAG PIPELINE
# ============================================================

def rag_pipeline(question):

    # Simulated retrieved sources

    sources = [
        {
            "id": "source-1",
            "label": "[1]",
            "document": "refund-policy.md",
            "chunk_id": "refund-policy-03",
            "text": (
                "Refund requests must be submitted "
                "within 14 days."
            )
        },
        {
            "id": "source-2",
            "label": "[2]",
            "document": "support-guide.md",
            "chunk_id": "support-guide-02",
            "text": (
                "Customers can contact support "
                "for help with refund requests."
            )
        }
    ]

    answer = (
        "According to the refund policy, "
        "refund requests must be submitted "
        "within 14 days [1]. "
        "Customers can contact support "
        "for help with refund requests [2]."
    )

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# QUERY ENDPOINT
# ============================================================

@app.post("/query")
def query(request: QueryRequest):

    start_time = time.time()

    request_id = str(uuid.uuid4())

    question = request.question

    error = None

    try:

        # ----------------------------------------------------
        # CHECK CACHE
        # ----------------------------------------------------

        cached_response = get_cached_answer(
            question
        )

        if cached_response is not None:

            cache_hit = True

            response = cached_response

        else:

            cache_hit = False

            # Run RAG pipeline
            response = rag_pipeline(
                question
            )

            # Save result
            save_cached_answer(
                question,
                response
            )

        # ----------------------------------------------------
        # TOKEN COUNT
        # ----------------------------------------------------

        input_tokens = estimate_tokens(
            question
        )

        output_tokens = estimate_tokens(
            response["answer"]
        )

        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------

        estimated_cost = estimate_cost(
            input_tokens,
            output_tokens
        )

        # ----------------------------------------------------
        # LATENCY
        # ----------------------------------------------------

        latency_ms = round(
            (time.time() - start_time) * 1000,
            2
        )

        # ----------------------------------------------------
        # LOG REQUEST
        # ----------------------------------------------------

        log_rag_request({

            "request_id": request_id,

            "question": question,

            "answer": response["answer"],

            "sources": response["sources"],

            "cache_hit": cache_hit,

            "input_tokens": input_tokens,

            "output_tokens": output_tokens,

            "estimated_cost": estimated_cost,

            "latency_ms": latency_ms,

            "error": error
        })

        # ----------------------------------------------------
        # RETURN RESPONSE
        # ----------------------------------------------------

        return {
            "request_id": request_id,

            "answer": response["answer"],

            "sources": response["sources"],

            "usage": {

                "input_tokens":
                    input_tokens,

                "output_tokens":
                    output_tokens,

                "estimated_cost":
                    estimated_cost,

                "cache_hit":
                    cache_hit,

                "latency_ms":
                    latency_ms
            }
        }

    except Exception as e:

        error = str(e)

        latency_ms = round(
            (time.time() - start_time) * 1000,
            2
        )

        log_rag_request({

            "request_id": request_id,

            "question": question,

            "answer": "",

            "sources": [],

            "cache_hit": False,

            "input_tokens": 0,

            "output_tokens": 0,

            "estimated_cost": 0,

            "latency_ms": latency_ms,

            "error": error
        })

        return {
            "request_id": request_id,
            "error": "Something went wrong."
        }


# ============================================================
# USAGE REPORT ENDPOINT
# ============================================================

@app.get("/usage")
def usage_report():

    summary = summarize_usage(
        log_records
    )

    return summary


# ============================================================
# ALL LOGS ENDPOINT
# ============================================================

@app.get("/logs")
def get_logs():

    return {
        "logs": log_records
    }


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message":
            "RAG Caching, Logging and Monitoring API"
    }
```

### Install

```bash
pip install fastapi uvicorn
```

### Run

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

### Test Task 1 — Cache

Send this request **twice**:

```json
{
  "question": "What is the refund policy?"
}
```

First response should contain:

```json
"cache_hit": false
```

Send exactly the same question again:

```json
{
  "question": "What is the refund policy?"
}
```

Now it should contain:

```json
"cache_hit": true
```

That demonstrates that the second identical query was served from cache.

### Test Task 2 — Logs

Open:

```text
http://127.0.0.1:8000/logs
```

You will see records containing:

```text
timestamp
request_id
question
answer_preview
sources
cache_hit
input_tokens
output_tokens
estimated_cost
latency_ms
error
```

This matches the assignment's requirement to log the request, answer, sources, cache status, errors, and timestamp.

### Test Task 3 — Cost

Every response contains:

```json
"usage": {
  "input_tokens": 7,
  "output_tokens": 31,
  "estimated_cost": 0.000019,
  "cache_hit": false,
  "latency_ms": 1.2
}
```

The token count is an **approximate estimate**, which is explicitly allowed by the assignment.

### Test Task 4 — Usage Report

Open:

```text
http://127.0.0.1:8000/usage
```

Example:

```json
{
  "total_requests": 3,
  "cache_hits": 1,
  "cache_misses": 2,
  "cache_hit_rate": 0.33,
  "total_input_tokens": 25,
  "total_output_tokens": 93,
  "total_estimated_cost": 0.000059,
  "average_latency_ms": 0.85
}
```

### Task 5 — Sample logs

After making a few requests, your terminal will show JSON logs similar to:

```json
{
  "timestamp": "2026-10-01T08:30:12.123456",
  "request_id": "abc123",
  "question": "What is the refund policy?",
  "answer_preview": "According to the refund policy...",
  "sources": [
    {
      "id": "source-1",
      "label": "[1]",
      "document": "refund-policy.md",
      "chunk_id": "refund-policy-03"
    }
  ],
  "cache_hit": false,
  "input_tokens": 7,
  "output_tokens": 31,
  "estimated_cost": 0.000019,
  "latency_ms": 1.2,
  "error": null
}
```

**Important:** this is a standalone working demonstration. If your existing RAG project already has `retrieve_context()` and your actual LLM generation code, replace only the `rag_pipeline()` function with your existing RAG pipeline. The caching/logging/monitoring structure can remain.
