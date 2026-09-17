```python
import os
import json
from openai import OpenAI


# -----------------------------------
# 1. ENVIRONMENT CONFIGURATION
# -----------------------------------

API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BASE_URL = os.environ.get("OPENAI_BASE_URL")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")


# Create OpenAI client
if BASE_URL:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
else:
    client = OpenAI(
        api_key=API_KEY
    )


# -----------------------------------
# 2. PREPARED SAMPLE CORPUS
# -----------------------------------

chunks = [
    {
        "text": "Password reset instructions for learner accounts.",
        "metadata": {
            "source": "account-guide.md",
            "chunk_index": 0,
            "section": "Password Reset"
        }
    },
    {
        "text": "Learners can recover access using their registered email.",
        "metadata": {
            "source": "account-guide.md",
            "chunk_index": 1,
            "section": "Account Recovery"
        }
    },
    {
        "text": "Learners should contact support if they cannot access their account.",
        "metadata": {
            "source": "support-guide.md",
            "chunk_index": 0,
            "section": "Account Support"
        }
    }
]


# -----------------------------------
# 3. GENERATE EMBEDDINGS
# -----------------------------------

texts = [chunk["text"] for chunk in chunks]

response = client.embeddings.create(
    model=MODEL,
    input=texts
)


# -----------------------------------
# 4. STORE VECTOR + SOURCE + METADATA
# -----------------------------------

records = []

for chunk, item in zip(chunks, response.data):

    record = {
        "text": chunk["text"],
        "metadata": chunk["metadata"],
        "embedding": item.embedding
    }

    records.append(record)


# -----------------------------------
# 5. VALIDATE RESULTS
# -----------------------------------

if not records:
    raise ValueError("No embeddings were generated")


vector_length = len(records[0]["embedding"])

for record in records:
    if len(record["embedding"]) != vector_length:
        raise ValueError("Embedding dimensions do not match")


# -----------------------------------
# 6. PRINT VERIFICATION OUTPUT
# -----------------------------------

print("=" * 60)
print("EMBEDDING GENERATION SUMMARY")
print("=" * 60)

print("Model:", MODEL)
print("Chunks embedded:", len(records))
print("Vector length:", vector_length)

print("\nSample embedding:")
print(records[0]["embedding"][:5])

print("\nSample stored record:")
print("Text:", records[0]["text"])
print("Metadata:", records[0]["metadata"])
print("Vector length:", len(records[0]["embedding"]))
print("Vector sample:", records[0]["embedding"][:5])


# -----------------------------------
# 7. SAVE SAMPLE OUTPUT
# -----------------------------------

output = []

for record in records:
    output.append({
        "text": record["text"],
        "metadata": record["metadata"],
        "vector_length": len(record["embedding"]),
        "embedding_sample": record["embedding"][:5]
    })


with open("sample_embedding_output.json", "w", encoding="utf-8") as file:
    json.dump(output, file, indent=4)


print("\nSample output saved to:")
print("sample_embedding_output.json")

print("\nEmbedding generation completed successfully.")

