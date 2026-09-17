```python
import os
import json
from openai import OpenAI


# Get configuration from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
BASE_URL = os.getenv("OPENAI_BASE_URL")


# Check API key
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY")


# Create client
if BASE_URL:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
else:
    client = OpenAI(
        api_key=API_KEY
    )


# Small prepared corpus
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


# Get text from all chunks
texts = []

for chunk in chunks:
    texts.append(chunk["text"])


# Generate embeddings
response = client.embeddings.create(
    model=MODEL,
    input=texts
)


# Store embeddings with text and metadata
records = []

for i in range(len(chunks)):
    records.append({
        "text": chunks[i]["text"],
        "metadata": chunks[i]["metadata"],
        "embedding": response.data[i].embedding
    })


# Check that embeddings were created
if len(records) == 0:
    raise ValueError("No embeddings were generated")


# Get vector dimension
vector_length = len(records[0]["embedding"])


# Verify every vector has the same dimension
for record in records:
    if len(record["embedding"]) != vector_length:
        raise ValueError("Embedding dimensions do not match")


# Print summary
print("========================================")
print("EMBEDDING GENERATION")
print("========================================")

print("Model:", MODEL)
print("Chunks embedded:", len(records))
print("Vector length:", vector_length)


# Print sample
print("\n========================================")
print("SAMPLE EMBEDDING")
print("========================================")

print("Text:", records[0]["text"])
print("Metadata:", records[0]["metadata"])
print("Vector length:", len(records[0]["embedding"]))
print("First 5 values:", records[0]["embedding"][:5])


# Save sample output
sample_output = []

for record in records:
    sample_output.append({
        "text": record["text"],
        "metadata": record["metadata"],
        "vector_length": len(record["embedding"]),
        "vector_sample": record["embedding"][:5]
    })


with open("sample_embedding_output.json", "w") as file:
    json.dump(sample_output, file, indent=4)


print("\nSample output saved to sample_embedding_output.json")
print("Embedding generation completed successfully.")
