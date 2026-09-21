def to_vector_record(chunk):
    return {
        "id": chunk["id"],
        "vector": chunk["embedding"],
        "text": chunk["text"],
        "metadata": {
            "source": chunk["metadata"]["source"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "section": chunk["metadata"].get("section")
        }
    }


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


# Convert embedded chunks into vector records
records = [to_vector_record(chunk) for chunk in embedded_chunks]

# Insert records in batches
inserted = 0
failures = []

for batch in batches(records, 100):
    try:
        collection.upsert(batch)
        inserted += len(batch)
    except Exception as error:
        failures.append({
            "batch_start_id": batch[0]["id"],
            "error": str(error)
        })


# Verify count
expected_count = len(embedded_chunks)
indexed_count = collection.count()

print("===== INDEXING SUMMARY =====")
print("Expected chunks:", expected_count)
print("Inserted this run:", inserted)
print("Indexed count:", indexed_count)
print("Failures:", failures)

if indexed_count == expected_count:
    print("COUNT VALIDATION: PASSED")
else:
    print("COUNT VALIDATION: FAILED")


# Spot-check first record
sample = embedded_chunks[0]
stored = collection.get(sample["id"])

print("\n===== SPOT CHECK =====")
print("ID:", stored["id"])
print("Source:", stored["metadata"]["source"])
print("Chunk index:", stored["metadata"]["chunk_index"])
print("Section:", stored["metadata"].get("section"))

print("Text matches:",
      stored["text"] == sample["text"])

print("Metadata source matches:",
      stored["metadata"]["source"] == sample["metadata"]["source"])

print("Vector length matches:",
      len(stored["vector"]) == len(sample["embedding"]))

if (
    stored["id"] == sample["id"]
    and stored["text"] == sample["text"]
    and stored["metadata"]["source"] == sample["metadata"]["source"]
    and stored["metadata"]["chunk_index"] == sample["metadata"]["chunk_index"]
    and len(stored["vector"]) == len(sample["embedding"])
):
    print("SPOT CHECK: PASSED")
else:
    print("SPOT CHECK: FAILED")