"""
Data Validation Utility
========================
Validates incoming CSV files for schema completeness and encoding
before they enter the RAG ingestion pipeline.
"""

import csv
import os


EXPECTED_COLUMNS = ["id", "title", "content", "source", "date"]


def validate_csv(filepath: str) -> dict:
    """
    Validate a CSV file for:
      - File existence
      - UTF-8 encoding
      - Required column presence
      - Non-zero row count

    Returns a dict with 'valid' (bool) and 'errors' (list[str]).
    """
    errors = []

    # Check file exists
    if not os.path.isfile(filepath):
        return {"valid": False, "errors": [f"File not found: {filepath}"]}

    # Check encoding
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            # Check required columns
            missing = [c for c in EXPECTED_COLUMNS if c not in headers]
            if missing:
                errors.append(f"Missing columns: {missing}")

            # Check row count
            rows = list(reader)
            if len(rows) == 0:
                errors.append("File has zero data rows")

    except UnicodeDecodeError:
        errors.append("File is not valid UTF-8")

    return {"valid": len(errors) == 0, "errors": errors}


if __name__ == "__main__":
    # Quick smoke test with a non-existent file
    result = validate_csv("data/test.csv")
    print(f"Validation result: {result}")
