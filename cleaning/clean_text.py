import re
import unicodedata


def clean_text(text):
    # Normalize Unicode encoding
    text = unicodedata.normalize("NFKC", text)

    # Remove page markers such as "Page 1 of 10"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove common navigation/boilerplate text
    boilerplate = [
        r"^\s*table of contents\s*$",
        r"^\s*back to top\s*$",
        r"^\s*next page\s*$",
        r"^\s*previous page\s*$",
    ]

    for pattern in boilerplate:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    # Collapse more than two consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text