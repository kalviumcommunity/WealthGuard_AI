def fixed_chunks(text, size=300, overlap=50):
    chunks = []
    i = 0

    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap

    return chunks


def paragraph_chunks(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def average_size(chunks):
    if len(chunks) == 0:
        return 0

    return sum(len(chunk) for chunk in chunks) / len(chunks)


text = """
Data privacy is an important part of modern software systems. Organizations collect
large amounts of personal information from users. This information must be stored
and processed carefully.

Access control determines who can access specific resources. Users should only receive
the permissions required for their work. Strong authentication and authorization help
protect sensitive information.

Data encryption protects information by converting readable data into an encoded form.
Encryption should be used both when data is stored and when it is transferred between
systems.

Regular security audits help organizations identify weaknesses in their systems.
Audits can reveal configuration problems, outdated software, and inappropriate access
permissions.

Organizations should also maintain clear security policies. Employees need to
understand how data should be handled, stored, and shared. Good policies reduce the
chance of accidental data exposure.
"""


fixed = fixed_chunks(text, size=300, overlap=50)
paragraph = paragraph_chunks(text)


print("DOCUMENT CHUNKING COMPARISON")
print("----------------------------")

print("Fixed-size strategy:")
print("Chunk count:", len(fixed))
print("Average chunk size:", round(average_size(fixed), 2), "characters")

print()

print("Paragraph strategy:")
print("Chunk count:", len(paragraph))
print("Average chunk size:", round(average_size(paragraph), 2), "characters")

print()
print("SAMPLE FIXED-SIZE CHUNK:")
print(fixed[0])

print()
print("SAMPLE PARAGRAPH CHUNK:")
print(paragraph[0])


print()
print("JUSTIFICATION:")
print("Paragraph chunking is chosen because the document contains")
print("well-defined paragraphs, and each paragraph represents a")
print("complete idea. This helps retrieval preserve meaning and")
print("reduces the chance of cutting an idea in the middle.")