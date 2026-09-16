import tiktoken

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")


def token_chunks(text, size=100, overlap=15):
    tokens = enc.encode(text)
    chunks = []
    i = 0

    while i < len(tokens):
        chunk_tokens = tokens[i:i + size]
        chunks.append(enc.decode(chunk_tokens))
        i += size - overlap

    return chunks


def print_stats(name, chunks):
    token_sizes = [len(enc.encode(chunk)) for chunk in chunks]

    print(name)
    print("Chunk count:", len(chunks))
    print("Average tokens:", round(sum(token_sizes) / len(token_sizes), 2))
    print()


# Same document used for all tests
text = """
Data privacy is an important part of modern software systems. Organizations
collect large amounts of personal information from users. This information
must be stored and processed carefully.

Access control determines who can access specific resources. Users should only
receive the permissions required for their work. Strong authentication and
authorization help protect sensitive information.

Data encryption protects information by converting readable data into an
encoded form. Encryption should be used both when data is stored and when it
is transferred between systems.

Regular security audits help organizations identify weaknesses in their
systems. Audits can reveal configuration problems, outdated software, and
inappropriate access permissions.

Organizations should also maintain clear security policies. Employees need to
understand how data should be handled, stored, and shared. Good policies
reduce the chance of accidental data exposure.
"""


# -------------------------------------------------
# TASK 1 & 2: Token size and controlled overlap
# -------------------------------------------------

CHUNK_SIZE = 100
OVERLAP = 15

chunks_with_overlap = token_chunks(
    text,
    size=CHUNK_SIZE,
    overlap=OVERLAP
)

chunks_without_overlap = token_chunks(
    text,
    size=CHUNK_SIZE,
    overlap=0
)


# -------------------------------------------------
# TASK 3: Show effect of overlap
# -------------------------------------------------

print("TOKEN-AWARE CHUNKING")
print("====================")
print()

print("Settings:")
print("Chunk size:", CHUNK_SIZE, "tokens")
print("Overlap:", OVERLAP, "tokens")
print()

print_stats("WITHOUT OVERLAP", chunks_without_overlap)
print_stats("WITH OVERLAP", chunks_with_overlap)


print("SAMPLE CHUNK WITHOUT OVERLAP")
print("----------------------------")

for i, chunk in enumerate(chunks_without_overlap[:2]):
    print("Chunk", i + 1)
    print(chunk)
    print("Tokens:", len(enc.encode(chunk)))
    print()


print("SAMPLE CHUNK WITH OVERLAP")
print("--------------------------")

for i, chunk in enumerate(chunks_with_overlap[:2]):
    print("Chunk", i + 1)
    print(chunk)
    print("Tokens:", len(enc.encode(chunk)))
    print()


# -------------------------------------------------
# Demonstrate boundary context
# -------------------------------------------------

boundary_text = """
A security audit found that several employees had unnecessary access
permissions. The security team immediately reviewed all user roles and
removed permissions that were not required for their jobs.
"""

print("BOUNDARY CONTEXT DEMONSTRATION")
print("===============================")

print("Without overlap:")

no_overlap_boundary = token_chunks(
    boundary_text,
    size=20,
    overlap=0
)

for i, chunk in enumerate(no_overlap_boundary):
    print("Chunk", i + 1, ":", chunk)

print()
print("With overlap:")

overlap_boundary = token_chunks(
    boundary_text,
    size=20,
    overlap=8
)

for i, chunk in enumerate(overlap_boundary):
    print("Chunk", i + 1, ":", chunk)


# -------------------------------------------------
# TASK 4: Justification
# -------------------------------------------------

print()
print("JUSTIFICATION")
print("=============")
print(
    "A chunk size of 100 tokens is used for this demonstration because "
    "it keeps each retrieved chunk small enough to combine multiple chunks "
    "within a model context window."
)

print(
    "An overlap of 15 tokens is approximately 15 percent of the chunk size. "
    "This provides boundary context without duplicating too much text."
)

print(
    "Larger overlap can improve context preservation but also increases "
    "the number of repeated tokens, storage requirements, and embedding cost."
)

print(
    ""
    "retrieval top-k value used by the application."
)