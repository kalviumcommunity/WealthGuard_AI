# Document Chunking Strategy Justification

## Assignment: Document Chunking Strategies

### Overview
This document justifies the choice of **paragraph-based chunking** for the WealthGuard AI corpus, comparing it against fixed-size and sentence-based alternatives.

---

## Corpus Analysis

### Document Types
The WealthGuard AI corpus contains structured financial documents:
- **Policy documents** (Markdown) - Clear section headers and paragraph breaks
- **Compliance notes** (Plain text) - Standalone warning statements
- **Product information** (HTML) - Structured content with paragraph tags

### Corpus Characteristics
- **Total size**: 366 characters (sample corpus)
- **Document count**: 3 documents
- **Structure**: Well-defined with natural paragraph boundaries
- **Content**: Financial policies, compliance warnings, product terms

---

## Chunking Strategies Compared

### 1. Fixed-Size Chunking (200 chars, 30 overlap)

**Implementation:**
```python
def fixed_chunks(text, size=200, overlap=30):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks
```

**Results:**
- Chunk count: 3
- Average size: 140.7 characters
- Size range: 26 - 200 characters

**Advantages:**
- Uniform chunk sizes
- Predictable storage and retrieval costs
- Overlap helps preserve boundary context

**Disadvantages:**
- Cuts sentences and ideas mid-stream
- May split complete concepts across chunks
- Doesn't respect document structure
- Can create chunks with incomplete information

**Example Issue:**
```
Chunk 1: "...lock-in period is three years.</p></body></html>
Compliance note: Capital protection is not guaranteed. Customers should"
Chunk 2: "review the applicable terms before investing.
# Secure Growth Plan"
```
The transition from HTML to compliance text to policy header creates unnatural boundaries.

---

### 2. Sentence-Based Chunking

**Implementation:**
```python
def sentence_chunks(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
```

**Results:**
- Chunk count: 4
- Average size: 90.0 characters
- Size range: 33 - 202 characters

**Advantages:**
- Respects sentence boundaries
- Complete grammatical units
- Good for factoid queries

**Disadvantages:**
- Too granular for complex queries
- Loses surrounding context
- May separate related sentences
- HTML content becomes a single large chunk

**Example Issue:**
```
Chunk 1: "Customers should review the applicable terms before investing."
Chunk 2: "# Secure Growth Plan
The annual management fee is 1.2 percent."
```
The header and its content are separated, losing structural context.

---

### 3. Paragraph-Based Chunking (CHOSEN)

**Implementation:**
```python
def paragraph_chunks(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]
```

**Results:**
- Chunk count: 4
- Average size: 89.5 characters
- Size range: 20 - 146 characters

**Advantages:**
- Respects natural document structure
- Each chunk is a complete idea
- Preserves context within paragraphs
- Aligns with how documents are organized
- Clear boundaries for source attribution

**Disadvantages:**
- Uneven chunk sizes
- Some paragraphs may be very short
- Less predictable than fixed-size

**Example Output:**
```
Chunk 1: "<!doctype html>...<p>The lock-in period is three years.</p>..."
Chunk 2: "Compliance note: Capital protection is not guaranteed. Customers should review the applicable terms before investing."
Chunk 3: "# Secure Growth Plan"
Chunk 4: "The annual management fee is 1.2 percent. The minimum investment is 50,000."
```
Each chunk represents a complete, self-contained unit of information.

---

## Justification for Paragraph-Based Chunking

### 1. Document Structure Alignment
WealthGuard documents are **naturally organized by paragraphs**:
- Policy documents use Markdown headers and paragraph breaks
- Compliance notes are standalone paragraphs
- HTML content uses `<p>` tags for information blocks

Paragraph chunking respects this organization rather than imposing an artificial structure.

### 2. Meaning Preservation
**Critical for financial applications:**
- Each paragraph contains a complete idea or statement
- No risk of cutting a sentence or concept in half
- Complete information is preserved for retrieval
- Reduces chance of missing relevant information

**Example:** The compliance warning about capital protection is kept intact:
```
"Compliance note: Capital protection is not guaranteed. Customers should review the applicable terms before investing."
```
A fixed-size chunker might split this into "Capital protection is not guaranteed." and "Customers should review...", losing the complete warning context.

### 3. Retrieval Precision
**WealthGuard query patterns:**
- "What is the management fee?" → Matches paragraph about fees
- "What is the lock-in period?" → Matches paragraph about lock-in
- "Is capital protection guaranteed?" → Matches compliance paragraph

Paragraph chunks provide **sufficient context** without noise:
- Not too small (like sentences) → preserves relationship between related statements
- Not too large (like whole documents) → precise retrieval
- Natural boundaries → easier to understand and verify

### 4. Context Window Efficiency
**Calculation for typical RAG scenario:**
- Context window: 4096 tokens (common for production models)
- Average paragraph: ~90 chars ≈ 20-25 tokens
- Top-k retrieval: 5 chunks
- Query: ~50 tokens
- Response: ~200 tokens
- **Total: (25 × 5) + 50 + 200 = 375 tokens** ✅ Well within budget

This allows:
- Retrieving multiple diverse chunks
- Combining information from different sources
- Leaving room for longer responses
- Supporting more complex queries

### 5. Source Attribution
**Paragraph boundaries make citation clear:**
- Each chunk corresponds to a document section
- Easy to map back to original document
- Natural boundaries for citation
- Transparent for human verification

### 6. Corpus-Suitability
**WealthGuard corpus characteristics:**
- Small, well-structured documents
- Clear paragraph organization
- Each paragraph is informationally complete
- No extremely long paragraphs (>1000 chars)

For this corpus, paragraph chunking is **optimal**. For other corpora (e.g., unstructured legal documents with very long paragraphs), a hybrid approach might be better.

---

## Trade-Offs Considered

### Why Not Fixed-Size?
- ❌ Cuts ideas mid-sentence
- ❌ Doesn't respect document structure
- ❌ May create chunks with incomplete information
- ❌ Harder to attribute to specific sections
- ✅ Only advantage: uniform size

### Why Not Sentence-Based?
- ❌ Too granular for financial context
- ❌ Loses relationship between related sentences
- ❌ May separate headers from content
- ❌ Less efficient for context window
- ✅ Only advantage: complete grammatical units

### Why Paragraph-Based?
- ✅ Respects document structure
- ✅ Preserves complete ideas
- ✅ Suitable size for retrieval
- ✅ Clear source attribution
- ✅ Fits query patterns
- ⚠️ Only trade-off: uneven sizes (acceptable for this corpus)

---

## Chunk Boundaries and Answer Quality

### Scenario: Query about "management fee"

**With fixed-size chunking:**
```
Chunk: "...management fee is 1.2 percent. The minimum..."
```
If "management fee" appears at the boundary, the complete answer "1.2 percent" might be split across chunks, making retrieval unreliable.

**With paragraph chunking:**
```
Chunk: "The annual management fee is 1.2 percent. The minimum investment is 50,000."
```
The complete statement is preserved, ensuring retrieval can match and return the full answer.

### Scenario: Query about "lock-in period"

**With fixed-size chunking (HTML):**
```
Chunk 1: "...<h1>Lock-in period</h1><p>The lock-in period is three years.</p>..."
```
If chunked mid-tag, the HTML structure is broken, and the information may not be properly extracted.

**With paragraph chunking:**
```
Chunk: "<!doctype html>...<p>The lock-in period is three years.</p>..."
```
The HTML paragraph is preserved as a complete unit, maintaining both structure and content.

---

## Chunk Size and Context Window Relationship

### The Formula
```
(chunk_size × top_k) + query_tokens + response_tokens ≤ context_window
```

### Practical Example
**Current setup (paragraph-based):**
- Chunk size: ~25 tokens (90 chars)
- Top-k: 5 chunks
- Query: 50 tokens
- Response: 200 tokens
- **Total: 375 tokens** out of 4096 → 9% usage

**Alternative (fixed-size 500 chars):**
- Chunk size: ~120 tokens
- Top-k: 5 chunks
- Query: 50 tokens
- Response: 200 tokens
- **Total: 850 tokens** out of 4096 → 21% usage

**Implications:**
- Smaller chunks allow more chunks in context → more diverse information
- Larger chunks limit context diversity → deeper but narrower information
- Optimal size balances precision with context richness
- Must be tuned based on retrieval tests and query patterns

### Why This Matters for WealthGuard
1. **Multi-source queries**: Users may ask questions requiring information from policies, compliance, and product docs simultaneously
2. **Complex responses**: Financial advice often requires combining multiple pieces of information
3. **Safety margins**: Context window should not be at maximum to allow for longer, more nuanced responses
4. **Model variations**: Different models have different context windows; smaller chunks provide flexibility

---

## Conclusion

**Paragraph-based chunking is the optimal choice for WealthGuard AI because:**

1. **Aligns with document structure** - Respects natural organization of policies, compliance notes, and product information
2. **Preserves meaning** - Each chunk is a complete idea, reducing risk of incomplete information
3. **Supports retrieval patterns** - Matches how users query financial information
4. **Efficient for context window** - Allows multiple chunks without overwhelming the model
5. **Enables clear attribution** - Easy to trace answers back to specific document sections
6. **Fits corpus characteristics** - Well-suited to the small, structured WealthGuard corpus

The trade-off of uneven chunk sizes is acceptable given the significant advantages in meaning preservation and retrieval quality for this specific use case.

---

## Files for Review

1. **`chunking/assignment_chunking.py`** - Complete implementation with all three strategies
2. **`chunking/chunking_assignment_samples.txt`** - Sample chunks from each strategy
3. **`chunking/CHUNKING_JUSTIFICATION.md`** - This justification document

Run the script to see live comparison:
```bash
python chunking/assignment_chunking.py
```
