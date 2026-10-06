# WealthGuard AI

**Evidence-Based AI Assistant for Wealth Management**

An AI-powered Retrieval-Augmented Generation (RAG) application enabling relationship managers to access accurate, consistent, and current information from approved organizational documents.

---

## Problem

Wealth divisions manage extensive documentation (policies, regulations, product brochures, compliance guidelines). Relationship managers manually search these resources when responding to customer inquiries, leading to:
- Inconsistent information delivery
- Outdated document reliance
- Extended response times
- Compliance and operational risks

---

## Solution

WealthGuard AI enables natural-language queries against a vetted knowledge base, retrieving approved information and generating responses with source citations. Example:

**Query:** "What is the current tax treatment of Product X?"

**Response:** "Based on the latest approved documentation, Product X follows the tax treatment specified in the applicable tax policy."

**Sources:**
- Tax Policy 2026 — Section 4.2
- Product X Brochure — Page 8

---

## Key Features

- **Document Upload & Indexing** — Upload documents (TXT, MD) that are automatically chunked, embedded, and indexed
- **Semantic Search** — Query the knowledge base using natural language
- **Retrieval-Augmented Generation** — Grounds responses in approved organizational documents
- **Source Citations** — References original sources for verification with similarity scores
- **Chunk Metadata & Source Tracking** — Track document source, chunk index, and section information
- **Human-in-the-Loop** — Supports decision-making without replacing professional judgment

---

## Technology Stack

**Backend:** Python, FastAPI

**Frontend:** Streamlit

**AI & RAG:**
- **Embedding Model:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database:** ChromaDB (persistent storage)
- **Chunking Strategy:** Token-based with overlap (100 tokens, 15 overlap)

**Document Processing:**
- **Supported Formats:** TXT, MD (Markdown)
- **Text Cleaning:** Normalization and preprocessing
- **Chunking:** Token-aware splitting with configurable overlap

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kalviumcommunity/WealthGuard_AI.git
cd WealthGuard_AI
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration (see Configuration section below)
```

### Running the Application

The application consists of two parts: the FastAPI backend and the Streamlit frontend.

#### Option 1: Run Both Services (Recommended for Development)

**Terminal 1 - Start the FastAPI Backend:**
```bash
cd rag
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
Loading embedding model...
Collection loaded: wealthguard_chunks
Existing chunks: 0
INFO:     Application startup complete.
```

**Terminal 2 - Start the Streamlit Frontend:**
```bash
streamlit run streamlit_app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

#### Option 2: Run Backend Only (API Mode)

If you only need the API endpoints:
```bash
cd rag
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Health check: `http://localhost:8000/`
- Upload document: `http://localhost:8000/upload`
- Query endpoint: `http://localhost:8000/query`

---

## Configuration

### Environment Variables

Create a `.env` file from `.env.example` and configure the following:

```bash
# API Configuration
RAG_API_URL=http://localhost:8000

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Database Configuration
VECTOR_DB_PATH=./vector_db/chroma_data
VECTOR_COLLECTION=wealthguard_chunks

# Application Configuration
MAX_FILE_SIZE=5242880  # 5MB in bytes
ALLOWED_EXTENSIONS=.txt,.md
CHUNK_SIZE=100
CHUNK_OVERLAP=15

# FastAPI Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# Streamlit Configuration
STREAMLIT_HOST=localhost
STREAMLIT_PORT=8501
```

### Optional: OpenAI API (for Chat Completion)

If you want to use OpenAI for chat completion (optional features):
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

**⚠️ Important:** Never commit `.env` files with real API keys to version control. Use `.env.example` as a template and keep actual secrets in your local environment or deployment platform's secret management.

---

## Usage

### 1. Upload and Index Documents

**Via API:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@policy.md"
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Document indexed successfully.",
  "filename": "policy.md",
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_created": 4,
  "total_indexed_chunks": 4,
  "searchable_immediately": true
}
```

**Supported file types:** `.txt`, `.md` (Markdown)
**Maximum file size:** 5 MB

### 2. Query the Knowledge Base

**Via Streamlit UI:**
1. Open http://localhost:8501 in your browser
2. Enter your question in the text area
3. Adjust "Sources to retrieve" slider (default: 3)
4. Click "Search the knowledge base"
5. View the grounded answer and retrieved sources

**Via API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the annual management fee?",
    "top_k": 3
  }'
```

**Expected Response:**
```json
{
  "question": "What is the annual management fee?",
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000_0",
      "text": "The annual management fee is 1.2 percent.",
      "source": "policy.md",
      "metadata": {
        "source_document": "policy.md",
        "upload_id": "550e8400-e29b-41d4-a716-446655440000",
        "chunk_index": 0,
        "section": "Uploaded Document"
      },
      "similarity": 0.9123
    }
  ],
  "status": "success"
}
```

### 3. View Source Citations

Each result includes:
- **Chunk ID:** Unique identifier for the specific chunk
- **Source Document:** Original filename
- **Similarity Score:** Cosine similarity (0-1, higher is better)
- **Metadata:** Upload ID, chunk index, section information
- **Text Preview:** The actual chunk content

---

## End-to-End Demo

### Step 1: Upload a Document

Create a sample document `refund-policy.md`:
```markdown
# Refund Policy

Refund requests must be submitted within 14 days of purchase.

To request a refund, customers must contact customer support
with their order number and reason for the refund.

Refunds are processed within 5-7 business days.
```

Upload it via the API:
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@refund-policy.md"
```

### Step 2: Ask a Question

**Question:** "What is the refund window?"

**Via Streamlit UI:**
1. Enter the question in the text area
2. Click "Search the knowledge base"

**Via API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the refund window?",
    "top_k": 3
  }'
```

### Step 3: View the Grounded Answer

**Response:**
```
Grounded answer:
Refund requests must be submitted within 14 days of purchase.

Retrieved sources:
1. refund-policy.md · 550e8400-e29b-41d4-a716-446655440000_0 · Similarity 0.945
   "Refund requests must be submitted within 14 days of purchase."
```

This confirms that:
- ✅ Document upload and indexing works
- ✅ Chunking and embedding works
- ✅ Semantic retrieval works
- ✅ Source citation works
- ✅ The full flow is reproducible

---

## API Documentation

### Health Check
```http
GET /
```

Returns API status and collection information.

### Upload Document
```http
POST /upload
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Document file to upload

**Response:** JSON with upload status, document ID, and chunk count

### Query Knowledge Base
```http
POST /query
Content-Type: application/json
```

**Body:**
```json
{
  "question": "Your question here",
  "top_k": 3
}
```

**Response:** JSON with retrieved chunks, sources, and similarity scores

---

## Development Tools

### Document Processing Demo

Run the document loader on the sample corpus:
```bash
python -m src.document_loader data/sample_corpus
```

### Chunk Metadata Demo

View chunk metadata and source tracking:
```bash
python -m src.chunk_metadata data/sample_corpus
```

### Embedding Quality Checks

Test embedding quality with known queries:
```bash
python -m src.embedding_sanity
```

Results saved to: `results/embedding_sanity_report.json`

### Retrieval Tuning

Compare retrieval settings:
```bash
python -m src.retrieval_tuning
```

Results saved to: `results/retrieval_tuning_report.json`

---

## Project Structure

```
WealthGuard_AI/
├── rag/                    # FastAPI backend
│   ├── main.py            # Main API application
│   └── evaluation/        # RAG evaluation tools
├── src/                   # Utility modules
│   ├── document_loader.py # Multi-format document intake
│   ├── chunk_metadata.py  # Chunk tracking and citations
│   ├── embeddings.py      # Embedding functions
│   └── chat_completion.py # OpenAI chat completion
├── chunking/              # Chunking strategies
│   ├── chunking.py        # Fixed-size and paragraph chunking
│   ├── token_chunker.py   # Token-aware chunking
│   └── assignment_chunking.py  # Chunking assignment
├── cleaning/              # Text cleaning utilities
├── data/                  # Data directory
│   └── sample_corpus/     # Sample documents for testing
├── vector_db/             # ChromaDB storage (gitignored)
├── uploads/               # Uploaded documents (gitignored)
├── streamlit_app.py       # Streamlit frontend
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

---

## Responsible AI

WealthGuard AI follows an **evidence-first approach** because it is intended for use in a financial environment.

The system is designed to:
* Prioritize approved organizational documentation
* Provide supporting sources for generated responses
* Flag potential conflicts or insufficient information
* Avoid unsupported financial recommendations
* Keep final customer-specific decisions with qualified human professionals

---

## Security Considerations

### Secrets Management
- Never commit `.env` files with real credentials
- Use environment variables for all sensitive configuration
- For production, use your cloud provider's secret management service
- Regularly rotate API keys and credentials

### File Upload Security
- Maximum file size: 5 MB
- Allowed extensions: `.txt`, `.md` only
- Files are validated before processing
- Uploaded files are stored in `uploads/` directory

### API Security (Future Enhancements)
- Add authentication (API keys, OAuth)
- Implement rate limiting
- Add request validation and sanitization
- Enable HTTPS for production deployments

---

## Deployment

### Development Deployment

For local development, follow the Quick Start section above.

### Production Deployment

**Recommended Steps:**

1. **Use environment variables** for all configuration
2. **Deploy backend** to a cloud service (e.g., Railway, Render, AWS ECS)
3. **Deploy frontend** to a static hosting service (e.g., Streamlit Cloud, Vercel)
4. **Use managed vector database** (e.g., ChromaDB Cloud, Pinecone, Qdrant Cloud)
5. **Enable HTTPS** for all endpoints
6. **Set up monitoring** (logs, metrics, alerts)
7. **Configure CI/CD** for automated testing and deployment

**Example Docker Setup (Future):**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Troubleshooting

### Backend won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check the `.env` file exists and is configured correctly

### Frontend can't connect to backend
- Verify the backend is running: `curl http://localhost:8000/`
- Check `RAG_API_URL` in `.env` matches the backend URL
- Check browser console for CORS errors

### Document upload fails
- Verify file is under 5 MB
- Check file extension is `.txt` or `.md`
- Ensure file contains valid UTF-8 text
- Check the backend logs for error details

### No results returned
- Verify documents have been uploaded and indexed
- Check the query is specific enough
- Try increasing `top_k` parameter
- Review the similarity scores in the response

---

## Project Goals

1. Improve consistency in information provided by relationship managers
2. Reduce reliance on outdated or incorrect documentation
3. Reduce the time required to locate relevant information
4. Provide transparent and source-backed AI responses
5. Support compliance-oriented information retrieval
6. Assist relationship managers while preserving human decision-making

---

## Future Enhancements

* Multi-format document support (PDF, DOCX)
* Multilingual support
* Voice-based interaction
* Automated document version comparison
* Document expiry notifications
* Advanced compliance workflows
* Improved retrieval and reranking
* Audit and usage analytics
* Integration with internal banking systems
* Real-time document synchronization
* Advanced caching mechanisms

---

## Team

### Nishant
**Project Team Member**

### Nikunj
**Project Team Member**

### Subhadeep Samanta
**Project Team Member**

The project was developed collaboratively by the team, with responsibilities distributed across application development, Retrieval-Augmented Generation, document processing, and system integration.

---

## Project Vision

> **Provide relationship managers with the right information, from the right approved source, at the right time — while keeping professional judgment with the human.**

**WealthGuard AI** aims to make wealth-management information retrieval more **accurate, consistent, transparent, and efficient** through Retrieval-Augmented Generation.

---

## License

This project is developed for educational purposes as part of the Kalvium program.

---

## Support

For issues, questions, or contributions, please contact the development team or open an issue in the repository.
