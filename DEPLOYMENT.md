# Deployment Guide

This guide provides step-by-step instructions for running the WealthGuard AI application locally and deploying it to production.

---

## Local Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Terminal or command prompt

### Step 1: Clone the Repository

```bash
git clone https://github.com/kalviumcommunity/WealthGuard_AI.git
cd WealthGuard_AI
```

### Step 2: Create Virtual Environment

```bash
# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Expected output: Dependencies will be installed including FastAPI, Streamlit, ChromaDB, Sentence Transformers, etc.

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file with your preferred configuration. For local development, the defaults in `.env.example` should work:

```bash
RAG_API_URL=http://localhost:8000
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DB_PATH=./vector_db/chroma_data
VECTOR_COLLECTION=wealthguard_chunks
MAX_FILE_SIZE=5242880
ALLOWED_EXTENSIONS=.txt,.md
CHUNK_SIZE=100
CHUNK_OVERLAP=15
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STREAMLIT_HOST=localhost
STREAMLIT_PORT=8501
```

### Step 5: Start the Backend

Open a terminal and run:

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

### Step 6: Start the Frontend

Open a second terminal and run:

```bash
streamlit run streamlit_app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Step 7: Access the Application

1. Open your browser and navigate to: http://localhost:8501
2. You should see the WealthGuard AI interface
3. The backend API is available at: http://localhost:8000

---

## Running Services Individually

### Backend Only (API Mode)

If you only need the API endpoints without the UI:

```bash
cd rag
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Test the API:
```bash
# Health check
curl http://localhost:8000/

# Expected response:
# {"message":"WealthGuard AI API is running","collection":"wealthguard_chunks","indexed_chunks":0}
```

### Frontend Only (UI Mode)

If the backend is already running elsewhere:

1. Ensure the `RAG_API_URL` in `.env` points to the running backend
2. Start the frontend:
```bash
streamlit run streamlit_app.py
```

---

## Testing the Full Flow

### 1. Upload a Document

Create a test file `test-document.md`:

```markdown
# Investment Policy

The minimum investment amount is $50,000.

The annual management fee is 1.2 percent.

Early withdrawal is subject to a 2% penalty.
```

Upload it via the API:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@test-document.md"
```

Expected response:
```json
{
  "status": "success",
  "message": "Document indexed successfully.",
  "filename": "test-document.md",
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_created": 3,
  "total_indexed_chunks": 3,
  "searchable_immediately": true
}
```

### 2. Query the Knowledge Base

**Via API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the minimum investment?",
    "top_k": 3
  }'
```

**Via Streamlit UI:**
1. Open http://localhost:8501
2. Enter: "What is the minimum investment?"
3. Click "Search the knowledge base"
4. View the grounded answer and sources

### 3. Verify the Response

Expected response should include:
- Grounded answer from the document
- Source citation with filename
- Chunk ID
- Similarity score

---

## Production Deployment

### Option 1: Railway (Recommended for Quick Deployment)

#### Deploy Backend

1. Create a Railway account at https://railway.app
2. Connect your GitHub repository
3. Select the WealthGuard AI repository
4. Configure environment variables in Railway dashboard:
   - `EMBEDDING_MODEL=all-MiniLM-L6-v2`
   - `VECTOR_COLLECTION=wealthguard_chunks`
   - `MAX_FILE_SIZE=5242880`
   - `ALLOWED_EXTENSIONS=.txt,.md`
   - `CHUNK_SIZE=100`
   - `CHUNK_OVERLAP=15`
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `uvicorn rag.main:app --host 0.0.0.0 --port $PORT`
7. Deploy

#### Deploy Frontend

1. Create a new Railway service for Streamlit
2. Configure environment variables:
   - `RAG_API_URL=https://your-backend-url.railway.app`
3. Set start command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy

### Option 2: Render

#### Deploy Backend

1. Create a Render account at https://render.com
2. Create a new Web Service
3. Connect your GitHub repository
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn rag.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render dashboard
6. Deploy

#### Deploy Frontend

1. Create a new Web Service for Streamlit
2. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
3. Add `RAG_API_URL` environment variable pointing to backend
4. Deploy

### Option 3: Docker Deployment

#### Build Docker Image

Create `Dockerfile` in the root directory:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p vector_db uploads

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
# Build image
docker build -t wealthguard-ai .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/vector_db:/app/vector_db \
  -v $(pwd)/uploads:/app/uploads \
  --env-file .env \
  wealthguard-ai
```

### Option 4: Streamlit Cloud (Frontend Only)

1. Create a Streamlit Cloud account at https://share.streamlit.io
2. Connect your GitHub repository
3. Select the WealthGuard AI repository
4. Configure environment variables in Streamlit Cloud dashboard
5. Deploy

Note: The backend must be deployed separately and the `RAG_API_URL` must point to it.

---

## Environment Variables Reference

### Required for Local Development

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_API_URL` | Backend API URL | `http://localhost:8000` |
| `EMBEDDING_MODEL` | Sentence Transformers model name | `all-MiniLM-L6-v2` |
| `VECTOR_DB_PATH` | ChromaDB storage path | `./vector_db/chroma_data` |
| `VECTOR_COLLECTION` | ChromaDB collection name | `wealthguard_chunks` |
| `MAX_FILE_SIZE` | Max upload size in bytes | `5242880` (5MB) |
| `ALLOWED_EXTENSIONS` | Allowed file extensions | `.txt,.md` |
| `CHUNK_SIZE` | Token chunk size | `100` |
| `CHUNK_OVERLAP` | Token chunk overlap | `15` |
| `FASTAPI_HOST` | FastAPI host | `0.0.0.0` |
| `FASTAPI_PORT` | FastAPI port | `8000` |
| `STREAMLIT_HOST` | Streamlit host | `localhost` |
| `STREAMLIT_PORT` | Streamlit port | `8501` |

### Optional (for Chat Completion)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_BASE_URL` | OpenAI API base URL |
| `OPENAI_MODEL` | OpenAI model name |

---

## Monitoring and Logging

### Backend Logs

The FastAPI backend logs to stdout/stderr. View logs by:

```bash
# If running with uvicorn
# Logs appear in the terminal where uvicorn is running
```

### Frontend Logs

Streamlit logs appear in the terminal where it's running and in the browser console.

### Vector Database

ChromaDB stores data in `vector_db/chroma_data`. This directory is gitignored.

### Uploaded Documents

Uploaded documents are stored in `uploads/`. This directory is gitignored.

---

## Troubleshooting

### Port Already in Use

If you get "Port already in use" error:

```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

### Module Not Found Errors

If you get "Module not found" errors:

```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### ChromaDB Errors

If you get ChromaDB-related errors:

```bash
# Delete and recreate the vector database
rm -rf vector_db/chroma_data
mkdir -p vector_db/chroma_data
```

### Frontend Can't Connect to Backend

1. Verify backend is running: `curl http://localhost:8000/`
2. Check `RAG_API_URL` in `.env` matches backend URL
3. Check for CORS errors in browser console
4. Ensure both are running on the same network or accessible

---

## Security Best Practices for Production

1. **Never commit `.env` files** with real credentials
2. **Use environment variables** for all sensitive configuration
3. **Enable HTTPS** for all endpoints
4. **Implement authentication** for API endpoints
5. **Add rate limiting** to prevent abuse
6. **Regularly update dependencies** for security patches
7. **Use managed services** for vector database in production
8. **Set up monitoring and alerts** for application health
9. **Implement backup strategy** for vector database
10. **Regular security audits** of the application

---

## Scaling Considerations

### Backend Scaling

- Use a load balancer (e.g., Nginx, AWS ALB)
- Deploy multiple instances of the FastAPI backend
- Use a managed vector database (ChromaDB Cloud, Pinecone, Qdrant)
- Implement caching for frequently accessed documents

### Frontend Scaling

- Streamlit Cloud handles scaling automatically
- For custom deployments, use a CDN for static assets
- Consider server-side rendering for better performance

### Database Scaling

- ChromaDB Cloud for managed scaling
- Pinecone or Qdrant for production-grade vector search
- Implement document versioning and expiration

---

## CI/CD Pipeline (Future)

Example GitHub Actions workflow:

```yaml
name: Deploy WealthGuard AI

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
      - name: Deploy to Railway
        run: railway up
```

---

## Support

For deployment issues or questions:
1. Check the troubleshooting section above
2. Review the main README.md
3. Check application logs for error details
4. Open an issue in the GitHub repository
