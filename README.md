# RAG App Starter

## Setup Instructions
1. Create virtual environment: `python -m venv .venv`
2. Activate environment: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables: `cp .env.example .env` and fill in your keys.
5. Run the application.



### 1) Business / leadership view
The wealth division is sitting on a large set of approved knowledge:
- investment policies
- tax rules
- product brochures
- client communication guidelines

But relationship managers are answering from memory, past experience, or scattered files. That creates:
- inconsistent advice across teams
- slower response times to client questions
- higher compliance and reputational risk
- poor onboarding for new advisors
- difficulty proving that advice was based on current approved material

### 2) Relationship manager (RM) view
RMs need quick answers to client questions such as:
- Which investment product fits a tax profile?
- What are the latest policy changes?
- Which brochure is current and approved?
- What can we legally say to a client?

Without a trusted tool, they waste time searching documents and risk giving outdated or subjective advice.

### 3) Compliance / legal / risk view
This is a serious governance issue. If the bank gives advice without grounding it in current approved material, it could lead to:
- policy violations
- incorrect tax advice
- reputational damage
- higher audit burden
- regulatory scrutiny

So the system must ensure:
- only approved sources are used
- responses are traceable to source documents
- outdated materials are avoided
- every answer can be cited back to policy or brochure

### 4) Technology / AI view
The core challenge is knowledge retrieval and grounding. A general LLM will generate plausible answers, but without access to the current approved corpus it may hallucinate or produce outdated guidance. We need a system that:
- ingests internal documents
- structures them for retrieval
- finds the most relevant approved content
- grounds the LLM answer in those documents
- shows citations and source references
- logs every interaction for governance

---

## How we can solve it

The best solution is a RAG-based internal knowledge assistant.

### Proposed solution
Build a secure AI assistant for the wealth division that:
1. Stores all approved documents in a central repository
2. Extracts and cleans text from PDFs, Word files, policies, and brochures
3. Splits documents into meaningful chunks
4. Converts chunks into embeddings
5. Stores them in a vector database
6. Retrieves the most relevant chunks for a user question
7. Sends those chunks to an LLM with prompt instructions
8. Generates an answer grounded in approved material
9. Returns citations to the exact source documents
10. Enforces access control, logging, and approval workflows

### Key features of the solution
- Search by natural language questions
- Answer with source-backed citations
- Filter by product, policy version, date, or document type
- Block unsupported or non-approved content
- Human review for sensitive answers
- Audit trail for all AI-generated responses

### Example workflow
A relationship manager asks:
“Which investment products are suitable for a high-income client in the 2025 tax bracket?”

The system:
- searches approved policy docs and product brochures
- retrieves the relevant rules and product details
- passes them to the LLM
- answers with citations like:
  - Policy Handbook, Section 4.2
  - Tax Rule Update v3.1
  - Product Brochure Q1 2025

This makes the answer usable, explainable, and compliant.

---

## Team Members (3)

### 1) Nikunj
Role: Product lead / business owner
Responsibilities:
- define user stories and business requirements
- align the project with wealth division goals
- coordinate with compliance and relationship managers
- prioritize features and ensure value delivery

### 2) Nishant
Role: AI / ML and RAG engineer
Responsibilities:
- document ingestion pipeline
- chunking and embedding strategy
- vector database design
- retrieval optimization
- LLM prompt design
- grounding and evaluation of answer quality

### 3) Subhadeep
Role: Full-stack developer / backend + UX
Responsibilities:
- build the application backend and APIs
- develop chat interface or admin portal
- connect frontend to RAG service
- implement auth, logging, and deployment
- support integration with storage and retrieval components

---

## Tech Stack and responsibility mapping

| Layer | Tech / Tool | Responsibility |
|---|---|---|
| Frontend | React / Next.js / Streamlit / simple web UI | Chat interface, query input, response display, citations, user experience |
| Backend API | Python FastAPI / Node.js Express | Receive user queries, orchestrate retrieval + generation, expose endpoints |
| LLM | OpenAI / Azure OpenAI / other LLM provider | Generate grounded answers based on retrieved context |
| Retrieval | Vector DB (Pinecone, Weaviate, Qdrant, pgvector) | Store embeddings and return the most relevant document chunks |
| Document Processing | Python, pdfplumber, LangChain, Unstructured, OCR tools | Load PDFs, clean text, extract content, prepare data |
| Embeddings | OpenAI Embeddings / Azure Embeddings | Convert text chunks to vector representations |
| Prompting | LLM prompt templates | instruct model to answer using only approved context |
| Data Storage | PostgreSQL / cloud object storage / file system | Store raw documents, metadata, and source references |
| Search Layer | Hybrid search + metadata filters | Combine semantic search with keyword filtering for precision |
| Security + Access Control | Auth system, RBAC, encrypted storage | Ensure only authorized users can access sensitive document sets |
| Monitoring | Logging, evaluation metrics, traces | Track usage, performance, failures, and answer quality |
| Deployment | Docker, cloud hosting, CI/CD | Run and maintain the application in production |

---

## What each part of the tech stack does

### Frontend
Responsible for:
- user-friendly chat UI
- showing answers and citations
- feedback buttons like “helpful / not helpful”
- search history and filters for product or policy domain

### Backend API
Responsible for:
- handling requests from the frontend
- calling the retrieval engine
- passing context to the LLM
- returning final answer with citations and metadata
- enforcing validation and access rules

### Retrieval engine
Responsible for:
- searching the approved corpus
- finding top relevant chunks
- combining semantic matching + metadata filters
- improving answer relevance and precision

### LLM layer
Responsible for:
- interpreting user intent
- using the retrieved context to craft safe, grounded responses
- handling refusal or uncertainty when data is insufficient
- generating concise but complete advice

### Document ingestion pipeline
Responsible for:
- converting files into clean text
- splitting into chunks
- attaching metadata such as source, product type, policy version, date
- detecting outdated or duplicate documents

### Vector database
Responsible for:
- storing embeddings efficiently
- enabling similar-item retrieval
- supporting top-k document lookup for a query

### Governance layer
Responsible for:
- controlling what users can access
- logging user queries and outputs
- storing version history of policies
- maintaining auditability for compliance review

---

## Sprint #2 focus: AI Application Development with RAG

Since your sprint list is centered on RAG, the project should follow this journey:

### Sprint 2 objective
Build an AI application that can answer questions from internal bank documents using retrieval-augmented generation.

### Core sprint flow
1. LLM application foundation
   - Understand model behavior, prompts, and architecture

2. Document processing
   - Prepare approved policy and brochure files

3. Chunking
   - Break large documents into useful retrieval units

4. Embeddings
   - Convert chunks into vector representations

5. Vector DB and retrieval
   - Store vectors and run relevant retrieval

6. RAG pipeline
   - Combine retrieval + generation with source-grounding

7. Integration
   - Connect frontend and backend APIs

8. Evaluation
   - Measure answer relevance, quality, and hallucination risk

9. Deployment
   - Ship a working pilot to the wealth division

---

## Final recommendation

The right solution is not a standalone general chatbot. It is a domain-specific, grounded knowledge assistant for the wealth division.

What makes it valuable:
- answers are based on approved internal material
- outputs are cited and auditable
- RMs get consistent support
- compliance risk is reduced
- the system scales across policy, tax, and product data

If you want, I can turn this into a polished one-page project brief or a presentation-ready slide format with:
- problem statement
- business case
- solution overview
- team roles
- architecture diagram
- sprint plan
- tech stack summary