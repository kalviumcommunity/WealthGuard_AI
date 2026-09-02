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

- **Retrieval-Augmented Generation** — Grounds responses in approved organizational documents
- **Source Citations** — References original sources for verification
- **Document Filtering** — Prioritizes valid and approved versions using metadata
- **Conflict Detection** — Identifies contradictions across documents
- **Human-in-the-Loop** — Supports decision-making without replacing professional judgment

---

## Technology Stack

**Frontend:** React.js, Tailwind CSS

**Backend:** Python, FastAPI

**AI & RAG:** Large Language Model (LLM), Embedding Model, LangChain / LlamaIndex

### Data & Storage

* **PostgreSQL** — Relational database for structured application data
* **Vector Database** — Stores document embeddings and enables semantic similarity search
* **Qdrant / Chroma / pgvector** — Possible vector storage solutions depending on deployment requirements

### Document Processing

* **PyMuPDF** — Python library for extracting and processing content from PDF documents
* **PDF/DOCX Parsers** — Used to extract text and metadata from organizational documents

### Deployment

* **Docker** — Containerization platform for consistent development and deployment environments

---

## Use Cases

### Tax and Regulatory Queries

Retrieve the latest applicable tax rules and regulations for investment products.

### Product Information

Provide product-specific information such as features, eligibility requirements, investment limits, and applicable conditions.

### Policy Queries

Answer questions using the latest approved investment and compliance policies.

### Document Verification

Allow relationship managers to review the source material supporting an AI-generated response.

---

## Responsible AI

WealthGuard AI follows an **evidence-first approach** because it is intended for use in a financial environment.

The system is designed to:

* Prioritize approved organizational documentation.
* Consider document versions and validity periods.
* Provide supporting sources for generated responses.
* Flag potential conflicts or insufficient information.
* Avoid unsupported financial recommendations.
* Keep final customer-specific decisions with qualified human professionals.

---

## Project Goals

1. Improve consistency in information provided by relationship managers.
2. Reduce reliance on outdated or incorrect documentation.
3. Reduce the time required to locate relevant information.
4. Provide transparent and source-backed AI responses.
5. Support compliance-oriented information retrieval.
6. Assist relationship managers while preserving human decision-making.

---

## Future Enhancements

* Multilingual support
* Voice-based interaction
* Automated document version comparison
* Document expiry notifications
* Advanced compliance workflows
* Improved retrieval and reranking
* Audit and usage analytics
* Integration with internal banking systems

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