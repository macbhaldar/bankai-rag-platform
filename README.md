# BankAI RAG Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

> **BankAI RAG Platform** is a modular banking intelligence platform that combines Retrieval-Augmented Generation (RAG), hybrid search, structured-data analytics, access-controlled retrieval, PII protection, evaluation, and auditability in a single Python application.

---

## Overview

BankAI RAG Platform is designed as a practical reference architecture for building trustworthy information-access systems for banking and financial-services environments.

The platform supports two complementary information paths:

1. **Unstructured intelligence** — retrieve relevant banking policies, procedures, product documents, compliance material, and other indexed documents, then answer questions with source-aware context.
2. **Structured intelligence** — answer analytical questions against controlled banking datasets using a read-only natural-language-to-SQL workflow.

The architecture is intentionally modular so that retrieval, embeddings, LLM providers, security policies, evaluation and the user interface can be developed independently.

### Core pipeline

```text
                         ┌──────────────────────┐
                         │   Streamlit UI       │
                         │ Search / RAG / Admin  │
                         └──────────┬───────────┘
                                    │ HTTP
                         ┌──────────▼───────────┐
                         │      FastAPI         │
                         │ REST + OpenAPI       │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
       ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
       │ Hybrid      │      │ RAG / LLM   │      │ Structured   │
       │ Retrieval   │      │ Generation  │      │ Analytics    │
       └──────┬──────┘      └─────────────┘      └──────┬──────┘
              │                                         │
       ┌──────▼─────────────────┐              ┌────────▼────────┐
       │ Dense + BM25 + RRF     │              │ Read-only SQL   │
       │ + optional reranking   │              │ execution       │
       └──────┬─────────────────┘              └────────┬────────┘
              │                                         │
       ┌──────▼───────┐                         ┌───────▼───────┐
       │ Vector Store │                         │ Banking Data  │
       │ + Embeddings │                         │ SQLite/CSV    │
       └──────────────┘                         └───────────────┘

                    Cross-cutting controls
          ┌─────────────────────────────────────────────┐
          │ ACL • PII Masking • Audit • Evaluation      │
          └─────────────────────────────────────────────┘
```

---

## Key Features

### Retrieval-Augmented Generation

- Document ingestion and preprocessing
- Markdown, TXT, PDF, DOCX, CSV and JSON support
- Configurable chunking and overlap
- Dense vector retrieval
- BM25 lexical retrieval
- Hybrid retrieval with Reciprocal Rank Fusion
- Optional reranking
- Metadata filtering
- Document-family deduplication
- Principal-aware retrieval
- Multi-query retrieval configuration
- Source/citation-aware RAG responses

### LLM Providers

The generation layer supports:

| Provider | Use |
|---|---|
| OpenAI-compatible | OpenAI and compatible gateways/local servers |
| Ollama | Local LLM inference |
| Extractive | Offline deterministic fallback |

Provider selection is configuration-driven.

```text
BANKRAG_LLM_PROVIDER=auto
```

With `auto`, the platform can select an available configured provider and fall back to the extractive implementation when external generation is unavailable.

### Banking Intelligence

The project includes banking-oriented sample data and document workflows covering areas such as:

- Customers
- Accounts
- Transactions
- Loans
- Securities
- Users and roles
- Banking policies and procedures
- Compliance-oriented documents

### Security

Security is treated as part of the retrieval architecture rather than only an API feature.

- Permission-aware document retrieval
- Principal-based access control
- Department/domain restrictions
- Document-level grants
- Break-glass access model
- PII masking
- Audit events
- Configurable API-key authentication
- Read-only structured SQL execution
- SQL safety validation
- Row limits for structured queries

When `BANKRAG_API_KEY` is configured, all `/api/*` endpoints require the `X-API-Key` header. The health endpoint remains public for monitoring.

### Structured Analytics

The structured-query subsystem allows users to ask questions about banking datasets in natural language.

Example:

```text
Which branch has the highest total loan exposure?
```

The workflow is:

```text
Natural-language question
        ↓
Question analysis
        ↓
Safe SQL generation / template resolution
        ↓
SQL validation
        ↓
Read-only execution
        ↓
Row-limited result
        ↓
Human-readable answer
```

### Evaluation

The platform includes evaluation infrastructure for measuring retrieval and QA quality.

Supported evaluation areas include:

- Recall@K
- MRR
- nDCG
- Document hit rate
- Section hit rate
- Family-level retrieval
- Hard-negative robustness
- QA benchmark evaluation
- JSON evaluation reports

---

## Architecture

```text
bankai-rag-platform/
│
├── app/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── core/
│   │   ├── audit.py
│   │   ├── config.py
│   │   ├── container.py
│   │   ├── logging.py
│   │   └── pii.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   ├── generation/
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   └── qa.py
│   │
│   ├── pipelines/
│   │   ├── chunking.py
│   │   ├── ingestion.py
│   │   └── loaders.py
│   │
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── embeddings.py
│   │   ├── hybrid.py
│   │   ├── reranker.py
│   │   └── vector_store.py
│   │
│   ├── security/
│   │   └── acl.py
│   │
│   ├── structured/
│   │   ├── engine.py
│   │   └── text2sql.py
│   │
│   └── ui/
│       ├── api_client.py
│       ├── streamlit_app.py
│       └── app_pages/
│
├── config/
│   └── settings.yaml
│
├── data/
│   ├── eval_results/
│   ├── processed/
│   ├── raw/
│   └── uploads/
│
├── dataset/
│   ├── qa/
│   ├── retrieval/
│   ├── structured/
│   └── raw_documents/
│
├── scripts/
│   ├── demo_queries.py
│   ├── run_api.py
│   ├── run_eval.py
│   ├── run_ingest.py
│   ├── dev.bat
│   └── dev.sh
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_chunking.py
│   ├── test_llm.py
│   ├── test_pii.py
│   ├── test_qa.py
│   ├── test_retrieval.py
│   └── test_structured.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .env.example
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Requirements

- Python **3.11 or newer**
- Git
- 4 GB+ RAM recommended for development
- Optional:
  - Ollama for local LLM inference
  - OpenAI-compatible API for hosted generation
  - GPU for optional embedding/reranking models

The platform can operate in a deterministic offline configuration without an external LLM.

---

## Installation

### Windows

```bat
git clone https://github.com/macbhaldar/bankai-rag-platform.git
cd bankai-rag-platform

py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

copy .env.example .env
```

### Linux / macOS

```bash
git clone https://github.com/macbhaldar/bankai-rag-platform.git
cd bankai-rag-platform

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

---

## Configuration

Configuration is managed through `config/settings.yaml` and environment variables.

Important variables include:

| Variable | Purpose |
|---|---|
| `BANKRAG_API_KEY` | API authentication |
| `BANKRAG_LLM_PROVIDER` | `auto`, `openai`, `ollama`, or `extractive` |
| `BANKRAG_LLM_MODEL` | LLM model name |
| `BANKRAG_LLM_API_KEY` | OpenAI-compatible API key |
| `BANKRAG_LLM_BASE_URL` | OpenAI-compatible endpoint |
| `BANKRAG_OLLAMA_URL` | Ollama server URL |
| `BANKRAG_EMBEDDING_PROVIDER` | Embedding backend |
| `BANKRAG_EMBEDDING_MODEL` | Embedding model |
| `BANKRAG_LOG_LEVEL` | Application log level |

Never commit production API keys or credentials to Git.

---

## Offline Development

For a completely local development path:

```text
BANKRAG_LLM_PROVIDER=extractive
BANKRAG_EMBEDDING_PROVIDER=hash
```

This mode is useful for:

- CI
- unit tests
- development
- API integration testing
- environments without external model access

The extractive provider returns relevant retrieved passages instead of generating text with an external model.

---

## Ingestion

Rebuild the local index:

```bash
python scripts/run_ingest.py --rebuild
```

Depending on the script options available in the installed version, ingestion can also be performed with a selected source directory, a document limit, or dry-run mode.

The ingestion pipeline performs:

```text
Documents
   ↓
Loader
   ↓
Normalization
   ↓
Metadata extraction
   ↓
Chunking
   ↓
Embedding
   ↓
Vector store
   ↓
BM25 index
   ↓
Document registry
```

---

## Running the API

### Windows

```bat
.venv\Scripts\python.exe scripts\run_api.py
```

### Linux/macOS

```bash
python scripts/run_api.py
```

The default API is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI specification:

```text
http://127.0.0.1:8000/openapi.json
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

## Running the Streamlit UI

Start the API first, then:

### Windows

```bat
.venv\Scripts\python.exe -m streamlit run app\ui\streamlit_app.py --server.port 8501
```

### Linux/macOS

```bash
python -m streamlit run app/ui/streamlit_app.py --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

The UI provides access to the platform's search, RAG, document, structured-data, evaluation, security and audit functionality.

---

## API Examples

### Search

```bash
curl -X POST http://127.0.0.1:8000/api/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the loan approval policy?\",\"k\":5}"
```

On a protected deployment:

```bash
curl -X POST http://127.0.0.1:8000/api/search ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: YOUR_API_KEY" ^
  -d "{\"query\":\"What is the loan approval policy?\",\"k\":5}"
```

### RAG question answering

```json
{
  "question": "What documents are required for loan approval?",
  "k": 5,
  "category": "lending"
}
```

POST to:

```text
/api/qa
```

### Structured query

```json
{
  "question": "Which branch has the highest total loan exposure?"
}
```

POST to:

```text
/api/structured
```

### Statistics

```text
GET /api/stats
```

### Documents

```text
GET /api/documents
```

### Evaluation

```json
{
  "benchmark": "retrieval",
  "k": 10
}
```

POST to:

```text
/api/eval
```

---

## Evaluation

Run the available benchmark workflows with:

```bash
python scripts/run_eval.py --benchmark retrieval
python scripts/run_eval.py --benchmark qa_test
python scripts/run_eval.py --benchmark hard_negatives
```

Evaluation reports are stored under:

```text
data/eval_results/
```

The evaluation framework is intended to make retrieval and QA changes measurable rather than relying only on subjective inspection.

---

## Testing

Run the complete test suite:

```bash
pytest -q
```

Run a specific test module:

```bash
pytest -q tests/test_retrieval.py
pytest -q tests/test_qa.py
pytest -q tests/test_structured.py
pytest -q tests/test_llm.py
```

CI automatically tests supported Python versions on pushes and pull requests.

---

## Security Model

The platform uses several layers of controls.

### Retrieval ACL

A user's principal is resolved against the access-control layer before retrieval results are returned.

```text
User / Principal
      ↓
ACL resolution
      ↓
Allowed document IDs
      ↓
Hybrid retrieval
      ↓
Filtered results
      ↓
RAG
```

This prevents unauthorized documents from entering the generation context.

### PII protection

PII masking can be enabled through configuration so sensitive fields are removed or masked before being surfaced through supported application paths.

### Audit

Security-sensitive and retrieval-related actions can be recorded in the audit subsystem.

The API exposes recent events through:

```text
GET /api/audit
```

### API authentication

Set:

```text
BANKRAG_API_KEY=change-me
```

Requests to protected endpoints must then include:

```text
X-API-Key: change-me
```

---

## Development Workflow

A recommended development cycle is:

```text
1. Modify module
       ↓
2. Add/update tests
       ↓
3. Run pytest
       ↓
4. Run ingestion if retrieval behavior changed
       ↓
5. Run evaluation benchmark
       ↓
6. Inspect audit/security behavior
       ↓
7. Commit
       ↓
8. Push
       ↓
9. GitHub Actions CI
```

For retrieval changes, evaluate both ordinary queries and hard negatives.

For security changes, verify that:

- authorized principals receive permitted documents;
- unauthorized principals receive no restricted documents;
- PII masking remains enabled where required;
- audit events are generated;
- API authentication works when configured.

---

## Design Principles

BankAI RAG Platform follows these principles:

### 1. Retrieval before generation

The LLM should receive evidence retrieved from the indexed knowledge base rather than being treated as the source of truth.

### 2. Security before generation

Authorization is applied before retrieved content reaches the generation layer.

### 3. Structured and unstructured data are separate paths

SQL analytics and document RAG have different failure modes and security requirements, so they are implemented as distinct workflows.

### 4. Provider independence

The application is not tightly coupled to one LLM vendor.

### 5. Offline fallback

The platform remains testable without a paid external LLM service.

### 6. Evaluation-driven development

Retrieval and QA changes should be measurable through repeatable benchmarks.

### 7. Auditability

Important application and security actions should be observable after execution.

---

## Limitations

This project is a development/reference platform and should not be treated as production banking infrastructure without additional engineering and security review.

Before production use, consider adding:

- enterprise identity and SSO integration;
- mTLS and network segmentation;
- secrets management;
- database encryption;
- production-grade persistent databases;
- distributed vector storage;
- queue-based ingestion;
- rate limiting;
- request tracing;
- centralized audit storage;
- model governance;
- prompt-injection defenses;
- stronger output validation;
- automated security scanning;
- penetration testing;
- disaster recovery;
- backup and retention policies;
- formal compliance controls.

---

## Roadmap

Potential future development includes:

- [ ] Enterprise SSO / OAuth2 / OIDC
- [ ] Fine-grained role and attribute-based access control
- [ ] Production PostgreSQL integration
- [ ] Distributed vector database support
- [ ] Advanced reranking models
- [ ] Query rewriting and decomposition
- [ ] Conversation memory with policy controls
- [ ] Prompt-injection detection
- [ ] Hallucination/grounding classifiers
- [ ] OpenTelemetry tracing
- [ ] Prometheus/Grafana monitoring
- [ ] Background ingestion workers
- [ ] Document versioning
- [ ] Approval workflows
- [ ] Model benchmarking dashboard
- [ ] Docker / Docker Compose deployment
- [ ] Kubernetes deployment manifests

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make the change.
4. Add or update tests.
5. Run:

```bash
pytest -q
```

6. Update documentation when behavior changes.
7. Submit a pull request.

Keep changes modular and avoid placing provider-specific logic directly inside API endpoints.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE).

---

## Author

**Maksud Bhaldar**

GitHub: [@macbhaldar](https://github.com/macbhaldar)

Project: [BankAI RAG Platform](https://github.com/macbhaldar/bankai-rag-platform)

---

## Disclaimer

This repository is intended for research, education, prototyping, and software-engineering demonstration. It does not constitute financial, banking, legal, compliance, investment, or security advice, and it should not be deployed in a regulated production environment without appropriate professional review.
