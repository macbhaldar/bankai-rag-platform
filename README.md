# BankAI RAG Platform
Bank Intelligence RAG Platform: Generative AI information-access platform for a bank.

[![Status](https://img.shields.io/badge/Status-Active%20Development-orange.svg)](#)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Banking%20%7C%20FinTech-blue.svg)](#)
[![AI](https://img.shields.io/badge/AI-Generative%20AI-purple.svg)](#)
[![RAG](https://img.shields.io/badge/Architecture-RAG-green.svg)](#)

------------------------------------------------------------------------

## Overview

**BankAI RAG Platform** is a research project for
building a banking focused Retrieval-Augmented Generation system.

The platform is designed around a simple problem:

> Banking organizations have large volumes of policies, manuals, financial reports, regulatory material, product documentation, research, and operational knowledge. Finding the correct information quickly and generating an answer that remains grounded in the source material is difficult.

BankAI explores how modern **LLMs, semantic search, document retrieval,
embeddings, and data-engineering pipelines** can be combined to create a
domain-specific information-access layer for banking.

------------------------------------------------------------------------

## Project Goals

-   Build a reusable banking-domain RAG architecture.
-   Ingest and organize financial and banking knowledge.
-   Convert documents into searchable representations.
-   Retrieve relevant information for a user query.
-   Provide LLM-generated answers grounded in retrieved context.
-   Preserve document/source metadata for traceability.
-   Establish an evaluation framework for retrieval and generation
    quality.
-   Create a foundation for enterprise-grade banking AI.
-   Explore security, governance, auditability, and model-risk controls.

------------------------------------------------------------------------

## Why RAG for Banking?

A general-purpose LLM does not automatically have access to an
institution's current internal knowledge.

RAG provides a mechanism to connect an LLM with an external knowledge
base:

``` text
                    BANKING KNOWLEDGE
                           |
        +------------------+------------------+
        |                  |                  |
     Policies          Reports           Regulations
     Manuals           Research          Procedures
     Products          Statements        FAQs
        |                  |                  |
        +------------------+------------------+
                           |
                  DOCUMENT INGESTION
                           |
                  CLEANING / PARSING
                           |
                       CHUNKING
                           |
                      EMBEDDINGS
                           |
                    VECTOR / SEARCH DB
                           |
                           |
USER QUESTION  ----------->|
                           |
                      RETRIEVAL
                           |
                    RELEVANT CONTEXT
                           |
                          LLM
                           |
                   GROUNDED RESPONSE
                           |
                   SOURCES / CITATIONS
```

The objective is not simply to generate fluent text. The objective is to
make the generated response **useful, traceable, and grounded in
retrieved evidence**.


## Quick start

### Windows 10/11

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python.exe scripts\run_ingest.py --rebuild
.venv\Scripts\python.exe scripts\run_api.py
```

In another terminal:

```bat
.venv\Scripts\python.exe -m streamlit run app\ui\streamlit_app.py --server.port 8501
```

Open the API documentation at `http://127.0.0.1:8000/docs` and the Streamlit UI at `http://127.0.0.1:8501`.

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python scripts/run_ingest.py --rebuild
./scripts/dev.sh
```

### Offline mode

The platform can run without an external LLM by setting:

```text
BANKRAG_LLM_PROVIDER=extractive
BANKRAG_EMBEDDING_PROVIDER=hash
```

The test suite uses these deterministic providers.

### Optional local generation

For Ollama, set `BANKRAG_LLM_PROVIDER=ollama`, `BANKRAG_OLLAMA_URL`, and `BANKRAG_LLM_MODEL`.

For an OpenAI-compatible endpoint, set `BANKRAG_LLM_PROVIDER=openai`, `BANKRAG_LLM_BASE_URL`, `BANKRAG_LLM_API_KEY`, and `BANKRAG_LLM_MODEL`.

### API security

Set `BANKRAG_API_KEY` to protect all `/api/*` routes. `/health` remains public for monitoring. The Streamlit client automatically forwards the configured key.

### Evaluation

```bash
python scripts/run_eval.py --benchmark retrieval
python scripts/run_eval.py --benchmark qa_test
python scripts/run_eval.py --benchmark hard_negatives
```

Reports are written to `data/eval_results/`.
