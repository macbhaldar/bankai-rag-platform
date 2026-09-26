"""FastAPI application exposing search, RAG QA, ingestion, evaluation, structured-data and audit endpoints."""

from __future__ import annotations
import time
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from app.api.schemas import (
    DeleteDocumentResponse,
    EvalRequest,
    IngestRequest,
    QARequest,
    SearchRequest,
    StructuredRequest,)
from app.core.config import Settings, get_settings
from app.core.container import Services, get_services
from app.core.logging import get_logger
from app.evaluation.evaluator import EvalError, run_benchmark, save_report
from app.generation.qa import QAResult
from app.pipelines.ingestion import IngestError, IngestReport
from app.structured.engine import SQLRejected
from app.structured.text2sql import answer_structured

logger = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _require_api_key(key: str | None = Depends(_api_key_header)) -> None:
    required = get_settings().api.key
    if required and key != required:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing X-API-Key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    services = get_services()
    logger.info("API starting: %d chunks indexed", services.store.count())
    yield

def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="BankRAG API",
        description="Bank Intelligence RAG Platform — semantic search, document QA, RAG and structured data answers.",
        version="0.1.0",
        lifespan=lifespan,
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        )
    guard = _require_api_key

    # health
    @app.get("/health", tags=["ops"])
    def health():
        services: Services = get_services()
        return {
            "status": "ok",
            "chunks": services.store.count(),
            "documents": len(services.ingester.documents()),
            "bm25_entries": len(services.bm25.entries),
            "embedding_provider": services.embeddings.name,
            "llm_provider": services.llm.provider,
            "llm_model": services.llm.model,
            "llm_mode": services.llm.mode,
            "acl_loaded": services.acl.loaded,
            }

    @app.get("/api/stats", tags=["ops"])
    def stats(principal: str | None = None):
        services: Services = get_services()
        allowed = services.acl.allowed_document_ids(principal)
        documents = services.ingester.documents()
        visible = [d for d in documents if allowed is None or d["document_id"] in allowed] if principal else documents
        domains: dict[str, int] = {}
        for doc in visible:
            domains[doc.get("domain", "general")] = domains.get(doc.get("domain", "general"), 0) + 1
        return {
            "documents": len(visible),
            "chunks": services.store.count(),
            "bm25_entries": len(services.bm25.entries),
            "domains": domains,
            "embedding_provider": services.embeddings.name,
            "llm": {"provider": services.llm.provider, "model": services.llm.model, "mode": services.llm.mode},
            "acl": services.acl.stats(),
            "structured_tables": len(services.structured_engine.schema()) if services.structured_engine else 0,
            "retrieval": {
                "dense_k": settings.retrieval.dense_k,
                "bm25_k": settings.retrieval.bm25_k,
                "final_k": settings.retrieval.final_k,
                "multi_query": settings.retrieval.multi_query,
                "reranker": settings.retrieval.use_reranker,},
            }

    @app.get("/api/meta/filters", tags=["ops"])
    def meta_filters():
        services: Services = get_services()
        documents = services.ingester.documents()
        domains = sorted({d.get("domain", "general") for d in documents})
        jurisdictions = sorted({d["jurisdiction"] for d in documents if d.get("jurisdiction")})
        families = sorted({d["family"] for d in documents})
        return {"domains": domains, "jurisdictions": jurisdictions, "families": families}
    