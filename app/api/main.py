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
    StructuredRequest,
)
from app.core.config import PROJECT_ROOT, Settings, get_settings
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
                "reranker": settings.retrieval.use_reranker,
            },
        }

    @app.get("/api/meta/filters", tags=["ops"])
    def meta_filters():
        services: Services = get_services()
        documents = services.ingester.documents()
        domains = sorted({d.get("domain", "general") for d in documents})
        jurisdictions = sorted({d["jurisdiction"] for d in documents if d.get("jurisdiction")})
        families = sorted({d["family"] for d in documents})
        return {"domains": domains, "jurisdictions": jurisdictions, "families": families}

    # search
    @app.post("/api/search", tags=["retrieval"])
    def search(request: SearchRequest):
        services: Services = get_services()
        started = time.perf_counter()
        allowed = services.acl.allowed_document_ids(request.principal)
        hits = services.retriever.retrieve(
            request.query,
            k=request.k,
            filters={
                "category": request.category,
                "jurisdiction": request.jurisdiction,
                "family": request.family,
                "document_id": request.document_id,
            },
            allowed_document_ids=allowed,
            dedupe_families=request.dedupe_families,
        )
        latency = round((time.perf_counter() - started) * 1000, 1)
        services.audit.append(
            "search", query=request.query, principal=request.principal,
            results=len(hits), latency_ms=latency,
        )
        return {
            "query": request.query,
            "k": request.k or settings.retrieval.final_k,
            "scoped_to_principal": request.principal,
            "latency_ms": latency,
            "results": [h.to_dict() for h in hits],
        }

    # rag qa
    @app.post("/api/qa", tags=["rag"])
    def ask(request: QARequest) -> dict:
        services: Services = get_services()
        allowed = services.acl.allowed_document_ids(request.principal)
        result: QAResult = services.rag.ask(
            request.question,
            k=request.k,
            filters={"category": request.category, "jurisdiction": request.jurisdiction},
            history=request.history,
            allowed_document_ids=allowed,
            dedupe_families=request.dedupe_families,
        )
        return result.to_dict()

    # ingest
    @app.post("/api/ingest", tags=["ingestion"], dependencies=[Depends(guard)])
    def ingest(request: IngestRequest):
        services: Services = get_services()
        source = Path(request.source_dir) if request.source_dir else None
        if source is not None:
            resolved = source if source.is_absolute() else (Path.cwd() / source)
            resolved = resolved.resolve()
            if not (resolved.is_dir() and str(resolved).startswith(str(PROJECT_ROOT))):
                raise HTTPException(status_code=400, detail=f"source_dir must be a directory inside {PROJECT_ROOT}")
        try:
            report: IngestReport = services.ingester.run(
                source_dir=source, rebuild=request.rebuild, dry_run=request.dry_run, limit=request.limit,
            )
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        services.acl.set_domains(services.ingester.domains())
        return report.to_dict()

    @app.post("/api/ingest/upload", tags=["ingestion"], dependencies=[Depends(guard)])
    async def ingest_upload(file: UploadFile = File(...)):
        services: Services = get_services()
        suffix = Path(file.filename or "upload.txt").suffix.lower() or ".txt"
        if suffix not in {".md", ".txt", ".pdf", ".docx", ".csv", ".json"}:
            raise HTTPException(status_code=400, detail=f"unsupported file type: {suffix}")
        target_dir = settings.paths.uploads_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{int(time.time())}_{Path(file.filename).name}"
        target.write_bytes(await file.read())
        try:
            report: IngestReport = services.ingester.run(source_dir=target)
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        services.acl.set_domains(services.ingester.domains())
        return report.to_dict()

    @app.get("/api/documents", tags=["ingestion"])
    def documents(
        limit: int = Query(default=50, ge=1, le=2500),
        offset: int = Query(default=0, ge=0),
        domain: str | None = None,
        principal: str | None = None,
    ):
        services: Services = get_services()
        allowed = services.acl.allowed_document_ids(principal)
        rows = services.ingester.documents()
        rows = [r for r in rows if (allowed is None or r["document_id"] in allowed)]
        if domain:
            rows = [r for r in rows if r.get("domain") == domain]
        total = len(rows)
        return {"total": total, "offset": offset, "documents": rows[offset : offset + limit]}

    @app.delete("/api/documents/{document_id}", tags=["ingestion"], dependencies=[Depends(guard)])
    def delete_document(document_id: str):
        services: Services = get_services()
        deleted = services.ingester.delete_document(document_id)
        services.acl.set_domains(services.ingester.domains())
        if not deleted:
            raise HTTPException(status_code=404, detail="document not found in registry")
        services.audit.append("delete_document", document_id=document_id)
        return DeleteDocumentResponse(deleted=True, document_id=document_id)

    # eval
    @app.post("/api/eval", tags=["evaluation"])
    def eval_endpoint(request: EvalRequest):
        services: Services = get_services()
        try:
            report = run_benchmark(services, benchmark=request.benchmark, k=request.k, limit=request.limit)
        except EvalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        path = save_report(report, settings.paths.eval_dir)
        payload = report.to_dict()
        payload["report_path"] = str(path)
        return payload

    @app.get("/api/eval/latest", tags=["evaluation"])
    def eval_latest():
        eval_dir = settings.paths.eval_dir
        if not eval_dir.exists():
            return {"report": None}
        reports = sorted(eval_dir.glob("eval_*.json"))
        if not reports:
            return {"report": None}
        import json

        return {"report": json.loads(reports[-1].read_text(encoding="utf-8"))}

    # structured
    @app.post("/api/structured", tags=["structured"])
    def structured(request: StructuredRequest):
        services: Services = get_services()
        if services.structured_engine is None:
            raise HTTPException(status_code=503, detail="structured data is disabled or unavailable")
        payload = answer_structured(services.structured_engine, services.llm, request.question)
        services.audit.append("structured_query", question=request.question, mode=payload.get("mode"))
        return payload

    @app.get("/api/structured/schema", tags=["structured"])
    def structured_schema():
        services: Services = get_services()
        if services.structured_engine is None:
            raise HTTPException(status_code=503, detail="structured data is disabled or unavailable")
        return {"tables": services.structured_engine.schema()}

    # security & audit
    @app.get("/api/security/users", tags=["security"])
    def security_users(limit: int = Query(default=100, ge=1, le=500)):
        services: Services = get_services()
        return {"users": services.acl.list_users(limit=limit), "stats": services.acl.stats()}

    @app.get("/api/audit", tags=["security"])
    def audit(limit: int = Query(default=50, ge=1, le=500)):
        services: Services = get_services()
        return {"events": services.audit.tail(limit=limit)}

    return app


app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.log_level.lower(),
    )

if __name__ == "__main__":
    main()
