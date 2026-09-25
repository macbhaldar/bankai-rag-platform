"""Process-wide service container shared by the API, scripts and tests."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from app.core.audit import AuditLog
from app.core.config import Settings, get_settings, reset_settings
from app.core.logging import get_logger, setup_logging
from app.evaluation import evaluator  # noqa: F401  (imported for convenience by callers)
from app.generation.llm import resolve_llm
from app.generation.qa import RAGChain
from app.pipelines.ingestion import Ingester
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import get_embedding_provider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import maybe_reranker
from app.retrieval.vector_store import ChromaVectorStore
from app.security.acl import ACLService
from app.structured.engine import StructuredEngine

logger = get_logger(__name__)


@dataclass
class Services:
    settings: Settings
    embeddings: object
    store: ChromaVectorStore
    bm25: BM25Index
    retriever: HybridRetriever
    llm: object
    rag: RAGChain
    ingester: Ingester
    audit: AuditLog
    acl: ACLService
    structured_engine: StructuredEngine | None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.paths.logs_dir / "app.log")
    embeddings = get_embedding_provider(settings)
    store = ChromaVectorStore(settings.paths.index_dir / "chroma")
    bm25 = BM25Index(settings.paths.index_dir / "bm25_corpus.json")
    retriever = HybridRetriever(store, bm25, embeddings, settings, maybe_reranker(settings))
    llm = resolve_llm(settings)
    audit = AuditLog(settings.paths.audit_dir / "audit_log.jsonl", mask_pii=settings.security.mask_pii)
    ingester = Ingester(settings, embeddings, store, bm25)
    acl = ACLService(settings.paths.structured_dir)
    acl.set_domains(ingester.domains())
    structured_engine = (
        StructuredEngine(settings.paths.structured_dir, max_rows=settings.structured.max_rows)
        if settings.structured.enabled and settings.paths.structured_dir.exists()
        else None)
    rag = RAGChain(llm, retriever, settings, audit)
    logger.info(
        "services ready: embedding=%s llm=%s chunks=%d docs=%d",
        embeddings.name,
        f"{llm.provider}:{llm.model}",
        store.count(),
        len(ingester.documents()),
        )
    return Services(
        settings=settings,
        embeddings=embeddings,
        store=store,
        bm25=bm25,
        retriever=retriever,
        llm=llm,
        rag=rag,
        ingester=ingester,
        audit=audit,
        acl=acl,
        structured_engine=structured_engine,
        )

_cache: Services | None = None


def get_services() -> Services:
    global _cache
    if _cache is None:
        _cache = build_services()
    return _cache

def reset_services() -> None:
    global _cache
    _cache = None
    reset_settings()
    