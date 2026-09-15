"""Ingestion orchestrator: incremental, observable, with per-stage timings."""
from __future__ import annotations
 
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
 
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.pii import scan_text
from app.pipelines.chunking import DocMeta, chunk_document, extract_metadata
from app.pipelines.loaders import discover_files, load_file
from app.retrieval.bm25 import BM25Index, Bm25Entry
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import ChromaVectorStore
 
logger = get_logger(__name__)
 
 
class IngestError(RuntimeError):
    pass
 
 
@dataclass
class IngestReport:
    run_id: str
    started_at: str
    finished_at: str = ""
    source: str = ""
    rebuild: bool = False
    dry_run: bool = False
    docs_seen: int = 0
    docs_ingested: int = 0
    docs_skipped: int = 0
    chunks_indexed: int = 0
    embedding_provider: str = ""
    embedding_dim: int = 0
    stage_ms: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
 
    def to_dict(self) -> dict:
        return asdict(self)
 
 
class Ingester:
    """Runs the load -> enrich -> chunk -> embed -> index pipeline.
 
    Documents already present in the registry with an unchanged SHA-256 are
    skipped, so repeated runs are cheap. `rebuild=True` wipes the index first.
    """
 
    def __init__(self, settings: Settings, embeddings: EmbeddingProvider, store: ChromaVectorStore, bm25: BM25Index):
        self.settings = settings
        self.embeddings = embeddings
        self.store = store
        self.bm25 = bm25
        self.registry_path = settings.paths.index_dir / "registry.json"
 
    # registry
    def _load_registry(self) -> dict:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("registry.json is corrupt; starting a fresh registry")
        return {"embedding": {}, "documents": {}}
 
    def _save_registry(self, registry: dict) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
 
    # pipeline
    def run(
        self,
        source_dir: Path | None = None,
        rebuild: bool = False,
        dry_run: bool = False,
        limit: int | None = None,
    ) -> IngestReport:
        source = Path(source_dir) if source_dir else self.settings.paths.corpus_dir
        if not source.exists():
            raise IngestError(f"source directory does not exist: {source}")
        report = IngestReport(
            run_id=uuid.uuid4().hex[:12],
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=str(source),
            rebuild=rebuild,
            dry_run=dry_run,
            embedding_provider=self.embeddings.name,
            embedding_dim=self.embeddings.dim,
        )
        registry = self._load_registry()
 
        if rebuild and not dry_run:
            self.store.clear()
            self.bm25.clear()
            registry = {"embedding": {}, "documents": {}}
            logger.info("rebuild requested: index cleared")
 
        existing_dim = registry.get("embedding", {}).get("dim")
        if existing_dim and existing_dim != self.embeddings.dim:
            raise IngestError(
                f"embedding dimension changed ({existing_dim} -> {self.embeddings.dim}); rerun with rebuild=True"
            )
 
        files = discover_files(source)
        if limit:
            files = files[:limit]
        report.docs_seen = len(files)
        logger.info("ingestion start: %d files from %s (rebuild=%s dry_run=%s)", len(files), source, rebuild, dry_run)
 
        pending: list[tuple[DocMeta, list]] = []
        t_load = time.perf_counter()
        for i, path in enumerate(files, 1):
            try:
                loaded = load_file(path)
                category = path.parent.name.lower() if path.parent != source else "general"
                meta = extract_metadata(loaded.text, path, category)
                meta.pii_findings = scan_text(loaded.text)
                previous = registry.get("documents", {}).get(meta.document_id)
                if previous and previous.get("sha256") == meta.sha256 and not rebuild:
                    report.docs_skipped += 1
                    continue
                chunks = chunk_document(
                    meta,
                    loaded.text,
                    max_words=self.settings.chunking.max_words,
                    overlap_words=self.settings.chunking.overlap_words,
                    min_words=self.settings.chunking.min_words,
                )
                pending.append((meta, chunks))
                for warning in loaded.warnings:
                    report.errors.append({"file": path.name, "error": warning})
            except Exception as exc:  # one bad file must not stop the pipeline
                logger.exception("failed to process %s", path)
                report.errors.append({"file": str(path), "error": str(exc)})
            if i % 500 == 0:
                logger.info("loaded %d/%d files", i, len(files))
        report.stage_ms["load_enrich_chunk"] = round((time.perf_counter() - t_load) * 1000, 1)
        logger.info("chunking complete: %d new documents, %d chunks", len(pending), sum(len(c) for _, c in pending))
 
        if dry_run:
            report.docs_ingested = len(pending)
            report.chunks_indexed = sum(len(c) for _, c in pending)
            report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return report
 
        # embed
        all_chunks = [chunk for _, chunks in pending for chunk in chunks]
        t_embed = time.perf_counter()
        embeddings: list[list[float]] = []
        batch = self.settings.embedding.batch_size
        for start in range(0, len(all_chunks), batch):
            batch_texts = [c.text for c in all_chunks[start : start + batch]]
            embeddings.extend(self.embeddings.embed_documents(batch_texts))
            if start and start % (batch * 20) == 0:
                logger.info("embedded %d/%d chunks", start + len(batch_texts), len(all_chunks))
        report.stage_ms["embed"] = round((time.perf_counter() - t_embed) * 1000, 1)
 
        # index
        t_index = time.perf_counter()
        self.store.add_chunks(all_chunks, embeddings)
        self.bm25.add_entries(
            Bm25Entry(id=c.id, text=c.text, meta=c.metadata) for c in all_chunks
        )
        self.bm25.persist()
        report.stage_ms["index"] = round((time.perf_counter() - t_index) * 1000, 1)
 
        # registry
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for meta, chunks in pending:
            registry.setdefault("documents", {})[meta.document_id] = {
                "document_id": meta.document_id,
                "title": meta.title,
                "filename": meta.filename,
                "category": meta.category,
                "domain": meta.domain,
                "family": meta.family,
                "version": meta.version,
                "jurisdiction": meta.jurisdiction,
                "effective_date": meta.effective_date,
                "product": meta.product,
                "n_chunks": len(chunks),
                "n_chars": meta.n_chars,
                "sha256": meta.sha256,
                "pii_findings": meta.pii_findings,
                "source_path": meta.source_path,
                "ingested_at": now,
            }
        registry["embedding"] = {"provider": self.embeddings.name, "dim": self.embeddings.dim}
        self._save_registry(registry)
 
        report.docs_ingested = len(pending)
        report.chunks_indexed = len(all_chunks)
        report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.info(
            "ingestion done: %d ingested, %d skipped, %d chunks in %.1fs",
            report.docs_ingested,
            report.docs_skipped,
            report.chunks_indexed,
            sum(report.stage_ms.values()) / 1000,
        )
        return report
 
    def delete_document(self, document_id: str) -> bool:
        registry = self._load_registry()
        if document_id not in registry.get("documents", {}):
            return False
        self.store.delete_document(document_id)
        self.bm25.remove_document(document_id)
        self.bm25.persist()
        registry["documents"].pop(document_id)
        self._save_registry(registry)
        return True
 
    def documents(self) -> list[dict]:
        docs = list(self._load_registry().get("documents", {}).values())
        return sorted(docs, key=lambda d: d["document_id"])
 
    def domains(self) -> dict[str, str]:
        """document_id -> domain, used by the ACL service."""
        return {d["document_id"]: d.get("domain", "") for d in self._load_registry().get("documents", {}).values()}
 