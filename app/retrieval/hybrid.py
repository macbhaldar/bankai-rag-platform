"""Hybrid retrieval: dense vector search + BM25 fused with Reciprocal Rank Fusion"""

from __future__ import annotations
 
from dataclasses import dataclass
 
from app.core.config import Settings
from app.core.logging import get_logger
from app.pipelines.chunking import version_key
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import ChromaVectorStore, build_where, where_document_in
 
logger = get_logger(__name__)

_MAX_DENSE_IN_IDS = 300
 
@dataclass
class RetrievalHit:
    id: str
    document_id: str
    family: str
    version: str
    section: str
    text: str
    meta: dict
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None
 
    def snippet(self, length: int = 240) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= length else text[: length - 1] + "…"
 
    def to_dict(self, include_text: bool = False) -> dict:
        payload = {
            "id": self.id,
            "document_id": self.document_id,
            "title": self.meta.get("title", ""),
            "family": self.family,
            "version": self.version,
            "section": self.section,
            "category": self.meta.get("category", ""),
            "jurisdiction": self.meta.get("jurisdiction", ""),
            "product": self.meta.get("product", ""),
            "dense_score": round(self.dense_score, 4),
            "sparse_score": round(self.sparse_score, 4),
            "fused_score": round(self.fused_score, 4),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
            "snippet": self.snippet(),
        }
        if include_text:
            payload["text"] = self.text
        return payload
 
 
@dataclass
class _Fused:
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    dense_seen: bool = False
 
 
class HybridRetriever:
    def __init__(
        self,
        store: ChromaVectorStore,
        bm25: BM25Index,
        embeddings: EmbeddingProvider,
        settings: Settings,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.store = store
        self.bm25 = bm25
        self.embeddings = embeddings
        self.settings = settings
        self.reranker = reranker
 
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filters: dict | None = None,
        allowed_document_ids: set[str] | None = None,
        extra_queries: list[str] | None = None,
        dedupe_families: bool = False,
    ) -> list[RetrievalHit]:
        cfg = self.settings.retrieval
        k = k or cfg.final_k
        filters = {key: value for key, value in (filters or {}).items() if value}
        where = build_where(**filters) if filters else None
 
        meta_ids = self.bm25.ids_matching_meta(filters) if filters else None
        if allowed_document_ids is None:
            effective_ids = meta_ids
        elif meta_ids is None:
            effective_ids = allowed_document_ids
        else:
            effective_ids = allowed_document_ids & meta_ids
 
        # dense search strategy
        if effective_ids is None:
            dense_where = where
        elif len(effective_ids) <= _MAX_DENSE_IN_IDS:
            dense_where = where_document_in(sorted(effective_ids))
        else:
            dense_where = where
 
        # sparse search restriction (document-id set)
        sparse_ids = effective_ids
 
        fused: dict[str, _Fused] = {}
        collected: dict[str, dict] = {}
        dense_max, sparse_max = 1e-9, 1e-9
        rrf_k = cfg.rrf_k
 
        for q in [query] + [extra for extra in (extra_queries or []) if extra]:
            dense = self.store.query(
                self.embeddings.embed_query(q),
                cfg.dense_k * (2 if effective_ids is not None and len(effective_ids) > _MAX_DENSE_IN_IDS else 1),
                where=dense_where,
            )
            if effective_ids is not None and len(effective_ids) > _MAX_DENSE_IN_IDS:
                dense = [d for d in dense if d["meta"].get("document_id") in effective_ids]
            for rank, item in enumerate(dense, 1):
                entry = fused.setdefault(item["id"], _Fused())
                entry.fused_score += 1.0 / (rrf_k + rank)
                entry.dense_seen = True
                entry.dense_score = max(entry.dense_score, item["similarity"])
                collected[item["id"]] = item
                dense_max = max(dense_max, item["similarity"])
 
            for rank, (chunk_id, score) in enumerate(self.bm25.search(q, cfg.bm25_k, allowed_document_ids=sparse_ids), 1):
                entry = fused.setdefault(chunk_id, _Fused())
                entry.fused_score += 1.0 / (rrf_k + rank)
                entry.sparse_score = max(entry.sparse_score, score)
                sparse_max = max(sparse_max, score)
 
        hits: list[RetrievalHit] = []
        missing = [cid for cid in fused if cid not in collected]
        if missing:
            collected.update(self.store.get(missing))
        for chunk_id, f in fused.items():
            blob = collected.get(chunk_id)
            if blob is None:
                continue
            meta = blob["meta"]
            hits.append(
                RetrievalHit(
                    id=chunk_id,
                    document_id=meta.get("document_id", ""),
                    family=meta.get("family", ""),
                    version=meta.get("version", "0"),
                    section=meta.get("section", ""),
                    text=blob["text"],
                    meta=meta,
                    dense_score=f.dense_score / dense_max if f.dense_seen else 0.0,
                    sparse_score=f.sparse_score / sparse_max if f.sparse_score else 0.0,
                    fused_score=f.fused_score,
                )
            )
 
        if dedupe_families:
            best_by_family: dict[str, RetrievalHit] = {}
            for hit in hits:
                current = best_by_family.get(hit.family)
                if current is None or (version_key(hit.version), hit.fused_score) > (
                    version_key(current.version),
                    current.fused_score,
                ):
                    best_by_family[hit.family] = hit
            hits = list(best_by_family.values())
 
        hits.sort(key=lambda h: h.fused_score, reverse=True)
        hits = hits[: max(k * 3, 20)]
 
        if self.reranker is not None and hits:
            hits = self.reranker.rerank(query, hits)
 
        hits.sort(
            key=lambda h: h.rerank_score if h.rerank_score is not None else h.fused_score,
            reverse=True,
        )
        return hits[:k]
 
    def _ids_from_where(self, where: dict) -> set[str] | None:
        """Extract document_id restrictions from a chroma where clause."""
        if "document_id" in where:
            op = where["document_id"]
            if "$eq" in op:
                return {op["$eq"]}
            if "$in" in op:
                return set(op["$in"])
        if "$and" in where:
            for condition in where["$and"]:
                if "document_id" in condition:
                    op = condition["document_id"]
                    if "$eq" in op:
                        return {op["$eq"]}
                    if "$in" in op:
                        return set(op["$in"])
        return None