"""Thin wrapper over a persistent ChromaDB collection."""
from __future__ import annotations
 
from pathlib import Path
 
from app.core.logging import get_logger
from app.pipelines.chunking import Chunk
 
logger = get_logger(__name__)
 
 
def build_where(
    category: str | None = None,
    jurisdiction: str | None = None,
    family: str | None = None,
    document_id: str | None = None,
    domain: str | None = None,
) -> dict | None:
    conditions = []
    for field, value in (
        ("category", category),
        ("jurisdiction", jurisdiction),
        ("family", family),
        ("document_id", document_id),
        ("domain", domain),
    ):
        if value:
            conditions.append({field: {"$eq": value}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}
 
 
def where_document_in(document_ids: list[str]) -> dict | None:
    if not document_ids:
        return None
    if len(document_ids) == 1:
        return {"document_id": {"$eq": document_ids[0]}}
    return {"document_id": {"$in": document_ids}}
 
 
class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "bankrag"):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
 
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection_name = collection_name
        self.col = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
 
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 500) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.col.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=embeddings[start : start + batch_size],
            )
 
    def query(self, embedding: list[float], k: int, where: dict | None = None) -> list[dict]:
        result = self.col.query(
            query_embeddings=[embedding],
            n_results=min(k, max(self.col.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        return [
            {
                "id": id_,
                "text": doc or "",
                "meta": meta or {},
                "similarity": max(0.0, min(1.0, 1.0 - dist)),
            }
            for id_, doc, meta, dist in zip(ids, docs, metas, dists)
        ]
 
    def get(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        found = self.col.get(ids=ids, include=["documents", "metadatas"])
        return {
            id_: {"id": id_, "text": doc or "", "meta": meta or {}}
            for id_, doc, meta in zip(found["ids"], found["documents"], found["metadatas"])
        }
 
    def delete_document(self, document_id: str) -> None:
        self.col.delete(where={"document_id": {"$eq": document_id}})
 
    def clear(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # collection may not exist yet
            pass
        self.col = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("vector store cleared")
 
    def count(self) -> int:
        return self.col.count()