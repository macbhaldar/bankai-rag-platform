"""BM25 keyword index over the same chunks that live in the vector store."""
from __future__ import annotations
 
import json
import re
from dataclasses import dataclass
from pathlib import Path
 
from rank_bm25 import BM25Okapi
 
from app.core.logging import get_logger
 
logger = get_logger(__name__)
 
STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in into is it its of on or
    should that the their then there these they this to was were what when where
    which who will with within without you your our we us do does done must may
    can could shall would""".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
 
 
def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]
 
 
@dataclass
class Bm25Entry:
    id: str
    text: str
    meta: dict
 
 
class BM25Index:
 
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self.entries: list[Bm25Entry] = []
        self._bm25: BM25Okapi | None = None
        self._by_id: dict[str, Bm25Entry] = {}
        if corpus_path.exists():
            try:
                raw = json.loads(corpus_path.read_text(encoding="utf-8"))
                self.entries = [Bm25Entry(id=e["id"], text=e["text"], meta=e["meta"]) for e in raw]
                self._by_id = {e.id: e for e in self.entries}
            except (json.JSONDecodeError, KeyError):
                logger.warning("bm25 corpus is corrupt; starting empty")
        logger.info("bm25 corpus loaded: %d entries", len(self.entries))
 
    def _mark_dirty(self) -> None:
        self._bm25 = None
 
    def _ensure_built(self) -> BM25Okapi:
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(e.text) for e in self.entries])
        return self._bm25
 
    def add_entries(self, entries) -> None:
        for entry in entries:
            if entry.id in self._by_id:
                continue
            self.entries.append(entry)
            self._by_id[entry.id] = entry
        self._mark_dirty()
 
    def remove_document(self, document_id: str) -> int:
        keep = [e for e in self.entries if e.meta.get("document_id") != document_id]
        removed = len(self.entries) - len(keep)
        self.entries = keep
        self._by_id = {e.id: e for e in self.entries}
        self._mark_dirty()
        return removed
 
    def clear(self) -> None:
        self.entries = []
        self._by_id = {}
        self._mark_dirty()
        self.persist()
 
    def persist(self) -> None:
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus_path.write_text(
            json.dumps([{"id": e.id, "text": e.text, "meta": e.meta} for e in self.entries]),
            encoding="utf-8",
        )
 
    def ids_matching_meta(self, meta_filter: dict | None) -> set[str] | None:
        """Document-ids whose chunk metadata satisfies a simple equality filter."""
        if not meta_filter:
            return None
        allowed: set[str] = set()
        for entry in self.entries:
            if all(entry.meta.get(k) == v for k, v in meta_filter.items()):
                allowed.add(entry.meta.get("document_id", "")) if allowed is not None else allowed
                if allowed is None:
                    allowed = set()
                allowed.add(entry.meta.get("document_id", ""))
        return allowed
 
    def search(self, query: str, k: int, allowed_document_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.entries:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        bm25 = self._ensure_built()
        scores = bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[tuple[str, float]] = []
        for i in ranked:
            entry = self.entries[i]
            if scores[i] <= 0:
                break
            if allowed_document_ids is not None and entry.meta.get("document_id") not in allowed_document_ids:
                continue
            results.append((entry.id, float(scores[i])))
            if len(results) >= k:
                break
        return results