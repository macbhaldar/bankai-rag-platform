"""Optional cross-encoder reranker"""

from __future__ import annotations
 
from app.core.config import Settings
from app.core.logging import get_logger
 
logger = get_logger(__name__)
 
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
 
 
class CrossEncoderReranker:
    _model = None  # class-level cache so it loads once per process
 
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model_name = model_name
        self.broken = False
 
    def _load(self):
        if CrossEncoderReranker._model is None:
            from sentence_transformers import CrossEncoder  # optional dependency
 
            logger.info("loading cross-encoder reranker: %s", self.model_name)
            CrossEncoderReranker._model = CrossEncoder(self.model_name, max_length=512)
        return CrossEncoderReranker._model
 
    def rerank(self, query: str, hits):
        if self.broken:
            return hits
        try:
            model = self._load()
            pairs = [(query, hit.text) for hit in hits]
            scores = model.predict(pairs)
            for hit, score in zip(hits, scores):
                hit.rerank_score = float(score)
            hits.sort(key=lambda h: h.rerank_score, reverse=True)
            return hits
        except Exception as exc:
            logger.warning("reranker disabled after failure: %s", exc)
            self.broken = True
            return hits
 
 
def maybe_reranker(settings: Settings) -> CrossEncoderReranker | None:
    if not settings.retrieval.use_reranker:
        return None
    try:
        import sentence_transformers  # noqa: F401
 
        return CrossEncoderReranker()
    except ImportError:
        logger.warning("BANKRAG_USE_RERANKER is on but sentence-transformers is not installed; skipping reranker")
        return None