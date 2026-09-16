"""Pluggable embedding providers"""
from __future__ import annotations
 
import hashlib
import math
import re
from abc import ABC, abstractmethod
 
from app.core.config import Settings
from app.core.logging import get_logger
 
logger = get_logger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
 
 
class EmbeddingError(RuntimeError):
    pass
 
 
class EmbeddingProvider(ABC):
    name: str = "base"
    dim: int = 0
 
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
 
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
 
 
class ChromaDefaultEmbeddings(EmbeddingProvider):
    """chromadb's bundled ONNX all-MiniLM-L6-v2 (downloads the model on first use)."""
 
    name = "chroma_default(onnx-minilm-l6-v2)"
    dim = 384
 
    def __init__(self) -> None:
        self._ef = None
 
    def _function(self):
        if self._ef is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
 
            self._ef = DefaultEmbeddingFunction()
        return self._ef
 
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        ef = self._function()
        try:
            return ef(input=texts)
        except TypeError:  # older chromadb signature
            return ef(texts)
 
 
class SentenceTransformerEmbeddings(EmbeddingProvider):
    name = "sentence_transformers"
 
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency
 
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
 
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)]
 
 
class OpenAIEmbeddings(EmbeddingProvider):
    name = "openai"
 
    def __init__(self, base_url: str, api_key: str, model: str = "text-embedding-3-small") -> None:
        import requests
 
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._model = model
        self._session = requests.Session()
 
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._session.post(
            f"{self._base}/embeddings",
            json={"input": texts, "model": self._model},
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()["data"]
        vectors = [item["embedding"] for item in sorted(payload, key=lambda d: d["index"])]
        self.dim = len(vectors[0])
        return vectors
 
 
class HashEmbeddings(EmbeddingProvider):
    """Offline bag-of-words hashing vectorizer (deterministic; tests/CI only)."""
 
    name = "hash"
    dim = 384
 
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
                vec[(h >> 16) % self.dim] += 0.5
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors
 
 
def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding.provider.lower()
    if provider == "hash":
        return HashEmbeddings()
    if provider == "chroma_default":
        return ChromaDefaultEmbeddings()
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddings(settings.embedding.model or "all-MiniLM-L6-v2")
    if provider == "openai":
        return OpenAIEmbeddings(settings.llm.base_url, settings.llm.api_key)
    if provider == "auto":
        try:
            return ChromaDefaultEmbeddings()
        except Exception as exc:
            logger.warning("chroma default embeddings unavailable (%s); using hash fallback", exc)
            return HashEmbeddings()
    raise EmbeddingError(f"unknown embedding provider: {provider!r}")
