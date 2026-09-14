"""Application settings: YAML defaults layered with BANKRAG_* environment overrides."""
from __future__ import annotations
 
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
 
import yaml
from pydantic import BaseModel, Field
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = PROJECT_ROOT / "config" / "settings.yaml"
 
 
class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    key: str = ""
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
 
 
class PathSettings(BaseModel):
    corpus_dir: Path = PROJECT_ROOT / "dataset" / "raw_documents"
    qa_dir: Path = PROJECT_ROOT / "dataset" / "qa"
    structured_dir: Path = PROJECT_ROOT / "dataset" / "structured"
    data_dir: Path = PROJECT_ROOT / "data"
 
    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"
 
    @property
    def audit_dir(self) -> Path:
        return self.data_dir / "audit"
 
    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"
 
    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval_results"
 
    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"
 
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"
 
 
class EmbeddingSettings(BaseModel):
    provider: str = "auto"  # auto | chroma_default | sentence_transformers | openai | hash
    model: str = ""
    batch_size: int = 64
 
 
class ChunkingSettings(BaseModel):
    max_words: int = 220
    overlap_words: int = 40
    min_words: int = 30
 
 
class RetrievalSettings(BaseModel):
    dense_k: int = 24
    bm25_k: int = 24
    final_k: int = 8
    rrf_k: int = 60
    use_reranker: bool = False
    multi_query: bool = False
    low_confidence_dense: float = 0.30
 
 
class LLMSettings(BaseModel):
    provider: str = "auto"  # auto | openai | ollama | extractive
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 700
    ollama_url: str = "http://127.0.0.1:11434"
 
 
class SecuritySettings(BaseModel):
    mask_pii: bool = True
 
 
class StructuredSettings(BaseModel):
    enabled: bool = True
    max_rows: int = 200
 
 
class Settings(BaseModel):
    api: ApiSettings = Field(default_factory=ApiSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    structured: StructuredSettings = Field(default_factory=StructuredSettings)
    log_level: str = "INFO"
 
 
# section.key -> (env var, python type)
_ENV_MAP: dict[str, tuple[str, type]] = {
    "api.host": ("BANKRAG_API_HOST", str),
    "api.port": ("BANKRAG_API_PORT", int),
    "api.key": ("BANKRAG_API_KEY", str),
    "paths.corpus_dir": ("BANKRAG_CORPUS_DIR", Path),
    "paths.qa_dir": ("BANKRAG_QA_DIR", Path),
    "paths.structured_dir": ("BANKRAG_STRUCTURED_DIR", Path),
    "paths.data_dir": ("BANKRAG_DATA_DIR", Path),
    "embedding.provider": ("BANKRAG_EMBEDDING_PROVIDER", str),
    "embedding.model": ("BANKRAG_EMBEDDING_MODEL", str),
    "chunking.max_words": ("BANKRAG_CHUNK_MAX_WORDS", int),
    "chunking.overlap_words": ("BANKRAG_CHUNK_OVERLAP_WORDS", int),
    "chunking.min_words": ("BANKRAG_CHUNK_MIN_WORDS", int),
    "retrieval.dense_k": ("BANKRAG_DENSE_K", int),
    "retrieval.bm25_k": ("BANKRAG_BM25_K", int),
    "retrieval.final_k": ("BANKRAG_FINAL_K", int),
    "retrieval.rrf_k": ("BANKRAG_RRF_K", int),
    "retrieval.use_reranker": ("BANKRAG_USE_RERANKER", bool),
    "retrieval.multi_query": ("BANKRAG_MULTI_QUERY", bool),
    "llm.provider": ("BANKRAG_LLM_PROVIDER", str),
    "llm.model": ("BANKRAG_LLM_MODEL", str),
    "llm.base_url": ("BANKRAG_LLM_BASE_URL", str),
    "llm.api_key": ("BANKRAG_LLM_API_KEY", str),
    "llm.temperature": ("BANKRAG_LLM_TEMPERATURE", float),
    "llm.max_tokens": ("BANKRAG_LLM_MAX_TOKENS", int),
    "llm.ollama_url": ("BANKRAG_OLLAMA_URL", str),
    "security.mask_pii": ("BANKRAG_MASK_PII", bool),
    "structured.enabled": ("BANKRAG_STRUCTURED_ENABLED", bool),
    "structured.max_rows": ("BANKRAG_STRUCTURED_MAX_ROWS", int),
    "log_level": ("BANKRAG_LOG_LEVEL", str),
}
 
 
def _coerce(raw: str, py_type: type) -> Any:
    if py_type is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if py_type is int:
        return int(raw)
    if py_type is float:
        return float(raw)
    if py_type is Path:
        p = Path(os.path.expanduser(raw.strip()))
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    return raw
 
 
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
 
 
def load_settings(yaml_path: Path | None = None) -> Settings:
    data: dict[str, Any] = _load_yaml(yaml_path or DEFAULT_YAML)
    for dotted, (env_var, py_type) in _ENV_MAP.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        section, key = dotted.split(".", 1)
        data.setdefault(section, {})[key] = _coerce(raw, py_type)
    return Settings(**data)
 
 
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
 
 
def reset_settings() -> None:
    """Drop the cached settings (used by tests and reconfiguration)."""
    get_settings.cache_clear()