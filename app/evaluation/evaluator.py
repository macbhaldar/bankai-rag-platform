"""Evaluation against the benchmarks shipped in dataset/qa and dataset/retrieval."""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from app.core.config import Settings
from app.core.logging import get_logger
from app.pipelines.chunking import family_of

logger = get_logger(__name__)

class EvalError(RuntimeError):
    pass

@dataclass
class EvalReport:
    benchmark: str
    k: int
    n_cases: int
    metrics: dict
    duration_ms: float
    generated_at: str
    model_info: dict
    per_case: list[dict] = field(default_factory=list)

    def to_dict(self, max_cases: int | None = 30) -> dict:
        return {
            "benchmark": self.benchmark,
            "k": self.k,
            "n_cases": self.n_cases,
            "metrics": {key: round(value, 4) for key, value in self.metrics.items()},
            "duration_ms": round(self.duration_ms, 1),
            "generated_at": self.generated_at,
            "model_info": self.model_info,
            "per_case": self.per_case if max_cases is None else self.per_case[:max_cases],
            }

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _model_info(services) -> dict:
    return {
        "embedding": services.embeddings.name,
        "llm": f"{services.llm.provider}:{services.llm.model}",
        "dense_k": services.settings.retrieval.dense_k,
        "bm25_k": services.settings.retrieval.bm25_k,
        "reranker": services.retriever.reranker is not None,
        }