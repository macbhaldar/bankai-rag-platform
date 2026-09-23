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


# Benchmark 1: Retrieval queries with graded relevance judgments

def evaluate_retrieval(services, k: int = 10, limit: int | None = None) -> EvalReport:
    qa_dir = services.settings.paths.qa_dir
    retrieval_dir = qa_dir.parent / "retrieval"
    queries_path, judgments_path = retrieval_dir / "queries.jsonl", retrieval_dir / "relevance_judgments.jsonl"
    if not queries_path.exists() or not judgments_path.exists():
        raise EvalError(f"retrieval benchmark not found under {retrieval_dir}")
 
    relevance: dict[str, dict[str, int]] = {}
    for row in _read_jsonl(judgments_path):
        relevance.setdefault(row["query_id"], {})[row["document_id"]] = int(row["relevance"])
    cases = []
    for row in _read_jsonl(queries_path):
        rels = relevance.get(row["query_id"])
        if rels:
            cases.append({"query_id": row["query_id"], "query": row["query"], "relevant": rels})
    if limit:
        cases = cases[:limit]
 
    hits_total, mrr_total, ndcg_total, fam_hits_total, fam_mrr_total = 0.0, 0.0, 0.0, 0.0, 0.0
    per_case: list[dict] = []
    started = time.perf_counter()
    for i, case in enumerate(cases, 1):
        results = services.retriever.retrieve(case["query"], k=k, dedupe_families=False)
        ranked_docs = list(dict.fromkeys(h.document_id for h in results))
        relevant = {doc for doc, rel in case["relevant"].items() if rel > 0}
        ideal = sorted(case["relevant"].values(), reverse=True)
 
        dcg, found_rank = 0.0, None
        for rank, doc in enumerate(ranked_docs, 1):
            rel = case["relevant"].get(doc, 0)
            if rel > 0:
                dcg += rel / math.log2(rank + 1)
                if found_rank is None:
                    found_rank = rank
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal[:k], 1)) or 1.0
 
        fam_relevant = {family_of(doc) for doc in relevant}
        fam_rank = next((r for r, doc in enumerate(ranked_docs, 1) if family_of(doc) in fam_relevant), None)
 
        hit = 1.0 if found_rank else 0.0
        fam_hit = 1.0 if fam_rank else 0.0
        hits_total += hit
        mrr_total += 1.0 / found_rank if found_rank else 0.0
        ndcg_total += dcg / idcg
        fam_hits_total += fam_hit
        fam_mrr_total += 1.0 / fam_rank if fam_rank else 0.0
        per_case.append(
            {"query_id": case["query_id"], "query": case["query"], "expected": sorted(relevant),
             "first_hit_rank": found_rank, "hit": hit, "family_hit": fam_hit,
             "top3": ranked_docs[:3]}
             )
        if i % 100 == 0:
            logger.info("eval retrieval %d/%d", i, len(cases))
    duration = (time.perf_counter() - started) * 1000
    n = max(len(cases), 1)
    return EvalReport(
        benchmark="retrieval",
        k=k,
        n_cases=len(cases),
        metrics={
            "recall_at_k": hits_total / n,
            "mrr": mrr_total / n,
            "ndcg_at_k": ndcg_total / n,
            "family_recall_at_k": fam_hits_total / n,
            "family_mrr": fam_mrr_total / n,
        },
        duration_ms=duration,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_info=_model_info(services),
        per_case=per_case,
        )


# Benchmark 2: QA split (document + section ground truth)

def evaluate_qa(services, split: str = "test", k: int = 10, limit: int | None = None) -> EvalReport:
    path = services.settings.paths.qa_dir / f"qa_{split}.jsonl"
    if not path.exists():
        raise EvalError(f"QA split not found: {path}")
    cases = _read_jsonl(path)
    if limit:
        cases = cases[:limit]
 
    doc_hits = fam_hits = sec_hits = mrr_total = 0.0
    per_case: list[dict] = []
    started = time.perf_counter()
    for i, case in enumerate(cases, 1):
        results = services.retriever.retrieve(case["question"], k=k, dedupe_families=False)
        ranked = list(dict.fromkeys(h.document_id for h in results))
        expected = case["document_id"]
        expected_family = family_of(expected)
        rank = next((r for r, doc in enumerate(ranked, 1) if doc == expected), None)
        fam_rank = next((r for r, doc in enumerate(ranked, 1) if family_of(doc) == expected_family), None)
        sec_hit = any(
            h.document_id == expected and h.section.strip().lower() == case.get("section", "").strip().lower()
            for h in results
            )
        doc_hits += 1.0 if rank else 0.0
        fam_hits += 1.0 if fam_rank else 0.0
        sec_hits += 1.0 if sec_hit else 0.0
        mrr_total += 1.0 / rank if rank else 0.0
        per_case.append(
            {"id": case["id"], "question": case["question"], "expected": expected,
             "expected_section": case.get("section", ""), "first_hit_rank": rank,
             "doc_hit": 1.0 if rank else 0.0, "family_hit": 1.0 if fam_rank else 0.0,
             "section_hit": 1.0 if sec_hit else 0.0}
             )
        if i % 50 == 0:
            logger.info("eval qa %d/%d", i, len(cases))
    duration = (time.perf_counter() - started) * 1000
    n = max(len(cases), 1)
    return EvalReport(
        benchmark=f"qa_{split}",
        k=k,
        n_cases=len(cases),
        metrics={
            "doc_hit_rate_at_k": doc_hits / n,
            "family_hit_rate_at_k": fam_hits / n,
            "section_hit_rate_at_k": sec_hits / n,
            "mrr": mrr_total / n,},
        duration_ms=duration,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_info=_model_info(services),
        per_case=per_case,
        )


# Benchmark 3: hard-negative robustness

def evaluate_hard_negatives(services, k: int = 50, limit: int | None = None) -> EvalReport:
    retrieval_dir = services.settings.paths.qa_dir.parent / "retrieval"
    hn_path, queries_path = retrieval_dir / "hard_negatives.jsonl", retrieval_dir / "queries.jsonl"
    if not hn_path.exists():
        raise EvalError(f"hard negatives benchmark not found: {hn_path}")
    positives: dict[str, str] = {}
    for row in _read_jsonl(queries_path):
        positives.setdefault(row["query"], row["positive_document"])

    matched = 0
    survived = 0
    per_case: list[dict] = []
    started = time.perf_counter()
    rows = _read_jsonl(hn_path)
    if limit:
        rows = rows[:limit]
    for row in rows:
        positive_doc = positives.get(row["query"])
        if not positive_doc:
            continue
        matched += 1
        results = services.retriever.retrieve(row["query"], k=k, dedupe_families=False)
        docs = list(dict.fromkeys(h.document_id for h in results))
        rank_pos = next((r for r, d in enumerate(docs, 1) if d == positive_doc), None)
        rank_neg = next((r for r, d in enumerate(docs, 1) if d == row["hard_negative_document"]), None)
        if rank_pos is not None and (rank_neg is None or rank_pos < rank_neg):
            survived += 1
        per_case.append(
            {"query_id": row["query_id"], "query": row["query"], "positive": positive_doc,
             "hard_negative": row["hard_negative_document"], "rank_positive": rank_pos,
             "rank_negative": rank_neg})
    duration = (time.perf_counter() - started) * 1000
    if matched == 0:
        raise EvalError("no hard-negative cases could be joined to a positive document")
    return EvalReport(
        benchmark="hard_negatives",
        k=k,
        n_cases=matched,
        metrics={"positive_outranks_negative": survived / matched,
                 "positive_found_at_k": sum(1 for c in per_case if c["rank_positive"]) / matched},
        duration_ms=duration,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_info=_model_info(services),
        per_case=per_case,)

def run_benchmark(services, benchmark: str = "retrieval", k: int = 10, limit: int | None = None) -> EvalReport:
    if benchmark == "retrieval":
        return evaluate_retrieval(services, k=k, limit=limit)
    if benchmark.startswith("qa"):
        split = benchmark.split("_", 1)[1] if "_" in benchmark else "test"
        return evaluate_qa(services, split=split, k=k, limit=limit)
    if benchmark == "hard_negatives":
        return evaluate_hard_negatives(services, k=k, limit=limit)
    raise EvalError(f"unknown benchmark: {benchmark!r} (use retrieval | qa_<split> | hard_negatives)")

def save_report(report: EvalReport, eval_dir: Path) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = eval_dir / f"eval_{report.benchmark}_{stamp}.json"
    path.write_text(json.dumps(report.to_dict(max_cases=None), ensure_ascii=False, indent=2), encoding="utf-8")
    return path