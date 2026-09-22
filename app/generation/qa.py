"""RAG orchestration: retrieve -> generate -> enforce citations -> ground the answer."""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from app.core.audit import AuditLog
from app.core.config import Settings
from app.core.logging import get_logger
from app.generation.llm import BaseLLM, ExtractiveLLM, LLMError
from app.generation.prompts import (SYSTEM_PROMPT, build_context_block, build_paraphrase_prompt, build_user_message,)
from app.retrieval.bm25 import STOPWORDS
from app.retrieval.hybrid import HybridRetriever, RetrievalHit

logger = get_logger(__name__)

CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_WORD_RE = re.compile(r"[a-z0-9]+")
 
def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 2 and t not in STOPWORDS}

@dataclass
class Citation:
    n: int
    document_id: str
    title: str
    section: str
    snippet: str
    source_path: str
    score: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "document_id": self.document_id,
            "title": self.title,
            "section": self.section,
            "snippet": self.snippet,
            "source_path": self.source_path,
            "score": round(self.score, 4),
            }

@dataclass
class QAResult:
    question: str
    answer: str
    mode: str
    provider: str
    model: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    grounding: float = 0.0
    warning: str | None = None
    timings_ms: dict = field(default_factory=dict)
 
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "citations": [c.to_dict() for c in self.citations],
            "sources": self.sources,
            "grounding": round(self.grounding, 3),
            "warning": self.warning,
            "timings_ms": self.timings_ms,
            }

class RAGChain:
    def __init__(self, llm: BaseLLM, retriever: HybridRetriever, settings: Settings, audit: AuditLog | None = None):
        self.llm = llm
        self.retriever = retriever
        self.settings = settings
        self.audit = audit

    def _paraphrases(self, question: str) -> list[str]:
        try:
            result = self.llm.generate(
                question,
                [
                    {"role": "system", "content": "You rewrite search queries. Output only the rewritten queries."},
                    {"role": "user", "content": build_paraphrase_prompt(question)},
                ],
                hits=[],
                )
            lines = [ln.strip("-•. ").strip() for ln in result.text.splitlines() if ln.strip()]
            return [ln for ln in lines if 3 < len(ln) < 200][:2]
        except Exception as exc:
            logger.warning("query expansion failed: %s", exc)
            return []

    @staticmethod
    def _grounding(answer: str, context_texts: list[str]) -> float:
        answer_tokens = _content_tokens(answer)
        if not answer_tokens:
            return 0.0
        context_tokens: set[str] = set()
        for text in context_texts:
            context_tokens |= _content_tokens(text)
        return len(answer_tokens & context_tokens) / len(answer_tokens)

    def ask(
        self,
        question: str,
        k: int | None = None,
        filters: dict | None = None,
        history: list[dict] | None = None,
        allowed_document_ids: set[str] | None = None,
        dedupe_families: bool = True,
    ) -> QAResult:
        t_retrieve = time.perf_counter()
        extra_queries = self._paraphrases(question) if (
            self.settings.retrieval.multi_query and self.llm.mode == "generative"
        ) else []
        hits = self.retriever.retrieve(
            question,
            k=k or self.settings.retrieval.final_k,
            filters=filters,
            allowed_document_ids=allowed_document_ids,
            extra_queries=extra_queries,
            dedupe_families=dedupe_families,
        )
        retrieve_ms = round((time.perf_counter() - t_retrieve) * 1000, 1)

        timings = {"retrieval_ms": retrieve_ms}
        if not hits:
            return QAResult(
                question=question,
                answer="I could not find any relevant documents in the knowledge base for this question.",
                mode=self.llm.mode,
                provider=self.llm.provider,
                model=self.llm.model,
                warning="no_results",
                timings_ms=timings,
                )

        context = build_context_block(hits[:8])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend({"role": m["role"], "content": m["content"]} for m in history[-4:])
        messages.append({"role": "user", "content": build_user_message(question, context, history)})

        t_generate = time.perf_counter()
        warning: str | None = None
        try:
            result = self.llm.generate(question, messages, hits)
        except LLMError as exc:
            logger.error("LLM generation failed, falling back to extractive: %s", exc)
            result = ExtractiveLLM().generate(question, messages, hits)
            warning = "llm_error_fallback"
        timings["generation_ms"] = round((time.perf_counter() - t_generate) * 1000, 1)

        answer = result.text.strip()
        if self.llm.mode == "generative":
            # drop citations that point beyond the retrieved context
            answer = CITATION_RE.sub(
                lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(hits[:8]) else "",
                answer,
            ).strip()

        cited = sorted({int(n) for n in CITATION_RE.findall(answer) if 1 <= int(n) <= len(hits[:8])})
        if not cited:
            cited = list(range(1, min(3, len(hits)) + 1))
            if result.mode == "generative":
                warning = warning or "uncited_answer"

        cited_texts = [hits[n - 1].text for n in cited] or [h.text for h in hits[:3]]
        grounding = self._grounding(answer, cited_texts)
        if hits[0].dense_score < self.settings.retrieval.low_confidence_dense:
            warning = warning or "low_confidence"

        citations = []
        for n in cited:
            hit = hits[n - 1]
            citations.append(
                Citation(
                    n=n,
                    document_id=hit.document_id,
                    title=hit.meta.get("title", ""),
                    section=hit.section,
                    snippet=hit.snippet(260),
                    source_path=hit.meta.get("source_path", ""),
                    score=hit.fused_score,
                )
            )

        total_ms = round(retrieve_ms + timings["generation_ms"], 1)
        timings["total_ms"] = total_ms
        if self.audit:
            self.audit.append(
                "qa",
                question=question,
                mode=result.mode,
                provider=result.provider,
                grounding=round(grounding, 3),
                latency_ms=total_ms,
                top_documents=[h.document_id for h in hits[:3]],
                warning=warning,
            )
        return QAResult(
            question=question,
            answer=answer,
            mode=result.mode,
            provider=result.provider,
            model=result.model,
            citations=citations,
            sources=[h.to_dict() for h in hits[:5]],
            grounding=grounding,
            warning=warning,
            timings_ms=timings,
        )