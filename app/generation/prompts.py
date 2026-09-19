"""Prompt templates for the RAG chain."""

from __future__ import annotations 
from app.retrieval.hybrid import RetrievalHit
 
SYSTEM_PROMPT = """You are ARIA, the AI knowledge assistant for internal bank staff.
You answer questions strictly from the numbered context passages retrieved from the bank's
document repository (policies, procedures, standards and manuals).
 
Rules:
1. Answer ONLY from the numbered context passages. Never use outside knowledge.
2. Cite every claim inline with the passage number in square brackets, e.g. [2].
3. If the context does not contain the answer, say so plainly and name what is missing.
4. Never invent figures, thresholds, dates, document names or versions.
5. Quote thresholds, limits and timeframes verbatim.
6. Be concise (under 180 words unless the question demands detail); use short bullets for lists.
7. You assist bank staff with internal documentation; never give personal financial advice.
"""
 
 
def build_context_block(hits: list[RetrievalHit]) -> str:
    parts = []
    for n, hit in enumerate(hits, 1):
        header = f"[{n}] {hit.meta.get('title', hit.document_id)} — section: {hit.section} (document {hit.document_id})"
        parts.append(f"{header}\n{hit.text}\n")
    return "\n".join(parts)
 
 
def build_user_message(question: str, context: str, history: list[dict] | None = None) -> str:
    blocks = []
    if history:
        transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:])
        blocks.append(f"Conversation so far:\n{transcript}")
    blocks.append(f"Context passages:\n\n{context}")
    blocks.append(f"Question: {question}")
    return "\n\n".join(blocks)
 
 
def build_paraphrase_prompt(question: str) -> str:
    return (
        "Rewrite the question below as two alternative search queries that would retrieve "
        "the same policy documents. Keep product names and topic words. "
        "Return exactly two lines, no numbering, no commentary.\n\n"
        f"Question: {question}"
    )
