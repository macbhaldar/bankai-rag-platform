"""End-to-end demo: semantic search, RAG QA with citations, structured text-to-SQL.

Usage: python scripts/demo_queries.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.container import get_services

DEMO_QUESTIONS = [
    "What records must be retained for investment portfolio?",
    "How should exceptions for trade finance be handled?",
    "When should issues involving foreign exchange be escalated?",]

SEARCH_QUERIES = [
    "sanctions screening escalation",
    "collateral valuation controls",]

STRUCTURED_QUESTIONS = [
    "How many transactions are flagged for review in each channel?",
    "Show the number of loans per risk band.",]

def main() -> int:
    services = get_services()
    print("=" * 78)
    print("BankRAG demo — embedding:", services.embeddings.name)
    print("LLM mode:", f"{services.llm.provider}:{services.llm.model} ({services.llm.mode})")
    print("=" * 78)

    print("\n--- Semantic search ---")
    for query in SEARCH_QUERIES:
        hits = services.retriever.retrieve(query, k=3, dedupe_families=True)
        print(f"\nQ: {query}")
        for rank, hit in enumerate(hits, 1):
            print(
                f"  {rank}. [{hit.document_id}] {hit.section} "
                f"(dense={hit.dense_score:.2f} sparse={hit.sparse_score:.2f})")
            print(f"     {hit.snippet(140)}")

    print("\n--- RAG QA ---")
    for question in DEMO_QUESTIONS:
        result = services.rag.ask(question)
        print(f"\nQ: {question}")
        print(f"A: {result.answer}")
        if result.citations:
            print("Sources: " + ", ".join(f"[{c.n}] {c.document_id} ({c.section})" for c in result.citations))
        print(f"   mode={result.mode} grounding={result.grounding:.0%} latency={result.timings_ms.get('total_ms')}ms")

    if services.structured_engine:
        print("\n--- Structured data (text-to-SQL) ---")
        from app.structured.text2sql import answer_structured

        for question in STRUCTURED_QUESTIONS:
            payload = answer_structured(services.structured_engine, services.llm, question)
            print(f"\nQ: {question}")
            print(f"   SQL: {' '.join(payload['sql'].split())}")
            for row in payload["rows"][:5]:
                print(f"   {row}")
            if payload.get("error"):
                print(f"   error: {payload['error']}")

    print("\n--- Audit trail (last 3 entries) ---")
    for row in services.audit.tail(3):
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())