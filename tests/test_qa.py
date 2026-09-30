"""RAG chain tests: citations, grounding and the audit trail."""
from app.core.audit import AuditLog
 
 
def test_qa_returns_cited_extractive_answer(services):
    