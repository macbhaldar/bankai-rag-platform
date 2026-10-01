"""RAG chain tests: citations, grounding and the audit trail."""

from app.core.audit import AuditLog
 
 
def test_qa_returns_cited_extractive_answer(services):
    result = services.rag.ask("How should credit policy exceptions be handled?")
    assert result.mode == "extractive"
    assert result.citations, "extractive answers must cite at least one passage"
    assert result.answer
    assert 0.0 <= result.grounding <= 1.0
    assert result.citations[0].document_id.startswith("0000")
 
 
def test_qa_no_results_path(services):
    result = services.rag.ask("zzzqqq unrelated gibberish topic")
    assert result.warning in {"no_results", "low_confidence"} or result.answer
 
 
def test_qa_appends_audit_event(services, tmp_path):
    log_path = services.settings.paths.audit_dir / "audit_log.jsonl"
    audit = AuditLog(log_path, mask_pii=True)
    before = len(audit.tail(1000))
    services.rag.ask("What requires dual approval for credit limits?")
    after = len(audit.tail(1000))
    assert after >= before + 1
    last = audit.tail(1)[0]
    assert last["event"] == "qa"
    assert "question" in last
 
 
def test_qa_respects_acl(services):
    allowed = services.acl.allowed_document_ids("USER0001")
    result = services.rag.ask(
        "Confirmed card fraud requires the account to be blocked immediately",
        allowed_document_ids=allowed,)
    cited_docs = {c.document_id for c in result.citations}
    assert all(doc in allowed for doc in cited_docs)
    