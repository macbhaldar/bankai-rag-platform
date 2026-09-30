"""Hybrid retrieval, ACL scoping and version-dedup tests"""


def test_relevant_document_ranks_first(services):
    hits = services.retriever.retrieve("sanctions matches must be escalated to compliance officer", k=3)
    assert hits, "expected retrieval results"
    assert hits[0].document_id == "00003_Sanctions_Manual_v3.1"

def test_filters_restrict_domain(services):
    hits = services.retriever.retrieve("approval authority", k=10, filters={"category": "credit"})
    assert hits
    assert all(h.meta["category"] == "credit" for h in hits)

def test_acl_scoping_limits_visibility(services):
    # USER0001 is an Analyst in department 'credit' + one explicit aml grant
    allowed = services.acl.allowed_document_ids("USER0001")
    assert allowed is not None
    assert "00003_Sanctions_Manual_v3.1" in allowed
    assert "00004_Fraud_Playbook_v1.0" not in allowed
    hits = services.retriever.retrieve("fraud card blocked immediately", k=10, allowed_document_ids=allowed)
    assert all(h.document_id in allowed for h in hits)

def test_break_glass_role_is_unrestricted(services):
    assert services.acl.allowed_document_ids("USER0002") is None

def test_unknown_principal_is_denied(services):
    assert services.acl.allowed_document_ids("USER9999") == set()

def test_version_dedupe_prefers_newest(services):
    hits = services.retriever.retrieve("credit policy approval authority", k=10, dedupe_families=True)
    families = [h.family for h in hits]
    assert "Credit_Policy" in families
    credit_hits = [h for h in hits if h.family == "Credit_Policy"]
    assert len(credit_hits) == 1
    assert credit_hits[0].version == "2.0"

