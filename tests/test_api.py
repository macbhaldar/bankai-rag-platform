"""API integration tests over an in-process TestClient (offline providers)"""

import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client(services):
    from app.api.main import app
 
    with TestClient(app) as test_client:
        yield test_client

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents"] == 4
    assert body["llm_mode"] == "extractive"

def test_search_endpoint(client):
    response = client.post("/api/search", json={"query": "sanctions screening escalation", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    assert body["results"][0]["document_id"] == "00003_Sanctions_Manual_v3.1"
    assert {"dense_score", "sparse_score", "fused_score"} <= set(body["results"][0])

def test_search_with_principal(client):
    response = client.post("/api/search", json={"query": "fraud card blocked", "principal": "USER0001"})
    assert response.status_code == 200
    for result in response.json()["results"]:
        assert result["document_id"] != "00004_Fraud_Playbook_v1.0"

def test_qa_endpoint(client):
    response = client.post("/api/qa", json={"question": "How should exceptions be handled in the credit policy?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"]
    assert body["grounding"] >= 0.0

def test_documents_listing(client):
    response = client.get("/api/documents?limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert all("sha256" in doc for doc in body["documents"])

def test_structured_endpoint(client):
    response = client.post("/api/structured", json={"question": "How many transactions are in review status?"})
    assert response.status_code == 200
    body = response.json()
    assert body["rows"]
    assert body["mode"] == "fallback"

def test_structured_schema(client):
    response = client.get("/api/structured/schema")
    assert response.status_code == 200
    assert {t["table"] for t in response.json()["tables"]} >= {"customers", "transactions"}

def test_eval_qa_benchmark(client):
    response = client.post("/api/eval", json={"benchmark": "qa_test", "k": 5, "limit": 3})
    assert response.status_code == 400