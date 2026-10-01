"""Structured data engine and text-to-SQL fallback tests"""

import pytest
from app.structured.engine import SQLRejected
from app.structured.text2sql import _fallback_sql, answer_structured

def test_run_sql_select(services):
    payload = services.structured_engine.run_sql("SELECT monitoring_status, COUNT(*) AS n FROM transactions GROUP BY monitoring_status")
    assert payload["row_count"] > 0
    assert payload["columns"][0] == "monitoring_status"

def test_run_sql_rejects_writes(services):
    for bad in [
        "INSERT INTO customers VALUES (1)",
        "DELETE FROM customers",
        "DROP TABLE customers",
        "PRAGMA table_info(customers)",
        "SELECT 1; SELECT 2",
        ]:
        with pytest.raises(SQLRejected):
            services.structured_engine.run_sql(bad)

def test_run_sql_appends_limit(services):
    payload = services.structured_engine.run_sql("SELECT * FROM transactions")
    assert "LIMIT" in payload["sql"].upper()
    assert payload["row_count"] <= services.settings.structured.max_rows

def test_fallback_template_matches_intent(services):
    assert "monitoring_status" in _fallback_sql("how many transactions are flagged for review?")
    assert "risk_band" in _fallback_sql("loans by risk band")

def test_answer_structured_offline_mode(services):
    payload = answer_structured(services.structured_engine, services.llm, "loans per risk band")
    assert payload["mode"] == "fallback"
    assert payload["rows"]
    assert payload["error"] is None

def test_schema_introspection(services):
    tables = {t["table"] for t in services.structured_engine.schema()}
    assert {"customers", "transactions", "loans"} <= tables