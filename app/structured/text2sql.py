"""Natural-language questions -> guarded SQL over the structured tables."""

from __future__ import annotations
import re
from app.core.logging import get_logger
from app.generation.llm import BaseLLM, LLMError
from app.structured.engine import SQLRejected, StructuredEngine

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a senior analytics engineer for a bank.
Given a SQLite schema, write ONE read-only SELECT statement that answers the user's question.
Rules:
- Use only the tables and columns listed in the schema (SQLite dialect).
- Never modify data: SELECT only, no CTEs, no PRAGMA.
- Aggregate where the question asks for counts/averages; ORDER BY meaningfully.
- Always end with LIMIT 200 or less.
- Reply with a single ```sql fenced block and nothing else.
"""

def _schema_prompt(schema: list[dict]) -> str:
    lines = []
    for table in schema:
        cols = ", ".join(f"{c['name']} {c['type']}" for c in table["columns"])
        lines.append(f"- {table['table']} ({table['row_count']} rows): {cols}")
    return "\n".join(lines)

_FALLBACK_TEMPLATES: list[tuple[str, str]] = [
    (r"monitor|flag|review|status", "SELECT monitoring_status, COUNT(*) AS n FROM transactions GROUP BY monitoring_status ORDER BY n DESC"),
    (r"channel", "SELECT channel, COUNT(*) AS n, ROUND(SUM(amount), 2) AS total_amount FROM transactions GROUP BY channel ORDER BY n DESC"),
    (r"loan.*(risk|risk_band)|risk.*loan", "SELECT risk_band, COUNT(*) AS n, ROUND(AVG(interest_rate), 2) AS avg_rate FROM loans GROUP BY risk_band ORDER BY n DESC"),
    (r"loan|lending", "SELECT status, COUNT(*) AS n, ROUND(SUM(principal), 2) AS total_principal FROM loans GROUP BY status ORDER BY n DESC"),
    (r"segment|customer", "SELECT segment, COUNT(*) AS n FROM customers GROUP BY segment ORDER BY n DESC"),
    (r"country|jurisdiction|region", "SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC"),
    (r"securit|asset", "SELECT asset_class, COUNT(*) AS n, ROUND(SUM(market_value), 2) AS total_market_value FROM securities GROUP BY asset_class ORDER BY n DESC"),
    (r"account|balance", "SELECT account_type, COUNT(*) AS n, ROUND(SUM(balance), 2) AS total_balance FROM accounts GROUP BY account_type ORDER BY n DESC"),
    (r"role|access", "SELECT role, COUNT(*) AS n FROM users GROUP BY role ORDER BY n DESC"),
]


def _fallback_sql(question: str) -> str:
    lowered = question.lower()
    for pattern, sql in _FALLBACK_TEMPLATES:
        if re.search(pattern, lowered):
            return sql
    return "SELECT 'customers' AS table_name, COUNT(*) AS rows FROM customers " \
           "UNION ALL SELECT 'accounts', COUNT(*) FROM accounts " \
           "UNION ALL SELECT 'transactions', COUNT(*) FROM transactions " \
           "UNION ALL SELECT 'loans', COUNT(*) FROM loans " \
           "UNION ALL SELECT 'securities', COUNT(*) FROM securities"


def _extract_sql(text: str) -> str:
    match = re.search(r"```sql\s*(.+?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"((?is)^\s*SELECT\b.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else text.strip()

def answer_structured(engine: StructuredEngine, llm: BaseLLM, question: str) -> dict:
    schema = engine.schema()
    sql = ""
    mode = "fallback"
    error = None
    if llm.mode == "generative":
        try:
            result = llm.generate(
                question,
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Schema:\n{_schema_prompt(schema)}\n\nQuestion: {question}"},
                ],
                hits=[],
                )
            sql = _extract_sql(result.text)
            mode = "text2sql"
        except LLMError as exc:
            logger.warning("text2sql failed: %s", exc)
            error = str(exc)
    if not sql:
        sql = _fallback_sql(question)
    try:
        payload = engine.run_sql(sql)
        payload.update({"mode": mode, "question": question, "error": error})
        return payload
    except SQLRejected as exc:
        return {
            "question": question,
            "sql": sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "mode": mode,
            "error": str(exc),
            }