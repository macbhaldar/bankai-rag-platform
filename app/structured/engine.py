"""SQLite-in-memory engine over the structured CSVs, with strict read-only guardrails."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
import pandas as pd
from app.core.logging import get_logger

logger = get_logger(__name__)

TABLES = ["customers", "accounts", "transactions", "loans", "securities", "users", "roles", "document_acl"]

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|GRANT|REVOKE|VACUUM|REPLACE)\b",
    re.IGNORECASE,)

class SQLRejected(ValueError):
    pass

class StructuredEngine:
    def __init__(self, structured_dir: Path, max_rows: int = 200):
        self.structured_dir = structured_dir
        self.max_rows = max_rows
        self._conn: sqlite3.Connection | None = None

    def _ensure_loaded(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            for table in TABLES:
                path = self.structured_dir / f"{table}.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                    frame.to_sql(table, conn, index=False, if_exists="replace")
                    logger.info("structured table loaded: %s (%d rows)", table, len(frame))
            self._conn = conn
        return self._conn

    def loaded(self) -> bool:
        return self._conn is not None

    def schema(self) -> list[dict]:
        conn = self._ensure_loaded()
        out = []
        for table in TABLES:
            try:
                info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            except sqlite3.Error:
                continue
            if not info:
                continue
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            out.append(
                {   "table": table,
                    "row_count": count,
                    "columns": [{"name": col[1], "type": col[2]} for col in info],}
                )
        return out

    def run_sql(self, sql: str) -> dict:
        """Validate and execute a read-only SELECT, capped at max_rows."""
        cleaned = sql.strip().rstrip(";").strip()
        if not cleaned:
            raise SQLRejected("empty SQL statement")
        if ";" in cleaned:
            raise SQLRejected("only a single statement is allowed")
        if not re.match(r"(?is)^\s*(SELECT)\b", cleaned):
            raise SQLRejected("only SELECT statements are allowed")
        if _FORBIDDEN.search(cleaned):
            raise SQLRejected("statement contains a forbidden keyword")
        if not re.search(r"(?is)\bLIMIT\s+\d+\s*$", cleaned):
            cleaned = f"{cleaned}\nLIMIT {self.max_rows}"

        conn = self._ensure_loaded()
        try:
            cursor = conn.execute(cleaned)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            raise SQLRejected(f"SQLite error: {exc}") from exc
        truncated = len(rows) > self.max_rows
        return {
            "sql": cleaned,
            "columns": columns,
            "rows": [list(row) for row in rows[: self.max_rows]],
            "row_count": len(rows[: self.max_rows]),
            "truncated": truncated,
            }