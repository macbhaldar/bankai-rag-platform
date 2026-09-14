"""Append-only JSONL audit trail for queries and administrative actions."""
from __future__ import annotations
 
import json
from datetime import datetime, timezone
from pathlib import Path
 
from app.core.logging import get_logger
from app.core.pii import mask_deep
 
logger = get_logger(__name__)
 
 
class AuditLog:
    def __init__(self, path: Path, mask_pii: bool = True):
        self.path = path
        self.mask_pii = mask_pii
        path.parent.mkdir(parents=True, exist_ok=True)
 
    def append(self, event: str, **fields: object) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            **fields,
        }
        if self.mask_pii:
            row = mask_deep(row)
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:  # audit must never break the request path
            logger.warning("audit write failed: %s", exc)
 
    def tail(self, limit: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        rows: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows[-limit:]