"""Document access control built on the dataset's ACL, users and roles tables."""

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)

BREAK_GLASS_LEVEL = 6

@dataclass
class UserContext:
    user_id: str
    role: str
    department: str
    access_level: int

class ACLService:
    def __init__(self, structured_dir: Path):
        self.loaded = False
        self.users: dict[str, UserContext] = {}
        self.explicit_grants: dict[str, set[str]] = {}
        self.doc_domains: dict[str, str] = {}
        self.n_acl_rows = 0
        self._cache: dict[str, set[str] | None] = {}
        try:
            self._load(structured_dir)
            self.loaded = True
        except FileNotFoundError as exc:
            logger.warning("ACL data unavailable (%s); retrieval will be unrestricted", exc)

    def _load(self, structured_dir: Path) -> None:
        roles: dict[str, int] = {}
        with (structured_dir / "roles.csv").open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                roles[row["role"]] = int(row["access_level"])
        with (structured_dir / "users.csv").open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                self.users[row["user_id"]] = UserContext(
                    user_id=row["user_id"],
                    role=row["role"],
                    department=row["department"],
                    access_level=roles.get(row["role"], 1),
                )
        with (structured_dir / "document_acl.csv").open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("permission", "read") == "read":
                    self.explicit_grants.setdefault(row["principal"], set()).add(row["document_id"])
                    self.n_acl_rows += 1
        logger.info("ACL loaded: %d users, %d grants", len(self.users), self.n_acl_rows)

    def set_domains(self, domains: dict[str, str]) -> None:
        self.doc_domains = {doc: (dom or "").lower() for doc, dom in domains.items()}
        self._cache.clear()

    def user(self, user_id: str) -> UserContext | None:
        return self.users.get(user_id)

    def list_users(self, limit: int = 500) -> list[dict]:
        return [
            {"user_id": u.user_id, "role": u.role, "department": u.department, "access_level": u.access_level}
            for u in list(self.users.values())[:limit]]

    def allowed_document_ids(self, user_id: str | None) -> set[str] | None:
        """None = unrestricted (no ACL feature / break-glass); set = the visible documents."""
        if not self.loaded or not user_id:
            return None
        if user_id in self._cache:
            return self._cache[user_id]
        user = self.users.get(user_id)
        if user is None:
            self._cache[user_id] = set()
            return set()
        if user.access_level >= BREAK_GLASS_LEVEL:
            self._cache[user_id] = None
            return None
        allowed = set(self.explicit_grants.get(user_id, set()))
        if user.department:
            allowed |= {doc for doc, dom in self.doc_domains.items() if dom == user.department.lower()}
        self._cache[user_id] = allowed
        return allowed

    def can_read(self, user_id: str | None, document_id: str) -> bool:
        allowed = self.allowed_document_ids(user_id)
        return allowed is None or document_id in allowed

    def stats(self) -> dict:
        return {
            "acl_loaded": self.loaded,
            "users": len(self.users),
            "explicit_grants": self.n_acl_rows,
            "domains_known": len(self.doc_domains),
            "break_glass_level": BREAK_GLASS_LEVEL,
            }