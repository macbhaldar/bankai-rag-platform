#Thin HTTP client the Streamlit UI uses to talk to the FastAPI backend.

from __future__ import annotations
import os
from pathlib import Path
import requests

DEFAULT_TIMEOUT = 120

def api_base() -> str:
    return os.environ.get("BANKRAG_API_URL", "http://127.0.0.1:8000").rstrip("/")

class ApiError(RuntimeError):
    pass

def _get(path: str, params: dict | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    try:
        response = requests.get(f"{api_base()}{path}", params=params, timeout=timeout)
    except requests.ConnectionError as exc:
        raise ApiError(f"API unreachable at {api_base()} — start it with scripts/dev.sh") from exc
    response.raise_for_status()
    return response.json()

def _post(path: str, json: dict | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    try:
        response = requests.post(f"{api_base()}{path}", json=json, timeout=timeout)
    except requests.ConnectionError as exc:
        raise ApiError(f"API unreachable at {api_base()} — start it with scripts/dev.sh") from exc
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.headers.get("content-type", "").startswith("application/json") else response.text
        raise ApiError(f"API error {response.status_code}: {detail}")
    return response.json()

def health() -> dict:
    return _get("/health", timeout=5)

def stats(principal: str | None = None) -> dict:
    params = {"principal": principal} if principal else None
    return _get("/api/stats", params=params)

def meta_filters() -> dict:
    return _get("/api/meta/filters")

def search(query: str, k: int | None = None, category: str | None = None,
           jurisdiction: str | None = None, principal: str | None = None,
           dedupe_families: bool = True) -> dict:
    return _post("/api/search", {
        "query": query, "k": k, "category": category, "jurisdiction": jurisdiction,
        "principal": principal, "dedupe_families": dedupe_families,})

def ask(question: str, k: int | None = None, category: str | None = None,
        principal: str | None = None, history: list[dict] | None = None) -> dict:
    return _post("/api/qa", {
        "question": question, "k": k, "category": category,
        "principal": principal, "history": history or [],})

def ingest(source_dir: str | None = None, rebuild: bool = False, dry_run: bool = False,
           limit: int | None = None) -> dict:
    return _post("/api/ingest", {
        "source_dir": source_dir, "rebuild": rebuild, "dry_run": dry_run, "limit": limit,})

def ingest_upload(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            response = requests.post(
                f"{api_base()}/api/ingest/upload",
                files={"file": (path.name, fh)},
                timeout=DEFAULT_TIMEOUT,)
    except requests.ConnectionError as exc:
        raise ApiError(f"API unreachable at {api_base()}") from exc
    response.raise_for_status()
    return response.json()

def documents(limit: int = 100, offset: int = 0, domain: str | None = None,
              principal: str | None = None) -> dict:
    params: dict = {"limit": limit, "offset": offset}
    if domain:
        params["domain"] = domain
    if principal:
        params["principal"] = principal
    return _get("/api/documents", params=params)

def delete_document(document_id: str) -> dict:
    try:
        response = requests.delete(f"{api_base()}/api/documents/{document_id}", timeout=60)
    except requests.ConnectionError as exc:
        raise ApiError(f"API unreachable at {api_base()}") from exc
    response.raise_for_status()
    return response.json()

def run_eval(benchmark: str, k: int = 10, limit: int | None = None) -> dict:
    return _post("/api/eval", {"benchmark": benchmark, "k": k, "limit": limit}, timeout=600)

def security_users(limit: int = 100) -> dict:
    return _get("/api/security/users", params={"limit": limit})

def audit_tail(limit: int = 50) -> dict:
    return _get("/api/audit", params={"limit": limit})

def structured_query(question: str) -> dict:
    return _post("/api/structured", {"question": question}, timeout=180)

def structured_schema() -> dict:
    return _get("/api/structured/schema", timeout=60)

