"""Pydantic request/response models for the API."""

from __future__ import annotations
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50)
    category: str | None = None
    jurisdiction: str | None = None
    family: str | None = None
    document_id: str | None = None
    principal: str | None = Field(default=None, description="user_id for ACL-scoped retrieval")
    dedupe_families: bool = Field(default=True, description="keep only the newest version per document family")

class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)
    category: str | None = None
    jurisdiction: str | None = None
    history: list[dict] = Field(default_factory=list, max_length=12)
    principal: str | None = None
    dedupe_families: bool = True

class IngestRequest(BaseModel):
    source_dir: str | None = None
    rebuild: bool = False
    dry_run: bool = False
    limit: int | None = Field(default=None, ge=1, le=10000)

class EvalRequest(BaseModel):
    benchmark: str = Field(default="retrieval", description="retrieval | qa_test | qa_validation | qa_train | hard_negatives")
    k: int = Field(default=10, ge=1, le=100)
    limit: int | None = Field(default=None, ge=1, le=1000)

class StructuredRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

class DeleteDocumentResponse(BaseModel):
    deleted: bool
    document_id: str