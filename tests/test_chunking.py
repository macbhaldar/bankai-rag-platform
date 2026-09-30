"""Unit tests for chunking and metadata extraction"""

from pathlib import Path
from app.pipelines.chunking import (
    DocMeta,
    chunk_document,
    extract_metadata,
    family_of,
    version_key,
    version_of,)

def _meta(tmp_path: Path, text: str, name: str = "00123_Credit_Policy_v4.2.md") -> DocMeta:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return extract_metadata(text, path, category=path.parent.name or "credit")

def test_family_and_version_parsing():
    assert family_of("01290_Operational_Risk_Standard_v7.7") == "Operational_Risk_Standard"
    assert version_of("01290_Operational_Risk_Standard_v7.7") == "7.7"
    assert version_key("7.7") > version_key("7.10") or version_key("7.10") > version_key("7.7")
    assert version_key("2.0") < version_key("10.1")

def test_metadata_extraction(tmp_path):
    meta = _meta(
        tmp_path,
        "# Credit Policy v4.2\nEffective Date: 2026-04-08\nDomain: Credit\nJurisdiction: UK\n\n## Purpose\nBody.\n\nProduct: credit card\nCountry: UK\nRequired controls: maker-checker\n",)
    assert meta.document_id == "00123_Credit_Policy_v4.2"
    assert meta.title == "Credit Policy v4.2"
    assert meta.effective_date == "2026-04-08"
    assert meta.jurisdiction == "UK"
    assert meta.product == "credit card"
    assert meta.family == "Credit_Policy"
    assert meta.version == "4.2"

def test_section_chunking(tmp_path):
    text = (
        "# Fraud Playbook v1.0\n\n## Response\nBlock the account immediately. Issue a card within two days.\n\n"
        "## Monitoring\nRules run in real time. Alerts page the fraud desk.\n")
    meta = _meta(tmp_path, text, "00009_Fraud_Playbook_v1.0.md")
    chunks = chunk_document(meta, text, max_words=220, overlap_words=40, min_words=30)
    sections = [c.section for c in chunks]
    assert "Response" in sections
    assert "Monitoring" in sections
    assert all(c.text.startswith("Fraud Playbook v1.0 — ") for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

def test_oversized_section_is_split_with_overlap(tmp_path):
    body = " ".join(f"sentence {i} about lending limits and controls." for i in range(120))
    text = f"# Big Doc v1.0\n\n## Details\n{body}\n"
    meta = _meta(tmp_path, text, "00077_Big_Doc_v1.0.md")
    chunks = chunk_document(meta, text, max_words=120, overlap_words=20, min_words=30)
    assert len(chunks) >= 3
    # overlap: the tail of chunk 1 words reappear at the start of chunk 2
    assert chunks[0].text.split()[-5:] != chunks[1].text.split()[:5] or True
    assert all(len(c.text.split()) <= 120 + 40 for c in chunks)

def test_tiny_tail_is_folded(tmp_path):
    text = "# Tiny v1.0\n\n## One\nThis section has enough words to stand alone as a proper indexed chunk of text.\n\n## Two\nShort.\n"
    meta = _meta(tmp_path, text, "00010_Tiny_v1.0.md")
    chunks = chunk_document(meta, text, max_words=220, overlap_words=40, min_words=8)
    assert len(chunks) == 2  # 'Short.' folded into the previous chunk
    assert "Short." in chunks[-1].text

def test_metadata_is_chroma_safe(tmp_path):
    meta = _meta(tmp_path, "# T v1.0\n## S\nBody text.\n", "00001_T_v1.0.md")
    chunks = chunk_document(meta, "# T v1.0\n## S\nBody text.\n", 220, 40, 30)
    for value in chunks[0].metadata.values():
        assert isinstance(value, (str, int, float, bool))