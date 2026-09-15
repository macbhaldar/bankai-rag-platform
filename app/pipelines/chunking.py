"""Section-aware chunking with bank-specific metadata enrichment."""

from __future__ import annotations
 
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
 
DOC_ID_RE = re.compile(r"^(?P<ordinal>\d+)_(?P<family>.+?)_v(?P<version>\d+(?:\.\d+)*)$")
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
 
 
def family_of(document_id: str) -> str:
    """'01290_Operational_Risk_Standard_v7.7' -> 'Operational_Risk_Standard'."""
    m = DOC_ID_RE.match(document_id)
    return m.group("family") if m else document_id
 
 
def version_of(document_id: str) -> str:
    m = DOC_ID_RE.match(document_id)
    return m.group("version") if m else "0"
 
 
def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))
 
 
@dataclass
class DocMeta:
    document_id: str
    filename: str
    title: str
    category: str
    domain: str
    jurisdiction: str = ""
    effective_date: str = ""
    product: str = ""
    country: str = ""
    required_controls: str = ""
    family: str = ""
    version: str = "0"
    ordinal: str = ""
    source_path: str = ""
    sha256: str = ""
    n_chars: int = 0
    pii_findings: dict = field(default_factory=dict)
 
    def to_metadata(self, chunk_index: int, section: str, n_words: int) -> dict:
        """Flat metadata dict safe for Chroma (str/int/float/bool only)."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "title": self.title,
            "category": self.category,
            "domain": self.domain,
            "family": self.family,
            "version": self.version,
            "jurisdiction": self.jurisdiction,
            "effective_date": self.effective_date,
            "product": self.product,
            "country": self.country,
            "section": section,
            "chunk_index": chunk_index,
            "n_words": n_words,
            "source_path": self.source_path,
        }
 
 
@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    section: str
    chunk_index: int
    metadata: dict
 
 
def _first_match(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else ""
 
 
def extract_metadata(text: str, path: Path, category: str) -> DocMeta:
    title = _first_match(text, r"^#\s+(.+?)$") or path.stem.replace("_", " ")
    return DocMeta(
        document_id=path.stem,
        filename=path.name,
        title=title,
        category=category,
        domain=category,
        jurisdiction=_first_match(text, r"^Jurisdiction:\s*(.+?)$"),
        effective_date=_first_match(text, r"^Effective Date:\s*(.+?)$"),
        product=_first_match(text, r"^Product:\s*(.+?)$"),
        country=_first_match(text, r"^Country:\s*(.+?)$"),
        required_controls=_first_match(text, r"^Required controls:\s*(.+?)$"),
        family=family_of(path.stem),
        version=version_of(path.stem),
        ordinal=(DOC_ID_RE.match(path.stem).group("ordinal") if DOC_ID_RE.match(path.stem) else ""),
        source_path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        n_chars=len(text),
    )
 
 
def _split_long_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Recursive split on sentence, then word boundaries, with word overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    units: list[list[str]] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if count + len(words) > max_words and current:
            units.append(current)
            tail = sum((s.split() for s in current[-2:]), [])[-overlap_words:]
            current = [" ".join(tail)] if tail else []
            count = len(current[0].split()) if current else 0
            if len(words) > max_words:
                for i in range(0, len(words), max_words):
                    units.append(words[i : i + max_words])
                current, count = [], 0
                continue
        current.append(sentence)
        count += len(words)
    if current:
        units.append(current)
    return [" ".join(u) for u in units if " ".join(u).strip()]
 
 
def _sections(text: str) -> list[tuple[str, str]]:
    """Split on ## headings; preamble -> 'Overview'; trailing footer -> 'Document Attributes'."""
    matches = list(HEADING_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    if not matches:
        return [("Overview", text.strip())]
    preamble = text[: matches[0].start()].strip()
    preamble = "\n".join(
        line for line in preamble.splitlines() if not line.startswith("#")
    ).strip()
    if preamble:
        sections.append(("Overview", preamble))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((m.group(1).strip(), body))
    return sections
 
 
def chunk_document(meta: DocMeta, text: str, max_words: int, overlap_words: int, min_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for section, body in _sections(text):
        pieces = [body] if len(body.split()) <= max_words else _split_long_text(body, max_words, overlap_words)
        for piece in pieces:
            words = len(piece.split())
            if words < min_words and chunks:
                # fold tiny tail pieces into the previous chunk instead of indexing noise
                prev = chunks[-1]
                prev.text = f"{prev.text}\n{piece}"
                continue
            enriched = f"{meta.title} — {section}\n{piece}"
            chunks.append(
                Chunk(
                    id=f"{meta.document_id}::{index:03d}",
                    document_id=meta.document_id,
                    text=enriched,
                    section=section,
                    chunk_index=index,
                    metadata=meta.to_metadata(index, section, words),
                )
            )
            index += 1
    if not chunks:  # pathological empty doc: keep a stub so the doc is not silently lost
        chunks.append(
            Chunk(
                id=f"{meta.document_id}::000",
                document_id=meta.document_id,
                text=meta.title,
                section="Overview",
                chunk_index=0,
                metadata=meta.to_metadata(0, "Overview", 0),
            )
        )
    return chunks