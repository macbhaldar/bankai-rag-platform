"""PII detection and masking used by audit logs and ingestion reports.
 
Bank data must never leak card numbers, national IDs or contact details into
logs. Masking keeps the last 4 digits of card numbers for traceability.
"""
from __future__ import annotations
 
import re
from typing import Any
 
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b")
 
 
def luhn_valid(number: str) -> bool:
    digits = re.sub(r"\D", "", number)
    if not 13 <= len(digits) <= 19:
        return False
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0
 
 
def mask_text(text: str) -> str:
    """Return `text` with PII replaced by masked placeholders."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = SSN_RE.sub("[SSN]", text)
    text = IBAN_RE.sub("[IBAN]", text)
 
    def _card(m: re.Match) -> str:
        raw = m.group(0)
        if luhn_valid(raw):
            digits = re.sub(r"\D", "", raw)
            return f"[CARD-{digits[-4:]}]"
        return raw
 
    text = CARD_RE.sub(_card, text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text
 
 
def scan_text(text: str) -> dict[str, int]:
    """Count PII findings by type (used by the ingestion enrichment report)."""
    findings: dict[str, int] = {}
    for name, pattern in (
        ("email", EMAIL_RE),
        ("ssn", SSN_RE),
        ("iban", IBAN_RE),
        ("card", CARD_RE),
        ("phone", PHONE_RE),
    ):
        hits = pattern.findall(text)
        if name == "card":
            hits = [h for h in hits if luhn_valid(h)]
        if hits:
            findings[name] = len(hits)
    return findings
 
 
def mask_deep(value: Any) -> Any:
    """Recursively mask strings inside dicts/lists (for structured log payloads)."""
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, dict):
        return {k: mask_deep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_deep(v) for v in value]
    return value