"""PII masking and detection tests."""

from app.core.pii import luhn_valid, mask_deep, mask_text, scan_text

def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("12345")

def test_card_masking_keeps_last_four():
    masked = mask_text("card 4111 1111 1111 1111 on file")
    assert "4111" not in masked
    assert "[CARD-1111]" in masked

def test_email_and_ssn_masking():
    masked = mask_text("contact jane.doe@bank.com or SSN 123-45-6789")
    assert "jane.doe@bank.com" not in masked
    assert "123-45-6789" not in masked
    assert "[EMAIL]" in masked and "[SSN]" in masked

def test_non_card_numbers_survive():
    text = "threshold 50000 and date 2026-04-08 remain untouched"
    assert mask_text(text) == text

def test_scan_text_counts():
    findings = scan_text("email a@b.com and card 4111111111111111")
    assert findings["email"] == 1
    assert findings["card"] == 1

def test_mask_deep_recurses():
    payload = {"question": "mail me at x@y.com", "nested": ["4111111111111111"], "n": 3}
    masked = mask_deep(payload)
    assert "[EMAIL]" in masked["question"]
    assert "[CARD-1111]" in masked["nested"][0]
    assert masked["n"] == 3
