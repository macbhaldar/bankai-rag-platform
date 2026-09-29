"""Shared fixtures: a hermetic BankRAG environment (hash embeddings, extractive LLM)."""

from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Environment must be fixed BEFORE any app import triggers get_settings()
os.environ["BANKRAG_EMBEDDING_PROVIDER"] = "hash"
os.environ["BANKRAG_LLM_PROVIDER"] = "extractive"
os.environ["BANKRAG_MASK_PII"] = "true"
os.environ["BANKRAG_LOG_LEVEL"] = "WARNING"

import pytest

DOC_1 = """# Credit Policy v1.2
Effective Date: 2026-01-15
Domain: Credit
Jurisdiction: UK

## Purpose
This credit policy defines the underwriting standards for personal lending portfolios.

## Approval Authority
Credit limits above 50000 require dual approval from the regional credit committee.

## Exceptions
Exceptions require documented justification, risk assessment and an expiry review date.
"""

DOC_2 = """# Credit Policy v2.0
Effective Date: 2026-06-01
Domain: Credit
Jurisdiction: UK

## Purpose
This revised credit policy raises the automated approval threshold to 25000 for prime customers.

## Approval Authority
Credit limits above 75000 require dual approval from the central credit committee.
"""

DOC_3 = """# Sanctions Manual v3.1
Effective Date: 2025-11-02
Domain: Aml
Jurisdiction: Singapore

## Screening
All customers are screened against sanctions watchlists at onboarding and via a nightly batch run.

## Escalation
Potential sanctions matches must be frozen immediately and escalated to the compliance officer within 24 hours.

Product: wire transfers
Country: Singapore
Required controls: screening, escalation
"""

DOC_4 = """# Fraud Playbook v1.0
Effective Date: 2026-02-20
Domain: Aml
Jurisdiction: UAE

## Response
Confirmed card fraud requires the account to be blocked immediately and a replacement card issued within 2 business days.

## Monitoring
Fraud rules run in real time on the transaction stream and page the fraud desk on high severity alerts.
"""


def _write_corpus(corpus: Path) -> None:
    (corpus / "credit").mkdir(parents=True, exist_ok=True)
    (corpus / "aml").mkdir(parents=True, exist_ok=True)
    (corpus / "credit" / "00001_Credit_Policy_v1.2.md").write_text(DOC_1, encoding="utf-8")
    (corpus / "credit" / "00002_Credit_Policy_v2.0.md").write_text(DOC_2, encoding="utf-8")
    (corpus / "aml" / "00003_Sanctions_Manual_v3.1.md").write_text(DOC_3, encoding="utf-8")
    (corpus / "aml" / "00004_Fraud_Playbook_v1.0.md").write_text(DOC_4, encoding="utf-8")

def _write_structured(structured: Path) -> None:
    (structured / "roles.csv").write_text(
        "role,access_level\nAnalyst,1\nAuditor,6\n", encoding="utf-8")
    
    (structured / "users.csv").write_text(
        "user_id,role,department\nUSER0001,Analyst,credit\nUSER0002,Auditor,aml\n", encoding="utf-8")
    
    (structured / "document_acl.csv").write_text(
        "document_id,principal,permission\n"
        "00003_Sanctions_Manual_v3.1,USER0001,read\n",
        encoding="utf-8",)
    
    (structured / "customers.csv").write_text(
        "customer_id,segment,country,risk_band, age\n"
        "C1,SME,UK,Low,35\nC2,Retail,UAE,Medium,44\nC3,Retail,UK,High,61\n",
        encoding="utf-8",)
    
    (structured / "accounts.csv").write_text(
        "account_id,customer_id,account_type,currency,balance\n"
        "A1,C1,current,GBP,1500.50\nA2,C2,savings,AED,9200.00\n",
        encoding="utf-8",)
    
    (structured / "transactions.csv").write_text(
        "transaction_id,account_id,date,direction,currency,amount,channel,monitoring_status\n"
        "T1,A1,2026-01-05,CREDIT,GBP,100.00,Online,Clear\n"
        "T2,A2,2026-01-06,DEBIT,AED,900.00,ATM,Review\n"
        "T3,A1,2026-01-07,CREDIT,GBP,250.00,Branch,Clear\n"
        "T4,A2,2026-01-08,DEBIT,AED,5000.00,Online,Review\n",
        encoding="utf-8",)
    
    (structured / "loans.csv").write_text(
        "loan_id,customer_id,product,currency,principal,interest_rate,status,risk_band\n"
        "L1,C1,mortgage,GBP,200000.00,4.5,Performing,Low\n"
        "L2,C2,working capital,AED,50000.00,9.1,Watch,Medium\n",
        encoding="utf-8",)
    
    (structured / "securities.csv").write_text(
        "security_id,asset_class,currency,price,market_value,duration_years\n"
        "S1,equity,USD,120.00,120000.00,0\nS2,bond,USD,98.50,98500.00,7\n",
        encoding="utf-8",)
    
@pytest.fixture(scope="session")
def services(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bankrag")
    corpus = tmp / "corpus"
    structured = tmp / "structured"
    _write_corpus(corpus)
    _write_structured(structured)

    os.environ["BANKRAG_DATA_DIR"] = str(tmp / "data")
    os.environ["BANKRAG_STRUCTURED_DIR"] = str(structured)
    os.environ["BANKRAG_CORPUS_DIR"] = str(corpus)

    from app.core.config import reset_settings
    from app.core.container import build_services, reset_services

    reset_services()
    services = build_services()
    report = services.ingester.run(rebuild=True)
    assert report.errors == [], f"ingestion errors: {report.errors}"
    assert report.docs_ingested == 4
    services.acl.set_domains(services.ingester.domains())
    yield services
    reset_services()