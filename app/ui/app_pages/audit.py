"""Audit trail page: who asked what, with PII masking applied by the platform."""

import pandas as pd
import streamlit as st
from app.ui import api_client

limit = st.slider("Events", 10, 500, 50)
try:
    events = api_client.audit_tail(limit=limit)["events"]
except api_client.ApiError as exc:
    st.error(str(exc))
    st.stop()

if not events:
    st.info("No audit events yet — ask a question on the Assistant page first.", icon=":material/history:")
    st.stop()

rows = [
    {
        "time": event.get("ts", ""),
        "event": event.get("event", ""),
        "query": event.get("question") or event.get("query", ""),
        "mode": event.get("mode", ""),
        "latency_ms": event.get("latency_ms", ""),
        "grounding": event.get("grounding", ""),
        "top_documents": ", ".join(event.get("top_documents", [])[:2]),
    }
    for event in events
    ]
st.caption(
    "Every question and search is appended to a JSONL audit trail with PII masking "
    "(cards, emails, phone numbers, national IDs) applied before write.")

st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)