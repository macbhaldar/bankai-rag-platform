"""Data explorer: natural-language questions answered with guarded SQL over the bank's tables."""

import pandas as pd
import streamlit as st
from app.ui import api_client

question = st.chat_input("Ask a data question, e.g. how many transactions are flagged for review by channel?")
examples = st.pills(
    "Examples",
    [
        "How many transactions are in review status per channel?",
        "Loans per risk band with average interest rate",
        "Customers per segment and country",
        "Market value per asset class",],
    label_visibility="collapsed",)
if examples and not question:
    question = examples

if question:
    try:
        with st.spinner("Generating and running SQL…"):
            payload = api_client.structured_query(question)
    except api_client.ApiError as exc:
        st.error(str(exc))
        st.stop()

    mode_badge = (
        ":green[text-to-SQL]" if payload.get("mode") == "text2sql" else ":orange[template fallback]")
    st.caption(f"{mode_badge} · {payload.get('row_count', 0)} rows"
               + (" · truncated" if payload.get("truncated") else ""))
    st.code(payload.get("sql", ""), language="sql")
    if payload.get("error"):
        st.error(payload["error"])
    if payload.get("columns"):
        st.dataframe(pd.DataFrame(payload["rows"], columns=payload["columns"]), width="stretch", hide_index=True)
    else:
        st.info("No rows returned.")
else:
    st.info(
        "Ask questions over the bank's structured tables (customers, accounts, transactions, loans, "
        "securities). With a generative LLM configured, questions are translated to guarded "
        "read-only SQL; without one, a deterministic template layer answers common intents.",
        icon=":material/table_chart:",)
    try:
        schema = api_client.structured_schema()
        with st.expander("Available tables", icon=":material/database:"):
            for table in schema["tables"]:
                cols = ", ".join(c["name"] for c in table["columns"])
                st.markdown(f"**{table['table']}** ({table['row_count']:,} rows)  \n{cols}")
    except api_client.ApiError:
        pass
    