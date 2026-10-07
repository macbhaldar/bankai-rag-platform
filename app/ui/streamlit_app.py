"""BankRAG control center — Streamlit UI for the Bank Intelligence RAG Platform"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.ui import api_client

st.set_page_config(
    page_title="BankRAG — Bank Intelligence RAG Platform",
    page_icon=":material/account_balance:",
    layout="wide",)

# shared state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "principal" not in st.session_state:
    st.session_state.principal = None

@st.cache_data(ttl=10)
def cached_health() -> dict | None:
    try:
        return api_client.health()
    except Exception:
        return None

@st.cache_data(ttl=300)
def cached_users() -> list[dict]:
    try:
        return api_client.security_users(limit=100)["users"]
    except Exception:
        return []

health = cached_health()

# sidebar
with st.sidebar:
    st.markdown(
        f"### :material/account_balance: BankRAG\n"
        f":small[Bank Intelligence RAG Platform]")
    if health:
        st.markdown(
            f":material/check_circle: **API online** — {health['documents']} documents, "
            f"{health['chunks']} chunks")
        st.caption(
            f"embeddings: `{health['embedding_provider']}` · "
            f"LLM: `{health['llm_provider']}:{health['llm_model']}` ({health['llm_mode']})")
    else:
        st.error("API offline — start it with `scripts/dev.sh` (or `scripts/dev.bat`).")

    users = cached_users()
    if users:
        labels = {u["user_id"]: f"{u['user_id']} — {u['role']} / {u['department']}" for u in users}
        selection = st.selectbox(
            "Acting as (ACL scoping)",
            ["(unrestricted)"] + list(labels),
            format_func=lambda value: "Unrestricted — full corpus" if value == "(unrestricted)" else labels[value],)
        st.session_state.principal = None if selection == "(unrestricted)" else selection

# navigation
page = st.navigation(
    {
        "Knowledge": [
            st.Page("app_pages/chat.py", title="Assistant", icon=":material/forum:", default=True),
            st.Page("app_pages/search.py", title="Semantic search", icon=":material/search:"),
            st.Page("app_pages/documents.py", title="Documents", icon=":material/library_books:"),
            ],
        "Data": [
            st.Page("app_pages/data_explorer.py", title="Data explorer", icon=":material/table_chart:"),
            st.Page("app_pages/ingestion.py", title="Ingestion", icon=":material/upload_file:"),
            ],
        "Governance": [
            st.Page("app_pages/evaluation.py", title="Evaluation", icon=":material/analytics:"),
            st.Page("app_pages/audit.py", title="Audit trail", icon=":material/history:"),
            ],
        },
    position="sidebar",)
st.title(page.title, icon=page.icon)
page.run()
