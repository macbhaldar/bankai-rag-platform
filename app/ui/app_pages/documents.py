"""Documents page: browse the indexed corpus and manage documents."""

import streamlit as st
from app.ui import api_client

try:
    filters = api_client.meta_filters()
except api_client.ApiError as exc:
    st.error(str(exc))
    st.stop()

with st.container(horizontal=True):
    domain = st.selectbox("Domain", ["(all)"] + filters["domains"], key="docs_domain")
    limit = st.slider("Page size", 10, 200, 50)

try:
    payload = api_client.documents(
        limit=limit,
        domain=None if domain == "(all)" else domain,
        principal=st.session_state.principal,)
except api_client.ApiError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"{payload['total']} documents visible"
           + (" — scoped by the acting-as principal" if st.session_state.principal else ""))
rows = [
    {
        "document": doc["document_id"],
        "title": doc["title"],
        "domain": doc["domain"],
        "family": doc["family"],
        "version": doc["version"],
        "jurisdiction": doc.get("jurisdiction", ""),
        "chunks": doc["n_chunks"],
        "ingested": doc.get("ingested_at", "")[:19],
    }
    for doc in payload["documents"]]
st.dataframe(
    rows,
    width="stretch",
    hide_index=True,)

with st.expander("Remove a document from the index", icon=":material/delete:"):
    document_id = st.selectbox("Document", ["(select)"] + [d["document_id"] for d in payload["documents"]])
    confirm = st.checkbox("I understand this permanently removes the document from the search index")
    if st.button("Delete document", type="primary", disabled=not confirm or document_id == "(select)"):
        try:
            api_client.delete_document(document_id)
            st.success(f"Deleted {document_id}")
            st.cache_data.clear()
        except api_client.ApiError as exc:
            st.error(str(exc))