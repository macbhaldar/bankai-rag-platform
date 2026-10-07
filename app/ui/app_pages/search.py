"""Semantic search page: hybrid retrieval with score transparency."""

import streamlit as st
from app.ui import api_client

try:
    meta = api_client.meta_filters()
    domain_options = ["(all)"] + meta["domains"]
except api_client.ApiError:
    domain_options = ["(all)"]

with st.container(horizontal=True):
    query = st.text_input("Query", placeholder="e.g. sanctions screening escalation", label_visibility="collapsed")
    domain = st.selectbox("Domain", domain_options, key="search_domain")
    k = st.slider("Results", 3, 20, 8)
    dedupe = st.toggle("Newest version only", value=True)

if query:
    try:
        with st.spinner("Searching…"):
            payload = api_client.search(
                query,
                k=k,
                category=None if domain == "(all)" else domain,
                principal=st.session_state.principal,
                dedupe_families=dedupe,)
    except api_client.ApiError as exc:
        st.error(str(exc))
        st.stop()

    st.caption(
        f"{len(payload['results'])} results · {payload['latency_ms']:.0f} ms"
        + (f" · ACL-scoped to `{payload['scoped_to_principal']}`" if payload.get("scoped_to_principal") else ""))
    for rank, hit in enumerate(payload["results"], 1):
        with st.container(border=True):
            top = max(hit["dense_score"], hit["sparse_score"], 0.01)
            dense_pct = int(hit["dense_score"] / top * 100)
            sparse_pct = int(hit["sparse_score"] / top * 100)
            st.markdown(
                f"**{rank}. {hit['title']}** — :blue[{hit['section']}]  \n"
                f"`{hit['document_id']}` · {hit['category']} · {hit['jurisdiction'] or '—'} · {hit['product'] or '—'}")
            st.caption(
                f"dense {hit['dense_score']:.3f} · keyword {hit['sparse_score']:.3f} · "
                f"fused {hit['fused_score']:.4f}"
                + (f" · reranked {hit['rerank_score']:.3f}" if hit.get("rerank_score") is not None else ""))
            st.progress(dense_pct / 100, text=f"dense {dense_pct}%")
            st.progress(sparse_pct / 100, text=f"keyword {sparse_pct}%")
            st.write(hit["snippet"])
else:
    st.info("Enter a query to search the indexed corpus with hybrid dense + keyword retrieval.", icon=":material/search:")
    