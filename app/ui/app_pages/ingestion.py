"""Ingestion page: upload documents or run the pipeline, and inspect run reports."""

import tempfile
from pathlib import Path
import streamlit as st
from app.ui import api_client

st.caption(
    "The pipeline loads, cleans, enriches, chunks, embeds and indexes documents. "
    "Ingestion is incremental: unchanged files are skipped by content hash.")

uploads = st.file_uploader(
    "Upload documents",
    accept_file="multiple",
    type=["md", "txt", "pdf", "docx", "csv", "json"],)
if uploads and st.button("Ingest uploads", type="primary", icon=":material/upload:"):
    progress = st.progress(0.0, text="Uploading…")
    for i, upload in enumerate(uploads, 1):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / upload.name
            target.write_bytes(upload.getvalue())
            try:
                report = api_client.ingest_upload(target)
                st.toast(f"{upload.name}: {report['chunks_indexed']} chunks indexed")
            except api_client.ApiError as exc:
                st.error(f"{upload.name}: {exc}")
        progress.progress(i / len(uploads), text=f"{i}/{len(uploads)}")
    st.cache_data.clear()

with st.expander("Run the pipeline over a server-side folder", icon=":material/folder_open:"):
    source_dir = st.text_input("Folder on the server", placeholder="dataset/raw_documents")
    col_a, col_b, col_c = st.columns(3)
    dry_run = col_a.toggle("Dry run", value=False)
    rebuild = col_b.toggle("Rebuild index", value=False)
    limit = col_c.number_input("Limit files", 0, 5000, 0)
    if st.button("Run ingestion", icon=":material/play_arrow:", disabled=not source_dir):
        try:
            with st.spinner("Ingesting…"):
                report = api_client.ingest(
                    source_dir=source_dir,
                    rebuild=rebuild,
                    dry_run=dry_run,
                    limit=limit or None,)
            st.session_state.last_ingest_report = report
            st.success(
                f"{report['docs_ingested']} documents ingested, "
                f"{report['docs_skipped']} skipped, {report['chunks_indexed']} chunks indexed.")
        except api_client.ApiError as exc:
            st.error(str(exc))

report = st.session_state.get("last_ingest_report")
if report:
    with st.expander("Last ingestion report", expanded=True, icon=":material/receipt_long:"):
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Documents seen", report["docs_seen"])
        col_b.metric("Ingested", report["docs_ingested"])
        col_c.metric("Skipped (unchanged)", report["docs_skipped"])
        col_d.metric("Chunks indexed", report["chunks_indexed"])
        st.caption(f"run {report['run_id']} · embeddings `{report['embedding_provider']}` (dim {report['embedding_dim']})")
        st.dataframe(
            [{"stage": stage, "ms": ms} for stage, ms in report["stage_ms"].items()],
            hide_index=True,
            width="content",)
        for error in report["errors"][:10]:
            st.warning(f"{error.get('file')}: {error.get('error')}")