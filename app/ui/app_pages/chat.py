"""Assistant page: conversational RAG with citations & guardrail badges."""

import streamlit as st
from app.ui import api_client

SUGGESTIONS = [
    "What records must be retained for investment portfolio?",
    "How should exceptions for trade finance be handled?",
    "When should issues involving foreign exchange be escalated?",
    "How should sanctions matches be escalated?",
    "What controls are required for credit card?",]


def _render_meta(meta: dict) -> None:
    if meta.get("citations"):
        with st.expander("Sources", icon=":material/menu_book:"):
            for citation in meta["citations"]:
                st.markdown(
                    f"**[{citation['n']}] {citation['title']}** — "
                    f":green[{citation['section']}] · `{citation['document_id']}`"
                )
                st.caption(citation["snippet"])
    badge = f":green[grounding {meta['grounding']:.0%}]" if meta.get("grounding", 0) >= 0.5 else (
        f":orange[grounding {meta['grounding']:.0%}]" if meta.get("grounding", 0) >= 0.2 else f":red[grounding {meta['grounding']:.0%}]")
    timings = meta.get("timings_ms", {})
    st.caption(
        f"{badge} · mode `{meta.get('mode')}` · model `{meta.get('model')}` · "
        f"retrieval {timings.get('retrieval_ms', 0):.0f} ms · total {timings.get('total_ms', 0):.0f} ms")
    if meta.get("warning"):
        st.warning(
            {
                "low_confidence": "Retrieval confidence is low — verify this answer against the cited documents.",
                "no_results": "No relevant documents were found in the knowledge base.",
                "uncited_answer": "The model did not cite its sources; treat the answer with caution.",
                "llm_error_fallback": "The generative model failed; an extractive fallback answer was produced.",
            }.get(meta["warning"], f"warning: {meta['warning']}"),
            icon=":material/warning:",)

for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"],
        avatar=":material/person:" if msg["role"] == "user" else ":material/smart_toy:",):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_meta(msg["meta"])

if not st.session_state.messages:
    st.caption("Ask any question about the bank's policies, procedures and standards. Answers cite their sources.")
    selected = st.pills("Try asking", SUGGESTIONS, label_visibility="collapsed")
    if selected:
        st.session_state.pending_question = selected
        st.rerun()

question = st.chat_input("Ask about policies, procedures, standards…", submit_mode="disable")
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.write(question)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        with st.status("Searching the knowledge base", type="step") as status:
            st.write(f"question: {question}")
            try:
                response = api_client.ask(
                    question,
                    principal=st.session_state.principal,
                    history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-6:]
                        if m["role"] in {"user", "assistant"}
                    ][-4:],)
                status.update(label="Answer ready", state="complete", expanded=False)
            except api_client.ApiError as exc:
                status.update(label="Request failed", state="error")
                st.error(str(exc))
                response = None
        if response:
            st.write(response["answer"])
            _render_meta(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"], "meta": response}
                )