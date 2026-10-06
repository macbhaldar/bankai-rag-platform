"""Evaluation page: run the retrieval/QA benchmarks and inspect the metrics."""

import streamlit as st
from app.ui import api_client

BENCHMARKS = {
    "Retrieval (Recall@K, MRR, nDCG)": "retrieval",
    "QA test split (doc & section hits)": "qa_test",
    "QA validation split": "qa_validation",
    "Hard-negative robustness": "hard_negatives",}

with st.container(horizontal=True):
    benchmark_label = st.segmented_control(
        "Benchmark", list(BENCHMARKS), default="Retrieval (Recall@K, MRR, nDCG)")
    k = st.slider("K", 5, 50, 10)
    limit = st.number_input("Limit cases (0 = all)", 0, 1000, 0)

if st.button("Run evaluation", type="primary", icon=":material/play_arrow:"):
    try:
        with st.spinner("Evaluating — this retrieves every benchmark case…"):
            report = api_client.run_eval(
                benchmark=BENCHMARKS[benchmark_label],
                k=k,
                limit=limit or None,)
        st.session_state.eval_report = report
    except api_client.ApiError as exc:
        st.error(str(exc))

report = st.session_state.get("eval_report")
if report:
    st.caption(
        f"{report['benchmark']} · {report['n_cases']} cases · K={report['k']} · "
        f"{report['duration_ms'] / 1000:.1f}s · embeddings `{report['model_info']['embedding']}`")
    metric_cols = st.columns(min(len(report["metrics"]), 4))
    for col, (name, value) in zip(iter(metric_cols), report["metrics"].items()):
        col.metric(label=name.replace("_", " "), value=f"{value:.3f}")
    st.dataframe(report["per_case"], width="stretch", hide_index=True)
    st.download_button(
        "Download full report (JSON)",
        data=__import__("json").dumps(report, indent=2),
        file_name=f"eval_{report['benchmark']}.json",
        mime="application/json",)
else:
    st.info(
        "Benchmarks ship with the dataset: 500 retrieval queries with graded relevance "
        "judgments, 1,000 QA pairs with document/section ground truth, and 100 hard negatives. "
        "Run one to measure Recall@K, MRR, nDCG and doc/section hit rates of the live index.",
        icon=":material/analytics:",)