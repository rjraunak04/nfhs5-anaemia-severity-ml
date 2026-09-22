"""Recruiter-facing aggregate dashboard for the anaemia ML project."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from anaemia_ml.dashboard import (
    DashboardDataError,
    load_portfolio_summary,
    portfolio_summary_from_bytes,
)

ROOT = Path(__file__).parent
DEFAULT_SUMMARY = ROOT / "demo" / "nfhs_development_summary.json"
SYNTHETIC_SUMMARY = ROOT / "demo" / "portfolio_summary.json"

st.set_page_config(
    page_title="NFHS-5 Anaemia Severity ML",
    page_icon="🩸",
    layout="wide",
)
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1180px;}
      [data-testid="stMetric"] {background: #f7f9fc; border: 1px solid #e5e7eb;
        border-radius: 12px; padding: 14px;}
      .truth-banner {background: #fff7ed; border-left: 5px solid #f97316;
        border-radius: 8px; padding: 14px 18px; margin: 10px 0 20px 0;}
      .small-note {color: #596579; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def _default_data(path: str) -> dict:
    return load_portfolio_summary(path)


def _read_data() -> dict:
    source = st.sidebar.radio(
        "Data source", ["NFHS development results", "Synthetic engineering demo"]
    )
    default_path = (
        DEFAULT_SUMMARY if source == "NFHS development results" else SYNTHETIC_SUMMARY
    )
    uploaded = st.sidebar.file_uploader(
        "Load aggregate portfolio_summary.json",
        type=["json"],
        help="Only aggregate Day 2 output is accepted; raw participant data are rejected.",
    )
    if uploaded is None:
        return _default_data(str(default_path))
    try:
        return portfolio_summary_from_bytes(uploaded.getvalue())
    except DashboardDataError as error:
        st.sidebar.error(str(error))
        return _default_data(str(default_path))


try:
    data = _read_data()
except DashboardDataError as error:
    st.error(f"Dashboard data failed validation: {error}")
    st.stop()

disclosure = data["disclosure"]
workflow = data["workflow"]
models = pd.DataFrame(data["model_comparison"])
calibration = data["calibration"]
explainability = data["explainability"]

st.title(data["project_title"])
st.caption("Survey-aware, leakage-resistant multiclass ML for anaemia severity in NFHS-5 India")
st.markdown(
    f'<div class="truth-banner"><strong>{escape(str(disclosure["headline"]))}</strong><br>'
    f'<span class="small-note">Source: {escape(str(disclosure["data_source"]))}. '
    "No clinical use and no final performance claim.</span></div>",
    unsafe_allow_html=True,
)

selected = data["selection"]
metric_columns = st.columns(4)
metric_columns[0].metric("Compared models", len(models))
metric_columns[1].metric("Selected model", selected["model_name"].replace("_", " ").title())
metric_columns[2].metric("Calibration", calibration["method"].replace("_", " ").title())
metric_columns[3].metric("Locked test", "Untouched")

overview_tab, calibration_tab, shap_tab, safeguards_tab = st.tabs(
    ["Overview", "Calibration", "SHAP", "Safeguards"]
)

with overview_tab:
    st.subheader("Development-only model comparison")
    if data["run_kind"] == "synthetic_smoke":
        st.success("All configured model paths completed the grouped evaluation smoke test.")
        st.dataframe(
            models[["display_name", "total_model_fits"]].rename(
                columns={"display_name": "Model", "total_model_fits": "Verified fits"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.info(
            "Synthetic fixture scores are intentionally hidden in the default recruiter "
            "view because they do not estimate performance on NFHS participants."
        )
    else:
        st.caption("Mean scores come from PSU-grouped nested cross-validation.")
        chart = models.set_index("display_name")[["macro_f1_mean", "balanced_accuracy_mean"]]
        st.bar_chart(chart, horizontal=True, x_label="Score", y_label="Model")
        display_columns = [
            "display_name",
            "macro_f1_mean",
            "macro_f1_standard_deviation",
            "balanced_accuracy_mean",
            "severe_recall_mean",
        ]
        st.dataframe(
            models[display_columns].rename(
                columns={
                    "display_name": "Model",
                    "macro_f1_mean": "Macro-F1",
                    "macro_f1_standard_deviation": "Macro-F1 SD",
                    "balanced_accuracy_mean": "Balanced accuracy",
                    "severe_recall_mean": "Severe recall",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.info("These are development estimates, not locked-test results or final claims.")

with calibration_tab:
    st.subheader("Probability calibration selection")
    if data["run_kind"] == "synthetic_smoke":
        st.success(
            "The calibration selection path completed. Synthetic diagnostic scores are hidden."
        )
    else:
        before = calibration["metrics_before"]
        after = calibration["deployed_metrics_estimate"]
        calibration_table = pd.DataFrame(
            {
                "Before": [
                    before["log_loss"],
                    before["multiclass_brier"],
                    before["expected_calibration_error"],
                ],
                "After selection": [
                    after["log_loss"],
                    after["multiclass_brier"],
                    after["expected_calibration_error"],
                ],
            },
            index=["Log loss", "Multiclass Brier", "ECE"],
        )
        st.bar_chart(calibration_table, horizontal=True)
    left, right = st.columns(2)
    left.metric("Temperature", f"{calibration['temperature']:.3f}")
    right.metric("Group cross-fit folds", calibration["cross_fit_splits"])
    st.caption(
        "Temperature scaling is accepted only when PSU-disjoint cross-fitted weighted "
        "log loss improves over identity."
    )

with shap_tab:
    st.subheader("Aggregate SHAP associations")
    if data["run_kind"] == "synthetic_smoke":
        st.info(
            "These attributions describe the synthetic fixture only; they are not NFHS findings."
        )
    st.caption(
        f"{explainability['sample_count']} calibration observations were sampled; "
        "row-level SHAP values were discarded immediately after aggregation."
    )
    global_features = pd.DataFrame(explainability["global_features"])
    if not global_features.empty:
        st.bar_chart(
            global_features.set_index("feature")[["mean_abs_shap"]],
            horizontal=True,
            x_label="Mean |SHAP|",
            y_label="Raw predictor",
        )
    st.warning("SHAP describes model associations, not causal effects or clinical importance.")

with safeguards_tab:
    st.subheader("What makes this evaluation trustworthy")
    partition_rows = workflow["partition_rows"]
    st.markdown(
        f"""
        - **PSU-disjoint partitions:** development {partition_rows["development"]},
          calibration {partition_rows["calibration"]}, locked test {partition_rows["locked_test"]}
        - **Fold-local preprocessing:** imputation, encoding, scaling and class weights fit
          only inside training folds
        - **Nested selection:** inner folds choose parameters; outer folds estimate
          development performance
        - **Calibration isolation:** only the calibration partition selects temperature
        - **Privacy:** dashboard accepts aggregate JSON only; no respondent rows or PSU IDs
        """
    )
    st.subheader("Responsible use")
    for statement in data["responsible_use"]:
        st.markdown(f"- {statement}")

st.divider()
st.caption(
    f"Day 2 workflow v{workflow['version']} · run: {data['run_kind']} · locked test: not evaluated"
)
