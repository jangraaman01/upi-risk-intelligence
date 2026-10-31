import altair as alt
import pandas as pd
import streamlit as st

from ui.common import J, REPORTS, metrics


def render():
    M = metrics()
    st.title("Model Performance (chronological hold-out)")
    st.dataframe(pd.DataFrame(M["models"]).T, width="stretch")
    cv = M["time_series_cv_pr_auc"]
    st.caption(f"Time-series CV PR-AUC {cv['mean']:.3f} +/- {cv['std']:.3f}  (folds {cv['folds']})  |  model v{M['model_version']}  |  trained {M['trained_at'][:19]}")
    img = REPORTS / "figures/model_overview.png"
    if img.exists():
        st.image(str(img))
    a, b = st.columns(2)
    a.subheader("Recall by fraud typology")
    r = pd.DataFrame({"typology": list(M["recall_by_fraud_type"]), "recall": list(M["recall_by_fraud_type"].values()), "test_cases": [M["fraud_by_type_test"][k] for k in M["recall_by_fraud_type"]]})
    a.altair_chart(alt.Chart(r).mark_bar(color="#1f77b4").encode(y=alt.Y("typology:N", sort="-x", title=None), x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")), tooltip=["typology", alt.Tooltip("recall:Q", format=".1%"), "test_cases"]).properties(height=280), width="stretch")
    b.subheader("Decision buckets (test)")
    t = pd.DataFrame(M["test_decisions"]).T; t["fraud_share"] = (t.fraud_in_bucket / t["count"]).round(4); b.dataframe(t, width="stretch")
    b.subheader("Thresholds"); b.json(M["thresholds"])
    st.subheader("Calibration")
    rel = pd.DataFrame(M["calibration"]["reliability"])
    ch = alt.Chart(rel).mark_line(point=True).encode(x=alt.X("predicted:Q", scale=alt.Scale(type="log")), y=alt.Y("observed:Q", scale=alt.Scale(type="log")), tooltip=["predicted", "observed", "n"])
    diag = alt.Chart(pd.DataFrame({"x": [1e-4, 1], "y": [1e-4, 1]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    st.altair_chart((ch + diag).properties(height=280), width="stretch")
    st.caption(f"Brier score: raw {M['calibration']['brier_raw']} -> calibrated {M['calibration']['brier_calibrated']}")
    st.subheader("Feature importance (mean |SHAP|, test sample)")
    imp = pd.Series(M["mean_abs_shap"]).rename("importance").reset_index().rename(columns={"index": "feature"})
    st.altair_chart(alt.Chart(imp).mark_bar().encode(y=alt.Y("feature:N", sort="-x", title=None), x="importance:Q").properties(height=420), width="stretch")
    _ = J
