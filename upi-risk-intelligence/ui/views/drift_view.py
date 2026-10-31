import altair as alt
import pandas as pd
import streamlit as st

from ui.common import J, REPORTS, metrics, scored
from src.monitoring import robust_anomalies


def render():
    st.title("Drift & Monitoring")
    M, d = metrics(), scored()
    st.subheader("Feature drift (PSI: training period vs. test period)")
    df = pd.DataFrame(M["drift_psi_train_vs_test"]).T.reset_index().rename(columns={"index": "feature"})
    st.altair_chart(alt.Chart(df).mark_bar().encode(y=alt.Y("feature:N", sort="-x", title=None), x="psi:Q", color=alt.Color("status:N", scale=alt.Scale(domain=["stable", "watch", "DRIFT"], range=["#2ca02c", "#ff7f0e", "#d62728"]))).properties(height=520), width="stretch")
    st.caption("PSI < 0.1 stable | 0.1-0.25 watch | > 0.25 drift.  Beware: history features such as `new_receiver` drift benignly as payee history accumulates.")
    st.subheader("Output monitoring: daily mean risk score and flag rate")
    daily = d.set_index("timestamp").resample("D").agg(mean_risk=("risk_score", "mean"), flag_rate=("decision", lambda s: (s != "APPROVE").mean()))
    a, b = st.columns(2)
    for col, name in ((a, "mean_risk"), (b, "flag_rate")):
        an = robust_anomalies(daily[name]).reset_index()
        line = alt.Chart(an).mark_line().encode(x=alt.X("timestamp:T", title=None), y=alt.Y("value:Q", title=name))
        base = alt.Chart(an).mark_line(strokeDash=[4, 4], color="gray").encode(x="timestamp:T", y="baseline:Q")
        pts = alt.Chart(an[an.is_anomaly]).mark_point(size=80, color="red", filled=True).encode(x="timestamp:T", y="value:Q")
        col.altair_chart((line + base + pts).properties(height=220, title=f"{name} (red = anomaly vs trailing median)"), width="stretch")
    bt = J("backtest.json")
    st.subheader("Concept drift & label delay (backtest)")
    if bt:
        st.write(bt["config"])
        img = REPORTS / "figures/backtest.png"
        if img.exists():
            st.image(str(img))
        st.json(bt["summary_post_drift"])
    else:
        st.info("Run `python -m src.backtest` to generate the drift / label-delay backtest.")
