import altair as alt
import pandas as pd
import streamlit as st

from ui.common import DECISION_COLORS, DECISIONS, inr, metrics, scored, shap_sample
from src.features import FEATURES, FRIENDLY


def render():
    st.title("Typology Explorer")
    d, M = scored(), metrics()
    types = sorted(t for t in d.fraud_type.unique() if t != "none")
    t = st.selectbox("Fraud typology", types)
    te = d[d.split == "test"]
    g, ge = te[te.fraud_type == t], te[te.is_fraud == 0]
    rec = float((g.decision != "APPROVE").mean()) if len(g) else float("nan")
    c = st.columns(5); c[0].metric("Payments (all time)", f"{(d.fraud_type == t).sum():,}"); c[1].metric("Value", inr(d[d.fraud_type == t].amount.sum())); c[2].metric("Test payments", f"{len(g):,}")
    c[3].metric("Flagged (recall)", f"{rec:.1%}"); c[4].metric("Mean risk", f"{g.risk_score.mean():.3f}")
    a, b = st.columns(2)
    a.subheader("How the model actions this typology")
    dc = g.decision.value_counts().reindex(DECISIONS).fillna(0).reset_index(); dc.columns = ["decision", "payments"]
    a.altair_chart(alt.Chart(dc).mark_bar().encode(x=alt.X("decision:N", sort=DECISIONS, title=None), y="payments:Q", color=alt.Color("decision:N", scale=DECISION_COLORS, legend=None)).properties(height=240), width="stretch")
    b.subheader("Behavioural fingerprint (z-score vs. genuine payments)")
    z = ((g[FEATURES].mean() - ge[FEATURES].mean()) / ge[FEATURES].std().replace(0, 1)).rename("z").reset_index(); z["label"] = z["index"].map(FRIENDLY)
    z = z.reindex(z.z.abs().sort_values(ascending=False).index).head(10)
    b.altair_chart(alt.Chart(z).mark_bar().encode(y=alt.Y("label:N", sort=None, title=None), x=alt.X("z:Q", title="standardised difference"), color=alt.condition(alt.datum.z > 0, alt.value("#d62728"), alt.value("#1f77b4"))).properties(height=300), width="stretch")
    ss = shap_sample()
    s = ss[ss.fraud_type == t]
    if len(s):
        sh = s[[f"shap_{f}" for f in FEATURES]].mean().rename("mean_shap").reset_index(); sh["label"] = sh["index"].str.replace("shap_", "").map(FRIENDLY)
        sh = sh.reindex(sh.mean_shap.abs().sort_values(ascending=False).index).head(8)
        st.subheader("What the model keys on (mean SHAP for this typology)"); st.altair_chart(alt.Chart(sh).mark_bar().encode(y=alt.Y("label:N", sort=None, title=None), x="mean_shap:Q", color=alt.condition(alt.datum.mean_shap > 0, alt.value("#d62728"), alt.value("#2ca02c"))).properties(height=250), width="stretch")
    ts = d[d.fraud_type == t].set_index("timestamp").resample("W").size().rename("payments")
    st.subheader("Weekly volume"); st.bar_chart(ts, height=180)
    st.subheader("Missed examples (APPROVE) - why they slip through")
    st.dataframe(g[g.decision == "APPROVE"].sort_values("risk_score", ascending=False)[["timestamp", "sender_id", "receiver_id", "amount", "channel", "city", "risk_score"] + ["amount_ratio", "sender_cnt_10m", "new_receiver", "new_device"]].head(10).round({"risk_score": 3, "amount_ratio": 3}), width="stretch")
    st.caption(f"Recall by typology on the hold-out: {pd.Series(M['recall_by_fraud_type']).round(3).to_dict()}")
