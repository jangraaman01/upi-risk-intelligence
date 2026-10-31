import altair as alt
import streamlit as st

from ui.common import DECISION_COLORS, DECISIONS, inr, kpi_row, metrics, scored
from src.monitoring import robust_anomalies


def render():
    d, M = scored(), metrics()
    st.title("Executive Summary")
    st.caption("Post warm-up transactions; model performance figures are from the chronological hold-out (last 25% of time).")
    te = d[d.split == "test"]
    bi = M["business_impact_test"]
    kpi_row([("Transactions", f"{len(d):,}"), ("Payment volume", inr(d.amount.sum())), ("Fraud rate", f"{d.is_fraud.mean():.2%}"),
             ("Fraud value", inr(d[d.is_fraud == 1].amount.sum())), ("Net savings (test)", inr(bi["net_savings_inr"]), f"{bi['savings_pct']}%")])
    kpi_row([("Fraud caught (test)", f"{bi['fraud_recall_flagged']:.1%}"), ("Alert precision", f"{bi['precision_flagged']:.1%}"), ("Flag rate", f"{bi['flag_rate']:.2%}"),
             ("Users", f"{d.sender_id.nunique():,}"), ("Merchants", f"{d.loc[d.receiver_type == 'MERCHANT', 'receiver_id'].nunique():,}")])

    daily = d.set_index("timestamp").resample("D").agg(txns=("amount", "size"), fraud=("is_fraud", "sum"), volume=("amount", "sum")).reset_index()
    daily["fraud_rate"] = daily.fraud / daily.txns
    an = robust_anomalies(daily.set_index("timestamp").fraud_rate).reset_index()
    a, b = st.columns(2)
    base = alt.Chart(daily).encode(x=alt.X("timestamp:T", title=None))
    a.subheader("Daily volume"); a.altair_chart(base.mark_area(opacity=.5, color="#1f77b4").encode(y="txns:Q", tooltip=["timestamp:T", "txns:Q", alt.Tooltip("volume:Q", format=",.0f")]).properties(height=250), width="stretch")
    b.subheader("Daily fraud rate with anomaly flags")
    line = alt.Chart(an).mark_line(color="#d62728").encode(x=alt.X("timestamp:T", title=None), y=alt.Y("value:Q", title="fraud rate", axis=alt.Axis(format="%")))
    pts = alt.Chart(an[an.is_anomaly]).mark_point(size=90, color="black", filled=True).encode(x="timestamp:T", y="value:Q", tooltip=["timestamp:T", alt.Tooltip("value:Q", format=".2%"), alt.Tooltip("score:Q", format=".1f")])
    b.altair_chart((line + pts).properties(height=250), width="stretch")
    b.caption(f"{int(an.is_anomaly.sum())} anomalous day(s) vs. a trailing 14-day robust baseline (|z| > 3.5).")

    a, b = st.columns(2)
    a.subheader("Alert funnel (test period)")
    f = te.groupby("decision").agg(payments=("amount", "size"), fraud=("is_fraud", "sum")).reindex(DECISIONS).reset_index()
    a.altair_chart(alt.Chart(f).mark_bar().encode(y=alt.Y("decision:N", sort=DECISIONS, title=None), x=alt.X("payments:Q", scale=alt.Scale(type="symlog")), color=alt.Color("decision:N", scale=DECISION_COLORS, legend=None),
                                                   tooltip=["decision", "payments", "fraud"]).properties(height=200), width="stretch")
    a.dataframe(f.set_index("decision"))
    b.subheader("Fraud value by typology")
    t = d[d.is_fraud == 1].groupby("fraud_type").agg(txns=("amount", "size"), value=("amount", "sum")).reset_index().sort_values("value", ascending=False)
    b.altair_chart(alt.Chart(t).mark_bar(color="#d62728").encode(y=alt.Y("fraud_type:N", sort="-x", title=None), x=alt.X("value:Q", title="fraud value (Rs)"), tooltip=["fraud_type", "txns", alt.Tooltip("value:Q", format=",.0f")]).properties(height=260), width="stretch")

    st.subheader("Week over week")
    w = d.set_index("timestamp").resample("W").agg(txns=("amount", "size"), fraud=("is_fraud", "sum"), fraud_value=("amount", lambda s: 0.0)).drop(columns="fraud_value")
    w["fraud_rate"] = (w.fraud / w.txns).round(4); w["txns_wow_%"] = (w.txns.pct_change() * 100).round(1); w["fraud_wow_%"] = (w.fraud.pct_change() * 100).round(1)
    st.dataframe(w.tail(10))
