import altair as alt
import streamlit as st

from ui.common import inr, merchants, scored


def render():
    st.title("Merchant Intelligence")
    d, Mx = scored(), merchants()
    m = d[d.receiver_type == "MERCHANT"]
    cat = m.groupby("merchant_category").agg(txns=("amount", "size"), volume=("amount", "sum"), fraud=("is_fraud", "sum"), flagged=("decision", lambda s: (s != "APPROVE").mean())).reset_index()
    cat["fraud_rate"] = cat.fraud / cat.txns
    a, b = st.columns(2)
    a.subheader("Fraud rate by category")
    a.altair_chart(alt.Chart(cat).mark_bar(color="#d62728").encode(y=alt.Y("merchant_category:N", sort="-x", title=None), x=alt.X("fraud_rate:Q", axis=alt.Axis(format="%")), tooltip=["merchant_category", "txns", "fraud", alt.Tooltip("fraud_rate:Q", format=".2%")]).properties(height=330), width="stretch")
    b.subheader("Volume vs. fraud (bubble = fraud count)")
    b.altair_chart(alt.Chart(cat).mark_circle().encode(x=alt.X("volume:Q", scale=alt.Scale(type="log")), y=alt.Y("fraud_rate:Q", axis=alt.Axis(format="%")), size="fraud:Q", color=alt.Color("merchant_category:N", legend=None),
                                                       tooltip=["merchant_category", "txns", "fraud"]).properties(height=330), width="stretch")
    st.subheader("Scam-merchant radar")
    st.caption("Merchants that many DISTINCT customers paid and whose payments were flagged - the merchant-side twin of mule detection.")
    f = m[m.decision != "APPROVE"].groupby("receiver_id").agg(flagged_payers=("sender_id", "nunique"), flagged_txns=("amount", "size"), flagged_value=("amount", "sum"), avg_risk=("risk_score", "mean"), true_fraud_share=("is_fraud", "mean")).reset_index()
    f = f.merge(Mx.rename(columns={"merchant_id": "receiver_id"})[["receiver_id", "category", "city", "acct_age_days"]], on="receiver_id", how="left").sort_values(["flagged_payers", "flagged_value"], ascending=False)
    st.dataframe(f.head(25).round(3), width="stretch")
    st.subheader("Merchant explorer")
    mid = st.selectbox("Merchant", f.receiver_id.head(25).tolist() or Mx.merchant_id.head(5).tolist())
    mt = m[m.receiver_id == mid]
    c = st.columns(4); c[0].metric("Payments", f"{len(mt):,}"); c[1].metric("Distinct payers", mt.sender_id.nunique()); c[2].metric("Value", inr(mt.amount.sum())); c[3].metric("Fraud share", f"{mt.is_fraud.mean():.1%}")
    daily = mt.set_index("timestamp").resample("D").agg(payments=("amount", "size"), fraud=("is_fraud", "sum")).reset_index()
    st.altair_chart(alt.Chart(daily).mark_bar().encode(x="timestamp:T", y="payments:Q", color=alt.value("#1f77b4")).properties(height=180) + alt.Chart(daily).mark_bar(color="#d62728").encode(x="timestamp:T", y="fraud:Q"), width="stretch")
