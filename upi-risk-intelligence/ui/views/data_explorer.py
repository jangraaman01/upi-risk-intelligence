import altair as alt
import pandas as pd
import streamlit as st

from ui.common import DECISIONS, scored


def render():
    st.title("Data Explorer")
    d = scored()
    c = st.columns(4)
    lo, hi = d.timestamp.min().date(), d.timestamp.max().date()
    rng = c[0].date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    typ = c[1].multiselect("Fraud type", sorted(d.fraud_type.unique())); dec = c[2].multiselect("Decision", DECISIONS); cities = c[3].multiselect("City", sorted(d.city.unique()))
    c = st.columns(4)
    cats = c[0].multiselect("Merchant category", sorted(d.merchant_category.unique())); amin, amax = c[1].number_input("Min amount", 0.0, 1e6, 0.0), c[2].number_input("Max amount", 0.0, 1e6, 1e6)
    split = c[3].multiselect("Split", ["train", "val", "test"])
    f = d
    if isinstance(rng, tuple) and len(rng) == 2:
        f = f[(f.timestamp.dt.date >= rng[0]) & (f.timestamp.dt.date <= rng[1])]
    for col, vals in (("fraud_type", typ), ("decision", dec), ("city", cities), ("merchant_category", cats), ("split", split)):
        if vals:
            f = f[f[col].isin(vals)]
    f = f[(f.amount >= amin) & (f.amount <= amax)]
    st.write(f"**{len(f):,}** payments match | fraud {f.is_fraud.mean():.2%} | flagged {(f.decision != 'APPROVE').mean():.2%}")
    dim = st.selectbox("Group by", ["fraud_type", "decision", "city", "merchant_category", "channel", "device_os", "upi_app", "dow", "hour", "split"])
    g = f.groupby(dim).agg(payments=("amount", "size"), volume=("amount", "sum"), fraud=("is_fraud", "sum"), mean_risk=("risk_score", "mean"), flagged=("decision", lambda s: (s != "APPROVE").mean())).reset_index()
    g["fraud_rate"] = g.fraud / g.payments
    a, b = st.columns(2)
    a.dataframe(g.round(4), width="stretch")
    b.altair_chart(alt.Chart(g).mark_bar().encode(x=alt.X(f"{dim}:N", sort="-y"), y="payments:Q", color=alt.Color("fraud_rate:Q", scale=alt.Scale(scheme="reds")), tooltip=list(g.columns)).properties(height=280), width="stretch")
    show = f.sort_values("timestamp").head(2000)
    st.dataframe(show[["txn_id", "timestamp", "sender_id", "receiver_id", "amount", "channel", "city", "merchant_category", "risk_score", "decision", "fraud_type"]], width="stretch")
    st.download_button("Download first 50,000 matching rows (CSV)", f.head(50000).to_csv(index=False).encode(), "payments_filtered.csv", "text/csv")
    st.subheader("Data quality")
    q = pd.DataFrame({"column": d.columns, "nulls": d.isna().sum().values, "unique": d.nunique().values, "dtype": d.dtypes.astype(str).values})
    st.dataframe(q, width="stretch")
