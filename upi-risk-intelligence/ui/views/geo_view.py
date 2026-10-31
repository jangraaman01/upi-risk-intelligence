import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ui.common import inr, kpi_row, scored
from src.geo import CITY_LAT, CITY_LON, CITY_NAMES


def render():
    st.title("Geo Intelligence")
    d = scored()
    st.caption("Payments are placed at the city the device appeared in. Impossible-travel detection uses the speed implied since the sender's previous payment.")
    g = d.groupby("city").agg(txns=("amount", "size"), fraud=("is_fraud", "sum"), fraud_value=("amount", lambda s: 0.0), flagged=("decision", lambda s: (s != "APPROVE").mean())).drop(columns="fraud_value")
    g["fraud_rate"] = g.fraud / g.txns
    g["fraud_value"] = d[d.is_fraud == 1].groupby("city").amount.sum()
    g = g.fillna(0).reset_index()
    pos = pd.DataFrame({"city": CITY_NAMES, "lat": CITY_LAT, "lon": CITY_LON})
    m = g.merge(pos, on="city")
    metric = st.radio("Map metric", ["fraud", "fraud_rate", "txns"], horizontal=True)
    mm = m.assign(size=(m[metric] / m[metric].max() * 60000 + 8000), latitude=m.lat, longitude=m.lon)
    st.map(mm, latitude="latitude", longitude="longitude", size="size", color="#d62728")
    a, b = st.columns(2)
    a.subheader("City league table"); a.dataframe(g.sort_values("fraud", ascending=False).set_index("city").round(4), width="stretch")
    b.subheader("Fraud rate by city")
    b.altair_chart(alt.Chart(g).mark_bar(color="#d62728").encode(y=alt.Y("city:N", sort="-x", title=None), x=alt.X("fraud_rate:Q", axis=alt.Axis(format="%")), tooltip=["city", "fraud", alt.Tooltip("fraud_rate:Q", format=".2%")]).properties(height=300), width="stretch")

    st.subheader("Impossible-travel events")
    d = d.assign(speed_kmh=np.expm1(d.geo_speed_log))
    thr = st.slider("Minimum implied speed (km/h)", 200, 3000, 600, step=100)
    ev = d[d.speed_kmh >= thr]
    kpi_row([("Events", f"{len(ev):,}"), ("Truly fraudulent", f"{ev.is_fraud.mean():.1%}" if len(ev) else "-"), ("Flagged by model", f"{(ev.decision != 'APPROVE').mean():.1%}" if len(ev) else "-"),
             ("Value", inr(ev.amount.sum()))])
    if len(ev):
        sc = alt.Chart(ev.sample(min(len(ev), 4000), random_state=1)).mark_circle(opacity=.6).encode(x=alt.X("speed_kmh:Q", scale=alt.Scale(type="log")), y=alt.Y("risk_score:Q"), color=alt.Color("is_fraud:N", scale=alt.Scale(domain=[0, 1], range=["#1f77b4", "#d62728"])),
                                                                                                    tooltip=["sender_id", "city", "amount", "speed_kmh", "risk_score", "fraud_type"]).properties(height=280)
        st.altair_chart(sc, width="stretch")
        st.dataframe(ev.sort_values("speed_kmh", ascending=False)[["timestamp", "sender_id", "city", "amount", "speed_kmh", "risk_score", "decision", "fraud_type"]].head(30))
    st.subheader("Trips vs. attacks")
    trip = d[(d.new_city == 1)]
    st.write(f"{len(trip):,} payments came from a city the customer had not used before: **{trip.is_fraud.mean():.1%}** are fraud, {(trip.decision != 'APPROVE').mean():.1%} are flagged - genuine travel is the main source of false alerts here.")
