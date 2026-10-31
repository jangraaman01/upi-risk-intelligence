import altair as alt
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from ui.common import DECISION_COLORS, DECISIONS, inr, scored, users


def render():
    st.title("Customer 360 / Investigation")
    d, U = scored(), users().set_index("user_id")
    risky = d[d.decision.isin(["REVIEW", "BLOCK_TRANSACTION"])].groupby("sender_id").risk_score.agg(["max", "size"]).sort_values(["size", "max"], ascending=False).head(15)
    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Most-flagged customers", ["(type an id)"] + risky.index.tolist())
    uid = c2.text_input("Customer id", risky.index[0] if pick == "(type an id)" else pick)
    if uid not in U.index:
        st.warning("Unknown customer id (format U00010)."); return
    u = U.loc[uid]; tx = d[d.sender_id == uid].sort_values("timestamp")
    inbound = d[d.receiver_id == uid]
    c = st.columns(5); c[0].metric("Home city", u.city); c[1].metric("Age", int(u.age)); c[2].metric("Account age (days)", int(u.acct_age_days)); c[3].metric("Payments sent", len(tx)); c[4].metric("Payments received", len(inbound))
    flags = []
    if u.is_mule: flags.append("known mule (simulation ground truth)")
    if u.is_dormant: flags.append("dormant account profile")
    if tx.city.nunique() > 1: flags.append(f"{tx.city.nunique()} cities used")
    if tx.device_id.nunique() > 2: flags.append(f"{tx.device_id.nunique()} devices")
    st.write("Signals:", ", ".join(flags) if flags else "none")
    if tx.empty:
        st.info("This customer has no outgoing payments in the data."); return
    tx = tx.assign(decision=pd.Categorical(tx.decision, DECISIONS))
    ch = alt.Chart(tx).mark_circle().encode(x=alt.X("timestamp:T", title=None), y=alt.Y("amount:Q", scale=alt.Scale(type="log")), size=alt.Size("risk_score:Q", scale=alt.Scale(range=[20, 400]), legend=None),
                                            color=alt.Color("decision:N", scale=DECISION_COLORS), shape=alt.Shape("is_fraud:N", scale=alt.Scale(domain=[0, 1], range=["circle", "diamond"])),
                                            tooltip=["timestamp:T", "receiver_id", alt.Tooltip("amount:Q", format=",.0f"), "channel", "city", "decision", alt.Tooltip("risk_score:Q", format=".3f"), "fraud_type"]).properties(height=300, title="Payment timeline (diamond = confirmed fraud in the simulation)")
    st.altair_chart(ch, width="stretch")
    a, b = st.columns(2)
    a.subheader("Top counterparties"); a.dataframe(tx.groupby("receiver_id").agg(payments=("amount", "size"), total=("amount", "sum"), max_risk=("risk_score", "max")).sort_values("total", ascending=False).head(10))
    b.subheader("Devices & cities"); b.dataframe(tx.groupby(["device_id", "city"]).agg(payments=("amount", "size"), first_seen=("timestamp", "min")).sort_values("first_seen").head(10))
    st.subheader("Relationship graph (ego network)")
    out = tx.groupby("receiver_id").amount.agg(["size", "sum"]).sort_values("sum", ascending=False).head(12)
    inn = inbound.groupby("sender_id").amount.agg(["size", "sum"]).sort_values("sum", ascending=False).head(12)
    G = nx.DiGraph(); G.add_node(uid)
    for r, row in out.iterrows(): G.add_edge(uid, r, w=row["size"])
    for s_, row in inn.iterrows(): G.add_edge(s_, uid, w=row["size"])
    fig, ax = plt.subplots(figsize=(6, 4.5)); pos = nx.spring_layout(G, seed=3, k=.8)
    nx.draw_networkx(G, pos, ax=ax, node_color=["#d62728" if n == uid else "#1f77b4" for n in G], node_size=[700 if n == uid else 260 for n in G], font_size=6, font_color="w",
                     width=[1 + G[a_][b_]["w"] / 3 for a_, b_ in G.edges], arrowsize=10)
    ax.axis("off"); st.pyplot(fig); plt.close(fig)
    _ = inr
