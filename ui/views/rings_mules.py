import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from ui.common import metrics, mule_candidates, rings, scored


def render():
    st.title("Fraud Rings & Mule Accounts (graph analytics)")
    d, R, M = scored(), rings(), metrics()
    a, b = st.columns(2)
    a.subheader("Ring detection quality"); a.json(M["rings"]); b.subheader("Mule ranking quality"); b.json(M["mule_ranking"])
    if R:
        st.dataframe(pd.DataFrame(R).drop(columns="members"), width="stretch")
        k = st.selectbox("Inspect ring", [r["ring_id"] for r in R]); mem = set(R[k]["members"])
        sub = d[(d.receiver_type == "P2P") & d.sender_id.isin(mem) & d.receiver_id.isin(mem)]
        e = sub.groupby(["sender_id", "receiver_id"]).amount.agg(["size", "sum"]).reset_index()
        G = nx.from_pandas_edgelist(e, "sender_id", "receiver_id", ["size"], create_using=nx.DiGraph)
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 5))
        nx.draw_networkx(G, nx.circular_layout(G), ax=ax, node_color="#d62728", node_size=500, font_size=7, font_color="w", width=[1 + x["size"] / 4 for *_, x in G.edges(data=True)], arrowsize=14)
        ax.axis("off"); c1.pyplot(fig); plt.close(fig)
        c2.write(f"**{len(mem)} accounts, {len(sub)} payments, {sub.amount.sum():,.0f} Rs circulated** between {sub.timestamp.min():%d %b %H:%M} and {sub.timestamp.max():%d %b %H:%M}")
        c2.write(f"Mean amount {sub.amount.mean():,.0f}; share of round amounts {(sub.is_round == 1).mean():.0%}; model flagged {(sub.decision != 'APPROVE').mean():.0%} of these payments.")
        c2.dataframe(sub.sort_values("timestamp")[["timestamp", "sender_id", "receiver_id", "amount", "risk_score", "decision"]].head(15))
    st.subheader("Top mule-account candidates (flagged inflow from distinct senders)")
    st.dataframe(mule_candidates(), width="stretch")
    st.subheader("Fan-in hot spots: receivers with the most new payers in 24h")
    fi = d[d.receiver_type == "P2P"].sort_values("recv_new_payers_24h", ascending=False).drop_duplicates("receiver_id").head(15)
    st.dataframe(fi[["receiver_id", "recv_new_payers_24h", "recv_cnt_24h", "recv_acct_age_log", "risk_score", "fraud_type"]].round(3), width="stretch")
