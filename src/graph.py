"""Fraud-ring detection: strongly-connected clusters of repeated P2P flows inside short windows."""
import networkx as nx
import numpy as np
import pandas as pd


def detect_rings(df, window_h=72, step_h=24, min_edge=2, min_size=3):
    p2p = df[df.receiver_type == "P2P"]
    found = []
    for s in pd.date_range(p2p.timestamp.min().floor("D"), p2p.timestamp.max(), freq=f"{step_h}h"):
        w = p2p[(p2p.timestamp >= s) & (p2p.timestamp < s + pd.Timedelta(hours=window_h))]
        e = w.groupby(["sender_id", "receiver_id"]).size().rename("n").reset_index()
        e = e[e.n >= min_edge]
        if e.empty:
            continue
        G = nx.from_pandas_edgelist(e, "sender_id", "receiver_id", create_using=nx.DiGraph)
        found += [set(c) for c in nx.strongly_connected_components(G) if len(c) >= min_size]
    merged = []
    for c in found:                                    # union overlapping clusters
        c = set(c)
        hit = [m for m in merged if m & c]
        for m in hit:
            c |= m; merged.remove(m)
        merged.append(c)
    rings = []
    for k, nodes in enumerate(sorted(merged, key=len, reverse=True)):
        sub = p2p[p2p.sender_id.isin(nodes) & p2p.receiver_id.isin(nodes)]
        rings.append(dict(ring_id=k, members=sorted(nodes), size=len(nodes), n_txn=len(sub),
                          total_amount=round(float(sub.amount.sum()), 2),
                          first_seen=str(sub.timestamp.min()), last_seen=str(sub.timestamp.max()),
                          risk=round(float(np.tanh(len(sub) / (3 * len(nodes)))), 3)))
    return rings


def evaluate_rings(df, rings):
    truth = df[df.ring_id >= 0]
    t_nodes = set(truth.sender_id) | set(truth.receiver_id)
    d_nodes = {m for r in rings for m in r["members"]}
    tp = len(t_nodes & d_nodes)
    true_rings = [set(g.sender_id) | set(g.receiver_id) for _, g in truth.groupby("ring_id")]
    rec = sum(any(len(t & set(r["members"])) / len(t | set(r["members"])) >= 0.5 for r in rings) for t in true_rings)
    return dict(true_rings=len(true_rings), rings_recovered=rec, detected_clusters=len(rings),
                node_precision=round(tp / max(len(d_nodes), 1), 3), node_recall=round(tp / len(t_nodes), 3))


def rank_mule_candidates(scored: pd.DataFrame, top=50) -> pd.DataFrame:
    """Guilt-by-association: accounts receiving P2P money from many DISTINCT senders whose payments were flagged."""
    f = scored[(scored.receiver_type == "P2P") & (scored.decision != "APPROVE")]
    g = f.groupby("receiver_id").agg(flagged_senders=("sender_id", "nunique"), flagged_txns=("txn_id", "size"),
                                     flagged_amount=("amount", "sum"), max_risk=("risk_score", "max")).reset_index()
    g = g.rename(columns={"receiver_id": "account"})
    return g.sort_values(["flagged_senders", "flagged_amount"], ascending=False).head(top).reset_index(drop=True)


def evaluate_mule_ranking(cands: pd.DataFrame, truth_accounts: set, k=20) -> dict:
    top = cands.head(k)
    return dict(k=k, precision_at_k=round(float(top.account.isin(truth_accounts).mean()), 3) if len(top) else 0.0)
