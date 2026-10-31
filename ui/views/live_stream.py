import time

import numpy as np
import pandas as pd
import streamlit as st

from ui.common import DECISIONS, inr, kpi_row, scored
from src.config import COSTS

CATCH = {"APPROVE": 0.0, "STEP_UP_AUTH": COSTS["stepup_catch"], "REVIEW": COSTS["review_catch"], "BLOCK_TRANSACTION": 1.0}


def _step(state, rows):
    state["n"] += len(rows)
    for dcs, g in rows.groupby("decision"):
        state["dec"][dcs] += len(g)
    fr = rows[rows.is_fraud == 1]
    state["fraud"] += len(fr); state["fraud_caught"] += int((fr.decision != "APPROVE").sum())
    state["loss_no_model"] += float(fr.amount.sum())
    state["loss_model"] += float((fr.amount * (1 - fr.decision.map(CATCH))).sum())
    state["hist"].append(dict(n=state["n"], without_model=state["loss_no_model"], with_model=state["loss_model"]))
    state["recent"] = pd.concat([rows[rows.decision != "APPROVE"], state["recent"]]).head(15)


def render():
    st.title("Live Stream Monitor")
    st.caption("Replays the held-out test period in time order, as if payments were arriving at the gateway. Loss with the model assumes each action stops its modelled share of fraud.")
    te = scored().query("split == 'test'").sort_values("timestamp").reset_index(drop=True)
    if "stream" not in st.session_state:
        st.session_state.stream = dict(pos=0, n=0, dec={k: 0 for k in DECISIONS}, fraud=0, fraud_caught=0, loss_no_model=0.0, loss_model=0.0, hist=[], recent=te.iloc[:0])
    s = st.session_state.stream
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    size = c1.slider("Payments per tick", 50, 5000, 1000, step=50)
    tick = c2.button("Process one tick"); auto = c3.button("Auto-play 25 ticks"); reset = c4.button("Reset")
    if reset:
        del st.session_state["stream"]; st.rerun()
    ph_kpi, ph_chart, ph_tbl = st.empty(), st.empty(), st.empty()

    def paint():
        with ph_kpi.container():
            kpi_row([("Processed", f"{s['n']:,}", f"of {len(te):,}"), ("Fraud seen", f"{s['fraud']:,}"), ("Fraud flagged", f"{s['fraud_caught'] / max(s['fraud'], 1):.1%}"),
                     ("Loss without model", inr(s["loss_no_model"])), ("Loss with model", inr(s["loss_model"]))])
        if s["hist"]:
            ph_chart.line_chart(pd.DataFrame(s["hist"]).set_index("n")[["without_model", "with_model"]], height=260)
        ph_tbl.dataframe(s["recent"][["timestamp", "sender_id", "receiver_id", "amount", "city", "risk_score", "decision", "fraud_type"]] if len(s["recent"]) else pd.DataFrame({"info": ["No alerts yet - process a tick"]}))

    def tick_once():
        if s["pos"] >= len(te):
            return False
        _step(s, te.iloc[s["pos"]: s["pos"] + size]); s["pos"] += size
        return True

    if tick:
        tick_once()
    if auto:
        for _ in range(25):
            if not tick_once():
                break
            paint(); time.sleep(0.12)
    paint()
    if s["n"]:
        st.write("Decisions so far:", {k: int(v) for k, v in s["dec"].items()}, "| fraud missed:", int(s["fraud"] - s["fraud_caught"]), "| share of stream:", f"{s['pos'] / len(te):.0%}")
    _ = np
