import altair as alt
import pandas as pd
import streamlit as st

from ui.common import engine, model_bundle, scored
from src.features import FEATURES
from src.geo import CITY_NAMES
from src.scoring import explain_frame


def render():
    st.title("Transaction Simulator & What-If")
    t1, t2 = st.tabs(["Replay a test transaction", "Score a custom payment + what-if"])
    with t1:
        d = scored(); te = d[d.split == "test"]
        kind = st.selectbox("Type", ["any", "legit"] + sorted(t for t in d.fraud_type.unique() if t != "none"))
        pool = te if kind == "any" else te[te.fraud_type == ("none" if kind == "legit" else kind)]
        if st.button("Score a random transaction") or "row" not in st.session_state:
            st.session_state["row"] = pool.sample(1).iloc[0]
        r = st.session_state["row"]
        st.write(f"**{r.sender_id} -> {r.receiver_id}** | Rs {r.amount:,.2f} | {r.channel} | {r.city} | {r.timestamp}")
        c = st.columns(3); c[0].metric("Risk score", f"{r.risk_score:.3f}"); c[1].metric("Decision", r.decision); c[2].metric("Ground truth", r.fraud_type)
        ex = pd.DataFrame(explain_frame(model_bundle()["lgbm"], pd.DataFrame([r[FEATURES]]).astype(float), 8))
        st.altair_chart(alt.Chart(ex).mark_bar().encode(y=alt.Y("label:N", sort=None, title=None), x=alt.X("contribution:Q", title="SHAP (log-odds)"),
                                                         color=alt.condition(alt.datum.contribution > 0, alt.value("#d62728"), alt.value("#2ca02c")), tooltip=["label", "value", "contribution"]).properties(height=260), width="stretch")
    with t2:
        st.caption("Scores against the live streaming state (loads the engine on first use). Leave the timestamp blank for 'now'.")
        c = st.columns(3)
        s = c[0].text_input("Sender", "U00010"); rcv = c[1].text_input("Receiver", "U00500"); amt = c[2].number_input("Amount (Rs)", 1.0, 1_000_000.0, 45000.0)
        c = st.columns(4)
        ch = c[0].selectbox("Channel", ["p2p", "collect", "qr", "intent"], index=1); dev = c[1].text_input("Device id (blank = usual)", "D_unseen")
        city = c[2].selectbox("City", ["(usual)"] + list(CITY_NAMES)); ts = c[3].text_input("Timestamp", "")
        if st.button("Score and explore alternatives"):
            eng = engine()
            res = eng.what_if(s, rcv, amt, ch, dev or None, ts or None, None if city == "(usual)" else city)
            b = res["base"]
            st.metric("Decision", b["decision"], f"risk {b['risk_score']:.3f}  |  mode {b['mode']}  |  city {b['city']}")
            for w in b["warnings"]:
                st.warning(w)
            st.subheader("What would change the outcome?")
            st.dataframe(pd.DataFrame(res["scenarios"]), width="stretch")
            full = eng.score_transaction(s, rcv, amt, ch, dev or None, ts or None, None if city == "(usual)" else city)
            st.subheader("Reasons"); st.dataframe(pd.DataFrame(full["reasons"])[["label", "value", "contribution"]], width="stretch")
