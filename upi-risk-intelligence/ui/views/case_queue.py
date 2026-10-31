import altair as alt
import pandas as pd
import streamlit as st

from ui.common import case_store, inr, model_bundle, scored
from src.cases import OUTCOMES
from src.features import FEATURES
from src.scoring import explain_frame


def render():
    st.title("Analyst Case Queue")
    st.caption("REVIEW decisions ranked by expected loss (risk x amount). Every resolution is stored as a human-confirmed label - the feedback loop for retraining.")
    d, cs = scored(), case_store()
    te = d[(d.split == "test") & (d.decision == "REVIEW")].copy()
    te["expected_loss"] = te.risk_score * te.amount
    done = cs.resolved_ids()
    q = te[~te.txn_id.isin(done)].sort_values("expected_loss", ascending=False)
    stats = cs.stats()
    c = st.columns(4); c[0].metric("Open cases (test period)", f"{len(q):,}"); c[1].metric("Resolved", stats["resolved"]); c[2].metric("Confirmed fraud", stats.get("confirmed_fraud", 0))
    c[3].metric("Queue precision (analyst-confirmed)", f"{stats['review_precision']:.0%}" if stats["review_precision"] is not None else "-")
    if q.empty:
        st.success("Queue is empty."); return
    st.subheader("Top of the queue")
    st.dataframe(q[["txn_id", "timestamp", "sender_id", "receiver_id", "amount", "channel", "city", "risk_score", "expected_loss"]].head(40), width="stretch")
    pick = st.selectbox("Open case", q.txn_id.head(40).tolist())
    r = q[q.txn_id == pick].iloc[0]
    a, b = st.columns([1, 1])
    a.markdown(f"**{r.sender_id} -> {r.receiver_id}**  \nAmount **{inr(r.amount)}** via `{r.channel}` from **{r.city}**  \nRisk **{r.risk_score:.3f}** | expected loss {inr(r.expected_loss)}  \nDevice `{r.device_id}` | category `{r.merchant_category}`")
    ex = pd.DataFrame(explain_frame(model_bundle()["lgbm"], pd.DataFrame([r[FEATURES]]).astype(float), 7))
    ch = alt.Chart(ex).mark_bar().encode(y=alt.Y("label:N", sort=None, title=None), x=alt.X("contribution:Q", title="SHAP (log-odds)"), color=alt.condition(alt.datum.contribution > 0, alt.value("#d62728"), alt.value("#2ca02c")),
                                         tooltip=["label", "value", "contribution"]).properties(height=220)
    b.altair_chart(ch, width="stretch")
    hist = d[d.sender_id == r.sender_id].sort_values("timestamp"); hist = hist[hist.timestamp < r.timestamp].tail(8)
    st.write("Sender's previous payments"); st.dataframe(hist[["timestamp", "receiver_id", "amount", "channel", "city", "device_id", "decision"]], width="stretch")
    notes = st.text_input("Notes", key="notes")
    cols = st.columns(3)
    for col, (label, out) in zip(cols, [("Confirm fraud", OUTCOMES[0]), ("Mark false positive", OUTCOMES[1]), ("Escalate", OUTCOMES[2])]):
        if col.button(label):
            cs.resolve(pick, out, notes=notes, risk_score=float(r.risk_score), amount=float(r.amount), sender_id=r.sender_id, receiver_id=r.receiver_id); st.rerun()
    if st.checkbox("Reveal ground truth (demo only)"):
        st.info(f"Label in the simulation: fraud = {int(r.is_fraud)}, typology = {r.fraud_type}")

    st.subheader("Capacity planner")
    per_day = te.groupby(te.timestamp.dt.date).size()
    p1, p2 = st.columns(2)
    mins = p1.slider("Minutes per case", 2, 30, 8); analysts = p2.slider("Analysts on shift (7 productive hours)", 1, 60, 10)
    cap = analysts * 7 * 60 / mins
    st.write(f"Capacity **{cap:,.0f}** cases/day vs. average arrivals **{per_day.mean():,.0f}** (peak **{per_day.max():,}**). "
             f"{'Capacity is sufficient.' if cap >= per_day.max() else 'Peak days exceed capacity: raise the REVIEW threshold or add analysts.'}  Analysts needed for peak: **{int(per_day.max() * mins / 420) + 1}**.")
    st.bar_chart(per_day.rename("review cases / day"), height=200)
