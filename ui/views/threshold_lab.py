import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ui.common import DECISIONS, inr, kpi_row, metrics, scored
from src.conformal import evaluate_sets, fit_class_conditional
from src.costs import ACTIONS, decide_array, expected_cost, optimize_thresholds_fast


def render():
    st.title("Threshold & Cost Lab")
    st.caption("Change the economics and watch the optimal policy move. Thresholds are optimised on a validation slice and evaluated on the untouched test period.")
    d, M = scored(), metrics()
    va = d[d.split == "val"]; vb = va.iloc[len(va) // 2:]; te = d[d.split == "test"]
    base = M["business_impact_test"]["cost_assumptions"]
    c = st.columns(4)
    costs = dict(
        stepup_cost=c[0].slider("Step-up cost (Rs)", 0.5, 30.0, float(base["stepup_cost"])), stepup_catch=c[0].slider("Step-up stops fraud", 0.2, 1.0, float(base["stepup_catch"])),
        stepup_abandon=c[1].slider("Genuine abandon rate", 0.0, 0.2, float(base["stepup_abandon"])), abandon_cost=c[1].slider("Cost per abandon (Rs)", 0.0, 500.0, float(base["abandon_cost"])),
        review_cost=c[2].slider("Review cost (Rs)", 1.0, 300.0, float(base["review_cost"])), review_catch=c[2].slider("Review stops fraud", 0.5, 1.0, float(base["review_catch"])),
        block_fp_cost=c[3].slider("Wrongful block cost (Rs)", 10.0, 3000.0, float(base["block_fp_cost"])))
    th, _ = optimize_thresholds_fast(vb.risk_score, vb.is_fraud, vb.amount, costs)
    mode = st.radio("Policy", ["Cost-optimal (recomputed live)", "Manual thresholds"], horizontal=True)
    if mode.startswith("Manual"):
        lo = np.log10
        s1, s2, s3 = st.columns(3)
        th = dict(step_up=10 ** s1.slider("log10 step-up threshold", -4.0, 0.0, float(lo(max(th["step_up"], 1e-4)))), review=10 ** s2.slider("log10 review threshold", -4.0, 0.0, float(lo(max(th["review"], 1e-4)))),
                  block=10 ** s3.slider("log10 block threshold", -4.0, 0.0, float(lo(max(th["block"], 1e-4)))))
    ths = (th["step_up"], th["review"], th["block"])
    p, y, a = te.risk_score.to_numpy(), te.is_fraud.to_numpy(), te.amount.to_numpy()
    cost, none = expected_cost(p, y, a, ths, costs), float(a[y == 1].sum())
    dec = decide_array(p, ths); fl = dec != "APPROVE"
    kpi_row([("Step-up at", f"{th['step_up']:.4f}"), ("Review at", f"{th['review']:.4f}"), ("Block at", f"{th['block']:.4f}"), ("Total cost", inr(cost)), ("Savings vs. no model", f"{100 * (none - cost) / none:.1f}%")])
    kpi_row([("Flag rate", f"{fl.mean():.2%}"), ("Fraud recall (flagged)", f"{fl[y == 1].mean():.1%}"), ("Alert precision", f"{y[fl].mean():.1%}" if fl.any() else "-"),
             ("Genuine blocked", f"{int(((dec == 'BLOCK_TRANSACTION') & (y == 0)).sum()):,}"), ("Genuine reviewed", f"{int(((dec == 'REVIEW') & (y == 0)).sum()):,}")])
    tbl = pd.DataFrame({"payments": pd.Series(dec).value_counts(), "fraud": pd.Series(dec[y == 1]).value_counts()}).reindex(ACTIONS).fillna(0).astype(int)
    a1, b1 = st.columns(2)
    a1.subheader("Outcome by action"); a1.dataframe(tbl)
    sweep = np.geomspace(1e-3, 0.95, 50)
    sw = pd.DataFrame({"review_threshold": sweep, "cost": [expected_cost(p, y, a, (ths[0], max(t, ths[0]), max(ths[2], t * 1.001)), costs) for t in sweep]})
    b1.subheader("Cost vs. review threshold (others fixed)")
    b1.altair_chart(alt.Chart(sw).mark_line().encode(x=alt.X("review_threshold:Q", scale=alt.Scale(type="log")), y=alt.Y("cost:Q", title="total cost (Rs)")).properties(height=230), width="stretch")

    st.subheader("Capacity-constrained alternative")
    cap = st.slider("Alerts the team can handle per 10,000 payments", 5, 400, 100)
    k = int(cap / 10000 * len(p)); idx = np.argsort(-p)[:k]
    st.write(f"Flagging the top {cap} per 10,000 catches **{y[idx].sum() / y.sum():.1%}** of fraud at **{y[idx].mean():.1%}** precision (capacity-driven, ignores amounts).")

    st.subheader("Uncertainty: conformal prediction sets")
    st.caption("Class-conditional split conformal on the validation slice. Payments whose set holds both classes (or neither) are 'uncertain' - natural candidates for a human.")
    q1, q2 = st.columns(2)
    af = q1.slider("Allowed miss rate for fraud (1 - coverage)", 0.02, 0.4, 0.10); ag = q2.slider("Allowed miss rate for genuine", 0.005, 0.1, 0.02)
    q = fit_class_conditional(vb.risk_score.to_numpy(), vb.is_fraud.to_numpy(), af, ag); ev = evaluate_sets(p, y, q)
    kpi_row([("Fraud coverage", f"{ev['coverage_fraud']:.1%}", f"target {1 - af:.0%}"), ("Genuine coverage", f"{ev['coverage_genuine']:.1%}", f"target {1 - ag:.0%}"), ("Uncertain share", f"{ev['share_uncertain']:.2%}"),
             ("Fraud rate among uncertain", f"{ev['fraud_in_uncertain']:.1%}"), ("Confident-fraud precision", f"{ev['precision_confident_fraud']:.1%}" if ev["precision_confident_fraud"] is not None else "-")])
    _ = DECISIONS
