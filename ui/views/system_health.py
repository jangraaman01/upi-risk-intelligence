import pandas as pd
import streamlit as st

from ui.common import DATA, J, ROOT, kpi_row, metrics, world


def _mb(p):
    return f"{p.stat().st_size / 1e6:,.1f} MB" if p.exists() else "-"


def render():
    st.title("System Health")
    M = metrics()
    g = J("gates.json")
    st.subheader("Release quality gates")
    if g:
        (st.success if g["all_passed"] else st.error)(f"{'ALL GATES PASSED' if g['all_passed'] else 'GATE FAILURES'}  (run {g['run_at']})")
        st.dataframe(pd.DataFrame(g["gates"]).assign(status=lambda x: x.passed.map({True: "PASS", False: "FAIL"}))[["status", "gate", "value"]], width="stretch")
    else:
        st.info("Run `python -m src.gates` to evaluate the gates.")
    st.subheader("Model & data")
    w, dm = world(), M["data"]
    kpi_row([("Model version", M["model_version"]), ("Trained", M["trained_at"][:10]), ("World preset", w.get("scale", "?")), ("Rows after warm-up", f"{dm['rows']:,}"), ("Days", dm.get("days", "?"))])
    kpi_row([("Users", f"{dm.get('users', '?'):,}" if isinstance(dm.get("users"), int) else "?"), ("Merchants", f"{dm.get('merchants', '?'):,}" if isinstance(dm.get("merchants"), int) else "?"), ("Fraud rate", f"{dm['fraud_rate']:.2%}"), ("Warm-up days", dm["warmup_days"]), ("Seed", w.get("seed", "?"))])
    lat = J("latency.json")
    if lat:
        st.subheader("Serving latency (single core, in-process)")
        st.dataframe(pd.DataFrame({k: v for k, v in lat.items() if isinstance(v, dict)}).T)
    st.subheader("Artifacts on disk")
    files = ["data/transactions.parquet", "data/scored.parquet", "data/shap_sample.parquet", "data/store.pkl", "data/audit.db", "data/cases.db", "models/artifacts.joblib", "reports/metrics.json"]
    st.dataframe(pd.DataFrame({"file": files, "size": [_mb(ROOT / f) for f in files]}), width="stretch")
    st.caption(f"Thresholds in production: {M['thresholds']}")
    _ = DATA
