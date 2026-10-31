"""UPI Risk Intelligence console - 17 modules in five groups.   streamlit run dashboard/dashboard.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from ui.bootstrap import ensure_artifacts  # noqa: E402
from ui.common import metrics, world  # noqa: E402
from ui.views import (audit_log, case_queue, customer360, data_explorer, drift_view, evidence, executive, explain, geo_view, live_stream, merchants_view,  # noqa: E402
                             model_perf, rings_mules, simulator, system_health, threshold_lab, typology)

st.set_page_config("UPI Risk Intelligence", layout="wide", page_icon=None)

GROUPS = {
    "Command Center": {"Executive Summary": executive.render, "Live Stream Monitor": live_stream.render, "Analyst Case Queue": case_queue.render},
    "Investigation": {"Customer 360": customer360.render, "Geo Intelligence": geo_view.render, "Merchant Intelligence": merchants_view.render,
                      "Rings & Mules": rings_mules.render, "Typology Explorer": typology.render},
    "Model & Policy": {"Model Performance": model_perf.render, "Threshold & Cost Lab": threshold_lab.render, "Explainability": explain.render, "Simulator & What-If": simulator.render},
    "Monitoring & Evidence": {"Drift & Monitoring": drift_view.render, "Evidence & Robustness": evidence.render, "System Health": system_health.render},
    "Data & Governance": {"Data Explorer": data_explorer.render, "Audit & Cases": audit_log.render},
}

ensure_artifacts()
M, W = metrics(), world()
st.sidebar.title("UPI Risk Intelligence")
group = st.sidebar.selectbox("Area", list(GROUPS))
page = st.sidebar.radio("Module", list(GROUPS[group]))
st.sidebar.divider()
st.sidebar.caption(f"Model v{M['model_version']}  |  trained {M['trained_at'][:10]}  \nWorld: {W.get('scale', 'custom')} ({M['data']['rows']:,} payments)  \nSimulated UPI data")
GROUPS[group][page]()
