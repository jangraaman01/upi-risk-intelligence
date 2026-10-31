import pandas as pd
import streamlit as st

from ui.common import case_store, metrics
from src.audit import AuditLog


def render():
    st.title("Audit, Cases & Governance")
    log = AuditLog(); cs = case_store()
    t = st.tabs(["Decision audit trail", "Analyst cases", "Fairness audit"])
    with t[0]:
        st.write("Every API decision is stored with its reasons and model version:", log.stats())
        rows = log.recent(200)
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["No decisions logged yet - call POST /score"]}), width="stretch")
    with t[1]:
        st.write(cs.stats()); rows = cs.recent(200)
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["No cases resolved yet - use the Analyst Case Queue"]}), width="stretch")
    with t[2]:
        for dim, v in metrics().get("fairness", {}).items():
            st.subheader(dim)
            st.caption(f"adverse-impact ratio {v['adverse_impact_ratio']} | chi-square p = {v['homogeneity_p_value']} | {'SIGNIFICANT' if v['significant'] else 'not significant'}")
            st.dataframe(pd.DataFrame(v["segments"]).T, width="stretch")
