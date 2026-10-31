import altair as alt
import pandas as pd
import streamlit as st

from ui.common import shap_sample
from src.features import FEATURES, FRIENDLY


def render():
    st.title("Explainability")
    ss = shap_sample()
    st.caption(f"SHAP contributions (log-odds of the raw model) on a stratified test sample of {len(ss):,} payments (up to 500 per fraud typology + genuine traffic).")
    cols = [f"shap_{f}" for f in FEATURES]
    h = ss[ss.is_fraud == 1].groupby("fraud_type")[cols].apply(lambda g: g.abs().mean()).reset_index().melt("fraud_type", var_name="feature", value_name="mean_abs_shap")
    h["feature"] = h.feature.str.replace("shap_", "").map(FRIENDLY)
    st.subheader("What drives each typology (mean |SHAP|)")
    st.altair_chart(alt.Chart(h).mark_rect().encode(x=alt.X("feature:N", title=None, sort=None), y=alt.Y("fraud_type:N", title=None), color=alt.Color("mean_abs_shap:Q", scale=alt.Scale(scheme="reds")),
                                                     tooltip=["fraud_type", "feature", alt.Tooltip("mean_abs_shap:Q", format=".2f")]).properties(height=300), width="stretch")
    st.subheader("Dependence plot")
    f = st.selectbox("Feature", FEATURES, format_func=lambda x: FRIENDLY[x], index=FEATURES.index("amount_ratio"))
    s = ss.sample(min(len(ss), 6000), random_state=1)
    st.altair_chart(alt.Chart(s).mark_circle(opacity=.5, size=24).encode(x=alt.X(f"{f}:Q", title=FRIENDLY[f]), y=alt.Y(f"shap_{f}:Q", title="SHAP contribution"), color=alt.Color("is_fraud:N", scale=alt.Scale(domain=[0, 1], range=["#1f77b4", "#d62728"])),
                                                       tooltip=["fraud_type", f, f"shap_{f}"]).properties(height=340).interactive(), width="stretch")
    st.caption("Reading: points above zero push the payment towards fraud. A clear step or curve shows how the model uses the feature; colour shows whether those payments really were fraud.")
    st.subheader("Global ranking")
    g = ss[cols].abs().mean().rename("mean_abs_shap").reset_index(); g["feature"] = g["index"].str.replace("shap_", "").map(FRIENDLY)
    st.altair_chart(alt.Chart(g).mark_bar().encode(y=alt.Y("feature:N", sort="-x", title=None), x="mean_abs_shap:Q").properties(height=460), width="stretch")
    _ = pd
