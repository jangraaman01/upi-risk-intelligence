"""Shared loaders and helpers for the Streamlit console (all heavy objects are cached once per session)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import DATA, REPORTS  # noqa: E402

alt.data_transformers.disable_max_rows()
DECISIONS = ["APPROVE", "STEP_UP_AUTH", "REVIEW", "BLOCK_TRANSACTION"]
DECISION_COLORS = alt.Scale(domain=DECISIONS, range=["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"])


def inr(x) -> str:
    x = float(x)
    return f"Rs {x / 1e7:,.2f} Cr" if abs(x) >= 1e7 else f"Rs {x / 1e5:,.2f} L" if abs(x) >= 1e5 else f"Rs {x:,.0f}"


def J(name):
    f = REPORTS / name
    return json.load(open(f)) if f.exists() else None


@st.cache_data(show_spinner="Loading scored transactions...")
def scored() -> pd.DataFrame:
    return pd.read_parquet(DATA / "scored.parquet")


@st.cache_data
def users() -> pd.DataFrame:
    return pd.read_parquet(DATA / "users.parquet")


@st.cache_data
def merchants() -> pd.DataFrame:
    return pd.read_parquet(DATA / "merchants.parquet")


@st.cache_data
def metrics() -> dict:
    return json.load(open(REPORTS / "metrics.json"))


@st.cache_data
def rings() -> list:
    return json.load(open(DATA / "rings.json"))


@st.cache_data
def mule_candidates() -> pd.DataFrame:
    return pd.read_parquet(DATA / "mule_candidates.parquet")


@st.cache_data
def shap_sample() -> pd.DataFrame:
    return pd.read_parquet(DATA / "shap_sample.parquet")


@st.cache_data
def world() -> dict:
    f = DATA / "world.json"
    return json.load(open(f)) if f.exists() else {}


@st.cache_resource(show_spinner="Starting the scoring engine (loads history + streaming state)...")
def engine():
    from src.scoring import RiskEngine
    return RiskEngine()


@st.cache_resource
def model_bundle():
    from src.scoring import load_artifacts
    return load_artifacts()


@st.cache_resource
def case_store():
    from src.cases import CaseStore
    return CaseStore()


def bar(df, x, y, color=None, horizontal=False, title="", height=260):
    enc = dict(x=alt.X(x, title=None) if not horizontal else alt.X(y), y=alt.Y(y) if not horizontal else alt.Y(x, sort="-x", title=None))
    if color:
        enc["color"] = color
    return alt.Chart(df, title=title).mark_bar().encode(**enc).properties(height=height)


def kpi_row(items):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.metric(*it)
