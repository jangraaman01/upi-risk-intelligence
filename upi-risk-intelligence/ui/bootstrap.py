"""First-launch bootstrap: hosted deployments (Streamlit Community Cloud, Spaces, ...) start from a git checkout that has no generated data.
If the artifacts are missing, simulate a small world and train the model once (about a minute); every later visit reuses the files."""
import os

import streamlit as st

from src.config import DATA, MODELS, REPORTS

NEEDED = [DATA / "scored.parquet", DATA / "transactions.parquet", DATA / "users.parquet", DATA / "merchants.parquet", DATA / "shap_sample.parquet",
          DATA / "rings.json", DATA / "mule_candidates.parquet", MODELS / "artifacts.joblib", REPORTS / "metrics.json"]


@st.cache_resource(show_spinner="First launch: simulating payments and training the model (about a minute, one time only)...")
def ensure_artifacts():
    if all(p.exists() for p in NEEDED):
        return "existing"
    from src.pipeline import run
    run(os.getenv("UPI_BOOT_SCALE", "small"))
    return "built"
