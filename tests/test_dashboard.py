from pathlib import Path

import pytest
from src.config import DATA, MODELS, REPORTS

pytestmark = pytest.mark.skipif(not ((MODELS / "artifacts.joblib").exists() and (DATA / "scored.parquet").exists() and (REPORTS / "metrics.json").exists()),
                                reason="run `python -m src.pipeline` first")


def test_every_dashboard_page_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "dashboard.py"), default_timeout=300).run()
    assert not at.exception
    n = 0
    for g in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].set_value(g).run()
        for p in at.sidebar.radio[0].options:
            at.sidebar.radio[0].set_value(p).run()
            assert not at.exception, f"{g} / {p}: {at.exception[0].value if at.exception else ''}"
            n += 1
    assert n == 17


def test_bootstrap_artifact_list_matches_what_the_pipeline_writes():
    from ui.bootstrap import NEEDED
    missing = [str(p) for p in NEEDED if not p.exists()]
    assert not missing, f"bootstrap expects files the pipeline does not produce: {missing}"
