import numpy as np
import pandas as pd
import pytest
from src.backtest import _flag_recall
from src.fairness import segment_report
from src.gates import checks
from src.generate_data import generate
from src.realdata import RealDataError, load, prepare
from src.robustness import cluster_bootstrap


def test_cluster_bootstrap_perfect_and_random_scores():
    rng = np.random.default_rng(0)
    clusters = rng.integers(0, 200, 4000); y = (rng.random(4000) < 0.05).astype(int)
    flags = rng.random(4000) < 0.03
    out = cluster_bootstrap(clusters, y, {"perfect": y + rng.random(4000) * 0.01, "random": rng.random(4000)}, flags, n_boot=60)
    lo, hi = out["perfect"]["pr_auc"]["ci95"]
    assert lo > 0.95 and hi <= 1.0
    r = out["random"]["pr_auc"]
    assert r["ci95"][1] < 0.5 and r["ci95"][0] <= r["point"] <= r["ci95"][1] + 0.05
    assert out["paired_diff_pr_auc_perfect_minus_random"]["ci95"][0] > 0.4       # paired difference is clearly positive


def test_segment_report_hand_computed():
    n = 1000
    sc = pd.DataFrame(dict(sender_id=["A"] * n + ["B"] * n, upi_app="a", is_fraud=0, timestamp=pd.date_range("2026-01-01", periods=2 * n, freq="min"),
                           decision=["REVIEW"] * 50 + ["APPROVE"] * (n - 50) + ["REVIEW"] * 100 + ["APPROVE"] * (n - 100)))
    users = pd.DataFrame(dict(user_id=["A", "B"], age=[25, 70], city=["X", "Y"], upi_app=["a", "a"]))
    r = segment_report(sc, users)
    assert r["age_band"]["segments"]["18-29"]["genuine_review_block_rate"] == pytest.approx(0.05)
    assert r["age_band"]["segments"]["60+"]["genuine_review_block_rate"] == pytest.approx(0.10)
    assert r["age_band"]["adverse_impact_ratio"] == pytest.approx(2.0)
    assert r["age_band"]["significant"] is True and r["age_band"]["segments"]["60+"]["rate_ci95"][0] < 0.10 < r["age_band"]["segments"]["60+"]["rate_ci95"][1]


def test_fairness_ignores_pure_noise():
    rng = np.random.default_rng(3); n = 6000
    sc = pd.DataFrame(dict(sender_id=rng.choice(list("ABCDE"), n), upi_app="a", is_fraud=0, timestamp=pd.date_range("2026-01-01", periods=n, freq="min"),
                           decision=np.where(rng.random(n) < 0.003, "REVIEW", "APPROVE")))
    users = pd.DataFrame(dict(user_id=list("ABCDE"), age=[20, 35, 50, 65, 40], city=list("VWXYZ"), upi_app=list("aabbc")))
    r = segment_report(sc, users, min_genuine=300)
    assert r["home_city"]["significant"] is False               # same true rate everywhere -> must not be flagged, whatever the raw ratio


def _good_metrics():
    return dict(models={"LightGBM": {"pr_auc": .9}, "LogisticRegression": {"pr_auc": .6}}, time_series_cv_pr_auc={"mean": .9},
                business_impact_test={"fraud_recall_flagged": .95, "flag_rate": .02},
                fairness={"home_city": {"adverse_impact_ratio": 3.5, "significant": False}, "age_band": {"adverse_impact_ratio": 1.2, "significant": True}},
                calibration={"brier_raw": .003, "brier_calibrated": .003},
                drift_psi_train_vs_test={"f": {"psi": .05}}, rings={"rings_recovered": 7, "true_rings": 8})


def test_quality_gates_pass_and_fail():
    assert all(ok for *_, ok in checks(_good_metrics()))
    gate = "no significant segment disparity > 2x (genuine review/block rate)"
    assert dict((n, ok) for n, _, ok in checks(_good_metrics()))[gate]                       # a noisy 3.5x ratio is ignored
    worse = _good_metrics(); worse["fairness"]["home_city"]["significant"] = True
    assert not dict((n, ok) for n, _, ok in checks(worse))[gate]
    bad = _good_metrics(); bad["models"]["LightGBM"]["pr_auc"] = .6; bad["drift_psi_train_vs_test"]["f"]["psi"] = .4
    failed = [n for n, _, ok in checks(bad) if not ok]
    assert any("PR-AUC" in n for n in failed) and any("PSI" in n for n in failed)


def test_drift_scenario_is_opt_in_and_late():
    base, *_ = generate(n_users=300, n_merchants=40, days=30, fraud_scale=.3, seed=2)
    drift, *_ = generate(n_users=300, n_merchants=40, days=30, fraud_scale=.3, seed=2, drift_scenario=True, drift_start=.7)
    assert "novel_merchant_scam" not in set(base.fraud_type)
    nov = drift[drift.fraud_type == "novel_merchant_scam"]
    assert len(nov) > 0 and (nov.timestamp - drift.timestamp.min().floor("D")).dt.days.min() >= int(30 * .7)


def test_flag_recall_helper():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]); s = np.array([.9, .8, .1, .2, .3, .1, .2, .1, .2, .3])
    typ = np.array(["a", "a"] + ["none"] * 8)
    assert _flag_recall(y, s, typ, rate=0.2)["recall_all"] == 1.0


def _fake_card(n=400):
    rng = np.random.default_rng(0)
    d = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    d.insert(0, "Time", np.sort(rng.uniform(0, 172800, n))); d["Amount"] = rng.uniform(1, 500, n); d["Class"] = (rng.random(n) < .05).astype(int)
    return pd.concat([d, d.iloc[:5]], ignore_index=True)                       # 5 exact duplicates


def test_realdata_loader_validation_and_chronological_split(tmp_path):
    p = tmp_path / "c.csv"; _fake_card().to_csv(p, index=False)
    with pytest.raises(RealDataError, match="unexpected dataset"):
        load(p)                                                                 # strict: not the genuine 284,807-row file
    with pytest.raises(RealDataError, match="not found"):
        load(tmp_path / "missing.csv")
    d, dupes = prepare(load(p, strict=False))
    assert dupes == 5 and not d.duplicated(subset=["Time", "V1", "Amount"]).any()
    assert d[d.split == "train"].Time.max() <= d[d.split == "val"].Time.min() <= d[d.split == "test"].Time.min()
