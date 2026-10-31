import numpy as np
import pandas as pd
import pytest
from src.cases import CaseStore
from src.conformal import evaluate_sets, fit_class_conditional
from src.costs import expected_cost, optimize_thresholds, optimize_thresholds_fast
from src.generate_data import generate
from src.geo import CITY_INDEX, CITY_LAT, CITY_LON, far_cities, haversine_km
from src.monitoring import robust_anomalies


def test_conformal_coverage_holds_out_of_sample():
    rng = np.random.default_rng(0)
    def draw(n):
        y = (rng.random(n) < .1).astype(int)
        p = np.where(y == 1, rng.beta(4, 2, n), rng.beta(1, 12, n))
        return p, y
    pc, yc = draw(20000); pt, yt = draw(20000)
    q = fit_class_conditional(pc, yc, alpha_fraud=0.10, alpha_genuine=0.02)
    ev = evaluate_sets(pt, yt, q)
    assert ev["coverage_fraud"] >= 0.88 and ev["coverage_genuine"] >= 0.97          # finite-sample guarantee, small tolerance
    assert 0 < ev["share_uncertain"] < 0.5


def test_case_store_roundtrip(tmp_path):
    cs = CaseStore(tmp_path / "c.db")
    cs.resolve("T1", "confirmed_fraud", risk_score=.9, amount=1000.0); cs.resolve("T2", "false_positive"); cs.resolve("T3", "confirmed_fraud")
    assert cs.resolved_ids() == {"T1", "T2", "T3"}
    s = cs.stats()
    assert s["resolved"] == 3 and s["confirmed_fraud"] == 2 and s["review_precision"] == pytest.approx(2 / 3, abs=1e-3)
    cs.resolve("T2", "escalated")                                                      # re-resolving replaces, never duplicates
    assert cs.stats()["resolved"] == 3
    with pytest.raises(ValueError):
        cs.resolve("T9", "maybe")


def test_fast_optimizer_matches_exact_within_tolerance():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < .03).astype(int)
    p = np.clip(.02 + .85 * y * rng.random(20000) + .1 * rng.random(20000), 0, 1)
    a = rng.uniform(100, 8000, 20000)
    th_f, c_f = optimize_thresholds_fast(p, y, a)
    th_e, c_e = optimize_thresholds(p, y, a)
    assert th_f["step_up"] < th_f["review"] < th_f["block"]
    assert expected_cost(p, y, a, (th_f["step_up"], th_f["review"], th_f["block"])) <= 1.10 * c_e


def test_robust_anomalies_flags_spike_without_lookahead():
    s = pd.Series(np.r_[np.random.default_rng(0).normal(10, .5, 40), 25.0, np.random.default_rng(1).normal(10, .5, 10)], index=pd.date_range("2026-01-01", periods=51))
    a = robust_anomalies(s)
    assert bool(a.is_anomaly.iloc[40]) and a.is_anomaly.iloc[10:40].sum() <= 2
    s2 = s.copy(); s2.iloc[45:] = 999                                                    # changing the FUTURE must not change earlier baselines
    assert np.allclose(robust_anomalies(s2).baseline.iloc[:45].fillna(0), a.baseline.iloc[:45].fillna(0))


def test_geo_distances():
    d = haversine_km(CITY_LAT[CITY_INDEX["Mumbai"]], CITY_LON[CITY_INDEX["Mumbai"]], CITY_LAT[CITY_INDEX["Delhi"]], CITY_LON[CITY_INDEX["Delhi"]])
    assert 1100 < d < 1250
    far = far_cities(600)
    assert all(len(f) > 0 for f in far) and CITY_INDEX["Mumbai"] not in far[CITY_INDEX["Mumbai"]]


def test_v2_world_has_typologies_and_genuine_anchor_before_impossible_travel():
    df, users, merchants = generate(n_users=600, n_merchants=80, days=40, fraud_scale=.8, seed=5)
    types = set(df.fraud_type) - {"none"}
    assert len(types) >= 7 and {"structuring", "impossible_travel", "investment_scam"} <= types
    assert {"city", "lat", "lon", "merch_cat", "receiver_acct_age_days"} <= set(df.columns) and len(merchants) == 80 and users.is_mule.sum() >= 40
    it = df[df.fraud_type == "impossible_travel"].iloc[0]
    prior = df[(df.sender_id == it.sender_id) & (df.is_fraud == 0) & (df.timestamp < it.timestamp) & (df.timestamp > it.timestamp - pd.Timedelta(hours=4))]
    assert len(prior) >= 1 and (prior.city != it.city).any()                           # genuine local payment minutes earlier, different city
    young = df[(df.fraud_type == "mule_ring")].receiver_acct_age_days
    assert young.median() < df[df.fraud_type == "none"].receiver_acct_age_days.median()   # fraud receivers are younger accounts on average
