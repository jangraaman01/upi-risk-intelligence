import numpy as np
import pandas as pd
from src.calibration import apply_calibrator, fit_calibrator
from src.costs import decide_array, expected_cost, optimize_thresholds
from src.graph import detect_rings
from src.monitoring import psi


def test_expected_cost_hand_computed():
    p, y, amt = np.array([0.9, 0.1]), np.array([1, 0]), np.array([1000.0, 100.0])
    c = dict(stepup_cost=3, stepup_catch=.75, stepup_abandon=.02, abandon_cost=50, review_cost=25, review_catch=.95, block_fp_cost=400)
    # fraud (p=.9) is blocked -> 0 ; genuine (p=.1) gets step-up -> 3 + .02*50 = 4
    assert expected_cost(p, y, amt, (0.05, 0.5, 0.8), c) == 4.0
    # approving everything loses the fraud amount
    assert expected_cost(p, y, amt, (0.95, 0.96, 0.97), c) == 1000.0


def test_decide_array_tiers():
    out = decide_array(np.array([0.0, 0.2, 0.5, 0.9]), (0.1, 0.4, 0.8))
    assert list(out) == ["APPROVE", "STEP_UP_AUTH", "REVIEW", "BLOCK_TRANSACTION"]


def test_optimizer_returns_ordered_thresholds():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.8 * y * rng.random(3000) + 0.1 * rng.random(3000), 0, 1)
    th, cost = optimize_thresholds(p, y, rng.uniform(100, 5000, 3000))
    assert th["step_up"] < th["review"] < th["block"] and cost >= 0


def test_calibration_is_monotonic():
    rng = np.random.default_rng(1)
    raw = rng.random(2000); y = (rng.random(2000) < raw ** 3).astype(int)
    cal = fit_calibrator(raw, y)
    out = apply_calibrator(cal, np.linspace(0.01, 0.99, 50))
    assert (np.diff(out) >= 0).all() and out.min() >= 0 and out.max() <= 1


def test_psi_detects_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.1
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_ring_detection_finds_cycle_and_ignores_chain():
    t0 = pd.Timestamp("2026-01-05 10:00")
    rows = []
    for k in range(4):                                  # A->B->C->A repeated: a ring
        for s, r in [("A", "B"), ("B", "C"), ("C", "A")]:
            rows.append((t0 + pd.Timedelta(minutes=10 * len(rows)), s, r))
    for k in range(4):                                  # X->Y->Z repeated but never returns: not a ring
        for s, r in [("X", "Y"), ("Y", "Z")]:
            rows.append((t0 + pd.Timedelta(minutes=10 * len(rows)), s, r))
    df = pd.DataFrame(rows, columns=["timestamp", "sender_id", "receiver_id"]).assign(receiver_type="P2P", amount=100.0)
    rings = detect_rings(df)
    assert len(rings) == 1 and rings[0]["members"] == ["A", "B", "C"]
