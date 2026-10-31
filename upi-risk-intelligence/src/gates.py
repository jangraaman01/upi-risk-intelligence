"""Release quality gates: the pipeline fails (CI goes red) if a retrained model regresses below these bars.

Run:  python -m src.gates      (also writes reports/gates.json for the dashboard's System Health page)
"""
import json
import sys
from datetime import datetime, timezone

from .config import REPORTS


def checks(m: dict, latency: dict | None = None):
    bi = m["business_impact_test"]
    yield "hold-out PR-AUC >= 0.85", m["models"]["LightGBM"]["pr_auc"], m["models"]["LightGBM"]["pr_auc"] >= 0.85
    yield "time-series CV PR-AUC >= 0.80", m["time_series_cv_pr_auc"]["mean"], m["time_series_cv_pr_auc"]["mean"] >= 0.80
    yield "beats logistic regression PR-AUC by >= 0.10", round(m["models"]["LightGBM"]["pr_auc"] - m["models"]["LogisticRegression"]["pr_auc"], 3), \
        m["models"]["LightGBM"]["pr_auc"] - m["models"]["LogisticRegression"]["pr_auc"] >= 0.10
    yield "fraud recall on flagged >= 0.90", bi["fraud_recall_flagged"], bi["fraud_recall_flagged"] >= 0.90
    yield "flag rate <= 6% of traffic", bi["flag_rate"], bi["flag_rate"] <= 0.06
    yield "calibration Brier <= 1.10 x raw", m["calibration"]["brier_calibrated"], m["calibration"]["brier_calibrated"] <= 1.10 * m["calibration"]["brier_raw"]
    mx = max(v["psi"] for v in m["drift_psi_train_vs_test"].values())
    yield "max feature PSI < 0.25", round(mx, 3), mx < 0.25
    yield "fraud rings recovered >= 75%", m["rings"]["rings_recovered"], m["rings"]["rings_recovered"] >= 0.75 * m["rings"]["true_rings"]
    if "conformal" in m:
        cov = m["conformal"]["test"]["coverage_fraud"]
        yield "conformal fraud coverage >= 85% (target 90%)", cov, cov >= 0.85
    if "fairness" in m:      # raw ratios are noisy at small n: only a SIGNIFICANT (Bonferroni) AND material (>2x) disparity fails the gate
        worst = max(((v["adverse_impact_ratio"] or 1.0) if v["significant"] else 1.0) for v in m["fairness"].values())
        yield "no significant segment disparity > 2x (genuine review/block rate)", worst, worst <= 2.0
    if latency:
        p99 = latency["live_with_shap"]["p99_ms"]
        yield "live scoring p99 (with SHAP) < 50 ms", p99, p99 < 50


def main():
    m = json.load(open(REPORTS / "metrics.json"))
    try:
        lat = json.load(open(REPORTS / "latency.json"))
    except FileNotFoundError:
        lat = None
    failed, results = 0, []
    for name, val, ok in checks(m, lat):
        print(f"[{'PASS' if ok else 'FAIL'}] {name:55s} {val}")
        results.append(dict(gate=name, value=float(val), passed=bool(ok)))
        failed += not ok
    json.dump(dict(run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), all_passed=failed == 0, gates=results), open(REPORTS / "gates.json", "w"), indent=2)
    print(f"\n{'ALL GATES PASSED' if not failed else str(failed) + ' GATE(S) FAILED'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
