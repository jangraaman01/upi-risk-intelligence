"""Serving latency benchmark: live (online store) vs point-in-time (history replay) scoring."""
import json
import time

import numpy as np

from .config import REPORTS
from .scoring import RiskEngine


def _pct(x):
    x = np.array(x) * 1000
    return dict(p50_ms=round(float(np.percentile(x, 50)), 2), p95_ms=round(float(np.percentile(x, 95)), 2),
                p99_ms=round(float(np.percentile(x, 99)), 2), mean_ms=round(float(x.mean()), 2))


def run(n_live=1500, n_pit=150, seed=0):
    eng = RiskEngine()
    rng = np.random.default_rng(seed)
    h = eng.history
    pick = h.iloc[rng.integers(0, len(h), n_live)]
    out = {"history_rows": len(h)}

    def bench(rows, **kw):
        lat = []
        for s, r, a, ch in zip(rows.sender_id, rows.receiver_id, rows.amount, rows.channel):
            t0 = time.perf_counter(); eng.score_transaction(s, r, float(a), ch, **kw); lat.append(time.perf_counter() - t0)
        return lat

    bench(pick.iloc[:50])                                                    # warm-up
    out["live_score_only"] = _pct(bench(pick, explain=False))
    out["live_with_shap"] = _pct(bench(pick, explain=True))
    pit_ts = str(h.timestamp.iloc[len(h) // 2])
    out["point_in_time_with_shap"] = _pct(bench(pick.iloc[:n_pit], timestamp=pit_ts, explain=True))
    out["note"] = "single CPU core, in-process (no HTTP overhead); live path = O(1) store lookups + one LightGBM predict"
    json.dump(out, open(REPORTS / "latency.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
