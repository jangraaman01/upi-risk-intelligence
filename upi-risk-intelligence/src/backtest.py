"""Label-delay & concept-drift backtest: how fast does a model decay when fraudsters change tactics, and what does retraining buy
when fraud labels only arrive `delay_days` late (chargebacks / customer reports)?

Policies (weekly evaluation windows):
  static           trained once on the first labelled period, never updated
  delayed_retrain  retrained weekly, but may only use labels older than `delay_days`   <- what production can actually do
  oracle_retrain   retrained weekly with instantly-available labels                     <- unattainable upper bound
A NEW fraud tactic ('novel_merchant_scam') appears two-thirds of the way through.

Run:  python -m src.backtest      (~6 min on one CPU core)
"""
import json
import time

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from .config import LGBM_PARAMS, REPORTS, WARMUP_DAYS
from .features import FEATURES, build_features
from .generate_data import PRESETS, generate
from .logger import get_logger
from .monitoring import psi, psi_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger("backtest")
P = {**LGBM_PARAMS, "n_estimators": 200}


def _flag_recall(y, s, typ, rate=0.01):
    k = max(1, int(rate * len(y))); thr = np.sort(s)[-k]
    flag = s >= thr
    out = dict(flag_rate=round(float(flag.mean()), 4), recall_all=round(float(flag[y == 1].mean()), 3) if y.sum() else None)
    for t in ("novel_merchant_scam", "account_takeover", "velocity_attack", "collect_scam", "mule_ring"):
        m = typ == t
        out[f"recall_{t}"] = round(float(flag[m].mean()), 3) if m.sum() >= 5 else None
        out[f"n_{t}"] = int(m.sum())
    return out


def run(days=84, seed=21, delay_days=7, step_days=7, first_eval=35):
    t0 = time.time()
    df, *_ = generate(**{**PRESETS['small'], 'days': days}, seed=seed, drift_scenario=True, drift_start=2 / 3)
    ft = build_features(df)
    d = df.drop(columns=[c for c in ft.columns if c in df.columns and c != "txn_id"]).merge(ft, on="txn_id").sort_values("timestamp", kind="stable")
    start = d.timestamp.min().floor("D")
    d["day"] = (d.timestamp - start).dt.days
    d = d[d.day >= WARMUP_DAYS].reset_index(drop=True)
    drift_day = int(days * 2 / 3)
    log.info("generated %d txns; novel tactic starts day %d", len(d), drift_day)

    def fit(lo, hi):
        tr = d[(d.day >= lo) & (d.day < hi)]
        return LGBMClassifier(**P).fit(tr[FEATURES], tr.is_fraud), tr

    static_model, static_tr = fit(WARMUP_DAYS, first_eval - delay_days)
    rows = []
    for w0 in range(first_eval, days, step_days):
        w1 = min(w0 + step_days, days)
        te = d[(d.day >= w0) & (d.day < w1)]
        y, typ = te.is_fraud.to_numpy(), te.fraud_type.to_numpy()
        models = {"static": static_model, "delayed_retrain": fit(WARMUP_DAYS, w0 - delay_days)[0], "oracle_retrain": fit(WARMUP_DAYS, w0)[0]}
        for name, m in models.items():
            s = m.predict_proba(te[FEATURES])[:, 1]
            r3 = _flag_recall(y, s, typ, rate=0.03)["recall_novel_merchant_scam"]
            rows.append(dict(window=f"d{w0}-{w1}", start_day=w0, policy=name, pr_auc=round(float(average_precision_score(y, s)), 4), novel_recall_at_3pct_budget=r3, **_flag_recall(y, s, typ),
                             score_psi_vs_train=round(psi(static_model.predict_proba(static_tr[FEATURES])[:, 1], s), 4) if name == "static" else None))
        ps = psi_report(static_tr, te, FEATURES)
        top = max(ps, key=lambda k: ps[k]["psi"])
        rows[-3]["max_feature_psi_vs_train"] = round(ps[top]["psi"], 4)
        rows[-3]["top_psi_feature"] = top
        rows[-3]["max_psi_excl_new_receiver"] = round(max(v["psi"] for k, v in ps.items() if k != "new_receiver"), 4)
        log.info("window %d-%d done (%.0fs elapsed)", w0, w1, time.time() - t0)
    res = dict(config=dict(days=days, delay_days=delay_days, step_days=step_days, first_eval=first_eval, drift_day=drift_day, flag_rate_budget=0.01),
               rows=rows)
    t = pd.DataFrame(rows)
    post = t[t.start_day >= drift_day]
    res["summary_post_drift"] = {p: dict(mean_pr_auc=round(float(g.pr_auc.mean()), 4), mean_novel_recall=round(float(g.recall_novel_merchant_scam.dropna().mean()), 3),
                                         mean_novel_recall_3pct_budget=round(float(g.novel_recall_at_3pct_budget.dropna().mean()), 3))
                                 for p, g in post.groupby("policy")}
    json.dump(res, open(REPORTS / "backtest.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for p, c in zip(["static", "delayed_retrain", "oracle_retrain"], ["#d62728", "#1f77b4", "#2ca02c"]):
        g = t[t.policy == p]
        ax[0].plot(g.start_day, g.pr_auc, "o-", color=c, label=p)
        ax[1].plot(g.start_day, g.recall_novel_merchant_scam.astype(float), "o-", color=c, label=p)
    for a, ttl, yl in [(ax[0], "PR-AUC by week", "PR-AUC"), (ax[1], "Recall on the NEW tactic (1% alert budget)", "recall")]:
        a.axvline(drift_day, color="k", ls="--"); a.set(title=ttl, xlabel="evaluation window start (day)", ylabel=yl); a.legend(fontsize=7)
    g = t[t.policy == "static"]
    ax[2].plot(g.start_day, g.score_psi_vs_train, "o-", label="score PSI"); ax[2].plot(g.start_day, g.max_feature_psi_vs_train, "s-", label="max feature PSI (new_receiver)")
    ax[2].plot(g.start_day, g.max_psi_excl_new_receiver, "^-", label="max PSI of all other features")
    ax[2].axhline(0.25, color="r", ls=":", label="drift alarm (0.25)"); ax[2].axvline(drift_day, color="k", ls="--")
    ax[2].set(title="What drift monitoring sees (static model)", xlabel="window start (day)", ylabel="PSI"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(REPORTS / "figures/backtest.png", dpi=130); plt.close(fig)
    log.info("backtest done in %.0fs | %s", time.time() - t0, res["summary_post_drift"])
    return res


if __name__ == "__main__":
    run()
