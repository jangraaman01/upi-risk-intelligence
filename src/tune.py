"""Hyper-parameter search (Optuna, TPE) with TIME-AWARE CV - and an honest check of whether tuning actually helped.

Run:  python -m src.tune        (~4 min on one CPU core)
"""
import json

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from .config import LGBM_PARAMS, REPORTS, SEED
from .dataset import load_modeling_data
from .features import FEATURES
from .logger import get_logger

log = get_logger("tune")


def paired_cluster_bootstrap(clusters, y, a, b, n=300, seed=0):
    import pandas as pd
    codes, uniq = pd.factorize(clusters)
    order = np.argsort(codes, kind="stable"); bounds = np.r_[0, np.cumsum(np.bincount(codes))]
    by = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        idx = np.concatenate([by[i] for i in rng.integers(0, len(by), len(by))])
        if y[idx].sum():
            diffs.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    return [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]


def main(n_trials=10, search_rows=300_000):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _, data = load_modeling_data()
    dev, te = data[data.split != "test"], data[data.split == "test"]
    search = dev.iloc[-search_rows:]                      # search on the most recent rows only (speed at scale); the final comparison uses all of `dev`
    folds = list(TimeSeriesSplit(n_splits=3).split(search))

    def objective(t):
        p = dict(LGBM_PARAMS, num_leaves=t.suggest_int("num_leaves", 8, 64), learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                 n_estimators=t.suggest_int("n_estimators", 100, 400, step=100), min_child_samples=t.suggest_int("min_child_samples", 10, 100),
                 subsample=t.suggest_float("subsample", 0.6, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        sc = [average_precision_score(search.iloc[b].is_fraud, LGBMClassifier(**p).fit(search.iloc[a][FEATURES], search.iloc[a].is_fraud).predict_proba(search.iloc[b][FEATURES])[:, 1])
              for a, b in folds]
        return float(np.mean(sc))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    fixed = {k: LGBM_PARAMS[k] for k in ("num_leaves", "learning_rate", "n_estimators", "min_child_samples", "subsample", "colsample_bytree")}
    default_cv = objective(optuna.trial.FixedTrial({**fixed, "reg_lambda": 0.01}))
    study.optimize(objective, n_trials=n_trials)
    best = dict(LGBM_PARAMS, **study.best_params)
    y = te.is_fraud.to_numpy()
    s_def = LGBMClassifier(**LGBM_PARAMS).fit(dev[FEATURES], dev.is_fraud).predict_proba(te[FEATURES])[:, 1]
    s_tun = LGBMClassifier(**best).fit(dev[FEATURES], dev.is_fraud).predict_proba(te[FEATURES])[:, 1]
    res = dict(n_trials=n_trials, cv_pr_auc=dict(default_params=round(default_cv, 4), best_tuned=round(study.best_value, 4)), best_params=study.best_params,
               test_pr_auc=dict(default=round(float(average_precision_score(y, s_def)), 4), tuned=round(float(average_precision_score(y, s_tun)), 4)))
    res["test_paired_diff_tuned_minus_default_ci95"] = paired_cluster_bootstrap(te.sender_id.to_numpy(), y, s_tun, s_def)
    lo, hi = res["test_paired_diff_tuned_minus_default_ci95"]
    res["conclusion"] = ("no statistically significant test gain (CV improvement did not transfer) -> keep defaults" if lo <= 0 <= hi
                         else "tuned model significantly better on test" if lo > 0 else "tuned model significantly WORSE on test -> keep defaults")
    json.dump(res, open(REPORTS / "tuning.json", "w"), indent=2)
    log.info("tuning done: %s", res["test_pr_auc"])
    return res


if __name__ == "__main__":
    main()
