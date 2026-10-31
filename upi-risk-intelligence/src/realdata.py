"""Real-data benchmark: ULB / Worldline credit-card fraud (284,807 txns, 492 frauds, 48h, PCA-anonymised features).

Why a second dataset? Public real data has no account/device identifiers, so the behavioural UPI system cannot be built on it.
It CAN validate the *modelling methodology* (chronological split, imbalance, calibration, cost-based thresholds, uncertainty).

Data: Kaggle 'mlg-ulb/creditcardfraud' (Database Contents License). Put creditcard.csv in data/external/ or run:
    python -m src.realdata --download      (fetches a public GitHub mirror and verifies 284,807 rows / 492 frauds)
"""
import json
import sys
import urllib.request

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import apply_calibrator, fit_calibrator
from .config import DATA, REPORTS, SEED
from .logger import get_logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger("realdata")
PATH = DATA / "external" / "creditcard.csv"
MIRROR = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
EXPECTED_ROWS, EXPECTED_FRAUD = 284_807, 492
V = [f"V{i}" for i in range(1, 29)]
FEATS = V + ["log_amount", "hour_sin", "hour_cos"]
LGBM = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0, random_state=SEED, verbose=-1)


class RealDataError(ValueError):
    pass


def download():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading mirror -> %s", PATH)
    urllib.request.urlretrieve(MIRROR, PATH)


def load(path=PATH, strict=True) -> pd.DataFrame:
    if not path.exists():
        raise RealDataError(f"{path} not found - download from Kaggle (mlg-ulb/creditcardfraud) or run `python -m src.realdata --download`")
    df = pd.read_csv(path)
    need = {"Time", "Amount", "Class", *V}
    if need - set(df.columns):
        raise RealDataError(f"missing columns: {sorted(need - set(df.columns))}")
    if strict and (len(df) != EXPECTED_ROWS or int(df.Class.sum()) != EXPECTED_FRAUD):
        raise RealDataError(f"unexpected dataset: {len(df)} rows / {int(df.Class.sum())} frauds (expected {EXPECTED_ROWS}/{EXPECTED_FRAUD})")
    return df


def prepare(df: pd.DataFrame):
    """De-duplicate (exact duplicate rows would straddle splits), order by time, add features, chronological 60/15/25 split."""
    n_dupes = int(df.duplicated().sum())
    d = df.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)
    d["log_amount"] = np.log1p(d.Amount)
    hr = (d.Time / 3600.0) % 24
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    n = len(d)
    d["split"] = np.where(np.arange(n) < .60 * n, "train", np.where(np.arange(n) < .75 * n, "val", "test"))
    return d, n_dupes


def recall_at_alert_rate(y, s, rate):
    k = max(1, int(rate * len(y)))
    idx = np.argsort(-s)[:k]
    return float(y[idx].sum() / y.sum())


def value_recall_at_alert_rate(y, s, amt, rate):
    k = max(1, int(rate * len(y)))
    idx = np.argsort(-s)[:k]
    return float(amt[idx][y[idx] == 1].sum() / amt[y == 1].sum())


def _boot(y, scores: dict, n=500, seed=0):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in scores}; diff = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        a = {k: average_precision_score(y[i], s[i]) for k, s in scores.items()}
        for k in a:
            out[k].append(a[k])
        diff.append(a["LightGBM"] - a["Logistic regression"])
    ci = lambda x, p: dict(point=round(float(p), 4), ci95=[round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)])
    res = {k: ci(v, average_precision_score(y, scores[k])) for k, v in out.items()}
    res["paired_diff_lgbm_minus_logreg"] = ci(diff, average_precision_score(y, scores["LightGBM"]) - average_precision_score(y, scores["Logistic regression"]))
    return res


def main():
    REPORTS.joinpath("figures").mkdir(parents=True, exist_ok=True)
    raw = load()
    d, n_dupes = prepare(raw)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    log.info("rows %d (dropped %d duplicates) | frauds train/val/test = %d/%d/%d", len(d), n_dupes, tr.Class.sum(), va.Class.sum(), te.Class.sum())
    y_te, a_te = te.Class.to_numpy(), te.Amount.to_numpy()

    lgb = LGBMClassifier(**LGBM).fit(tr[FEATS], tr.Class)
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)).fit(tr[FEATS], tr.Class)
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(tr[FEATS])
    S = {"LightGBM": lgb.predict_proba(te[FEATS])[:, 1], "Logistic regression": lr.predict_proba(te[FEATS])[:, 1],
         "Isolation Forest (unsup.)": -iso.score_samples(te[FEATS])}
    rates = [0.0005, 0.001, 0.005]
    res = dict(dataset=dict(rows=int(len(d)), duplicates_dropped=n_dupes, fraud=int(d.Class.sum()), fraud_rate=round(float(d.Class.mean()), 5),
                            frauds_by_split={s: int(g.Class.sum()) for s, g in d.groupby("split")}, hours=round(float(d.Time.max() / 3600), 1)))
    res["models"] = {k: dict(pr_auc=round(average_precision_score(y_te, s), 4), roc_auc=round(roc_auc_score(y_te, s), 4),
                             **{f"recall@{r:.2%}_alerts": round(recall_at_alert_rate(y_te, s, r), 3) for r in rates},
                             **{f"value_recall@{r:.2%}_alerts": round(value_recall_at_alert_rate(y_te, s, a_te, r), 3) for r in rates}) for k, s in S.items()}
    res["bootstrap_pr_auc_test"] = _boot(y_te, {k: S[k] for k in ("LightGBM", "Logistic regression")})
    res["bootstrap_pr_auc_test"]["note"] = "row-level bootstrap (no account ids in this dataset, so clustering is unknown); n_fraud_test=%d" % int(y_te.sum())

    # random split vs chronological split on REAL data
    rtr, rte = train_test_split(d, test_size=0.25, stratify=d.Class, random_state=0)
    m = LGBMClassifier(**LGBM).fit(rtr[FEATS], rtr.Class)
    ctr, cte = d[d.split != "test"], te
    mt = LGBMClassifier(**LGBM).fit(ctr[FEATS], ctr.Class)
    res["split_comparison_lgbm_pr_auc"] = dict(random_stratified=round(average_precision_score(rte.Class, m.predict_proba(rte[FEATS])[:, 1]), 4),
                                               chronological=round(average_precision_score(cte.Class, mt.predict_proba(cte[FEATS])[:, 1]), 4))

    # calibration (Platt on validation)
    cal = fit_calibrator(lgb.predict_proba(va[FEATS])[:, 1], va.Class)
    p_cal = apply_calibrator(cal, S["LightGBM"])
    res["calibration"] = dict(brier_raw=round(brier_score_loss(y_te, S["LightGBM"]), 6), brier_calibrated=round(brier_score_loss(y_te, p_cal), 6),
                              brier_no_skill=round(brier_score_loss(y_te, np.full(len(y_te), tr.Class.mean())), 6))

    # cost-based alerting: missed fraud costs its Amount, each alert costs c (analyst time). Threshold chosen on validation only.
    sv, sa = lgb.predict_proba(va[FEATS])[:, 1], S["LightGBM"]
    cand = np.unique(np.r_[np.quantile(sv, np.linspace(0.90, 0.99995, 120)), np.geomspace(1e-4, .99, 60)])
    cost_rows = {}
    for c in (1.0, 5.0, 25.0):
        v_cost = [float(va.Amount[(sv < t) & (va.Class == 1)].sum() + c * (sv >= t).sum()) for t in cand]
        t = cand[int(np.argmin(v_cost))]
        flag = sa >= t
        loss_none = float(a_te[y_te == 1].sum())
        loss = float(a_te[(~flag) & (y_te == 1)].sum() + c * flag.sum())
        cost_rows[f"alert_cost={c:g}"] = dict(
            alert_rate=round(float(flag.mean()), 5), recall=round(float(y_te[flag].sum() / y_te.sum()), 3),
            precision=round(float(y_te[flag].mean()) if flag.any() else 0.0, 3),
            loss_without_model=round(loss_none, 1), loss_with_model=round(loss, 1), savings_pct=round(100 * (loss_none - loss) / loss_none, 1))
    res["cost_based_alerting_test"] = cost_rows

    contrib = lgb.booster_.predict(te[FEATS].sample(min(5000, len(te)), random_state=1), pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), FEATS).sort_values(ascending=False)
    res["top_features_mean_abs_shap"] = {k: round(float(v), 4) for k, v in imp.head(8).items()}

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, s in S.items():
        pr, rc, _ = precision_recall_curve(y_te, s); ax[0].plot(rc, pr, label=f"{k} (AP={res['models'][k]['pr_auc']:.2f})")
    ax[0].axhline(y_te.mean(), color="gray", ls=":", label="no skill")
    ax[0].set(title="Real data - precision-recall (chronological test)", xlabel="recall", ylabel="precision"); ax[0].legend(fontsize=7)
    xs = np.array([0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01])
    for k, s in S.items():
        ax[1].plot(xs * 100, [recall_at_alert_rate(y_te, s, r) for r in xs], "o-", label=k)
    ax[1].set(title="Fraud caught vs. alert budget", xlabel="% of transactions alerted", ylabel="recall", xscale="log"); ax[1].legend(fontsize=7)
    top = imp.head(10)[::-1]; ax[2].barh(top.index, top.values); ax[2].set_title("Mean |SHAP| (anonymised features)")
    fig.tight_layout(); fig.savefig(REPORTS / "figures/real_data.png", dpi=130); plt.close(fig)
    json.dump(res, open(REPORTS / "real_data.json", "w"), indent=2)
    log.info("real-data benchmark done | LightGBM AUPRC %.3f | LogReg %.3f", res["models"]["LightGBM"]["pr_auc"], res["models"]["Logistic regression"]["pr_auc"])
    return res


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    main()
