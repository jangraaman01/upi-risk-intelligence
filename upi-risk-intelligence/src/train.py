"""End-to-end training pipeline.

validate -> features -> time split -> train (LightGBM + baselines) -> Platt calibration -> cost-optimal thresholds
-> evaluate (hold-out, time-series CV, drift, rings, mule ranking, business impact) -> artifacts + figures.
"""
import datetime as dt
import json
import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import apply_calibrator, fit_calibrator
from .config import COSTS, DATA, LGBM_PARAMS, MODEL_VERSION, MODELS, REPORTS, SEED, WARMUP_DAYS
from .conformal import evaluate_sets, fit_class_conditional
from .costs import ACTIONS, decide_array, expected_cost, optimize_thresholds
from .dataset import load_modeling_data
from .fairness import segment_report
from .features import FEATURES, FRIENDLY
from .graph import detect_rings, evaluate_mule_ranking, evaluate_rings, rank_mule_candidates
from .logger import get_logger
from .monitoring import psi_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger("train")


def main():
    for d in (MODELS, REPORTS / "figures", DATA):
        d.mkdir(parents=True, exist_ok=True)
    df, data = load_modeling_data()
    users = pd.read_parquet(DATA / "users.parquet")
    log.info("loaded %d transactions (fraud %.2f%%)", len(df), 100 * df.is_fraud.mean())
    n = len(data)
    tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
    va_a, va_b = va.iloc[: len(va) // 2], va.iloc[len(va) // 2:]       # calibrate on va_a, tune thresholds on va_b
    log.info("split train=%d val=%d test=%d", len(tr), len(va), len(te))

    # ---- models ----
    lgbm = LGBMClassifier(**LGBM_PARAMS).fit(tr[FEATURES], tr.is_fraud)
    logreg = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(tr[FEATURES], tr.is_fraud)
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(tr[FEATURES])
    raw_te = lgbm.predict_proba(te[FEATURES])[:, 1]
    scores = {"LightGBM": raw_te, "LogisticRegression": logreg.predict_proba(te[FEATURES])[:, 1],
              "IsolationForest (unsupervised)": -iso.score_samples(te[FEATURES])}
    metrics = {"model_version": MODEL_VERSION, "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "data": dict(rows=n, raw_rows=len(df), users=int(df.sender_id.nunique()), merchants=int(df.loc[df.receiver_type == "MERCHANT", "receiver_id"].nunique()), days=int((df.timestamp.max() - df.timestamp.min()).days) + 1, fraud_rate=round(float(data.is_fraud.mean()), 4), warmup_days=WARMUP_DAYS, test_fraud_rate=round(float(te.is_fraud.mean()), 4)),
               "models": {k: dict(roc_auc=round(roc_auc_score(te.is_fraud, v), 4), pr_auc=round(average_precision_score(te.is_fraud, v), 4))
                          for k, v in scores.items()}}

    # ---- time-series cross-validation (expanding window, train+val only) ----
    dev = data[data.split != "test"]
    cv = []
    for a, b in TimeSeriesSplit(n_splits=4).split(dev):
        m = LGBMClassifier(**LGBM_PARAMS).fit(dev.iloc[a][FEATURES], dev.iloc[a].is_fraud)
        cv.append(average_precision_score(dev.iloc[b].is_fraud, m.predict_proba(dev.iloc[b][FEATURES])[:, 1]))
    metrics["time_series_cv_pr_auc"] = dict(folds=[round(float(x), 4) for x in cv], mean=round(float(np.mean(cv)), 4), std=round(float(np.std(cv)), 4))

    # ---- calibration ----
    cal = fit_calibrator(lgbm.predict_proba(va_a[FEATURES])[:, 1], va_a.is_fraud)
    p_va_b = apply_calibrator(cal, lgbm.predict_proba(va_b[FEATURES])[:, 1])
    p_te = apply_calibrator(cal, raw_te)
    edges = [0, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 1.0001]      # probability bins (deciles are useless at ~1% base rate)
    bins = pd.cut(pd.Series(p_te), edges, labels=False, right=False)
    rel = pd.DataFrame(dict(p=p_te, y=te.is_fraud.to_numpy(), b=bins)).groupby("b").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    metrics["calibration"] = dict(brier_raw=round(brier_score_loss(te.is_fraud, raw_te), 5),
                                  brier_calibrated=round(brier_score_loss(te.is_fraud, p_te), 5),
                                  reliability=rel.round(5).reset_index().to_dict("list"))

    # ---- conformal prediction: coverage-guaranteed prediction sets, uncertain -> human ----
    conf_q = fit_class_conditional(p_va_b, va_b.is_fraud.to_numpy())
    metrics["conformal"] = dict(params=conf_q, test=evaluate_sets(p_te, te.is_fraud.to_numpy(), conf_q))

    # ---- cost-optimal thresholds + business impact ----
    th, val_cost = optimize_thresholds(p_va_b, va_b.is_fraud.to_numpy(), va_b.amount.to_numpy())
    ths = (th["step_up"], th["review"], th["block"])
    y_te, a_te = te.is_fraud.to_numpy(), te.amount.to_numpy()
    cost_model = expected_cost(p_te, y_te, a_te, ths)
    cost_none = float(a_te[y_te == 1].sum())
    dec_te = decide_array(p_te, ths)
    flagged = dec_te != "APPROVE"
    metrics["thresholds"] = th
    metrics["business_impact_test"] = dict(
        fraud_loss_without_model_inr=round(cost_none, 2), total_cost_with_model_inr=round(cost_model, 2),
        net_savings_inr=round(cost_none - cost_model, 2), savings_pct=round(100 * (cost_none - cost_model) / cost_none, 1),
        flag_rate=round(float(flagged.mean()), 4), fraud_recall_flagged=round(float(flagged[y_te == 1].mean()), 4),
        precision_flagged=round(float(y_te[flagged].mean()), 4), cost_assumptions=COSTS)
    metrics["test_decisions"] = {d: dict(count=int((dec_te == d).sum()), fraud_in_bucket=int(y_te[dec_te == d].sum())) for d in ACTIONS}
    typ = te.fraud_type.to_numpy()
    metrics["recall_by_fraud_type"] = {t: round(float(flagged[typ == t].mean()), 3) for t in sorted(set(typ[y_te == 1]))}
    metrics["fraud_by_type_test"] = {t: int((typ == t).sum()) for t in sorted(set(typ[y_te == 1]))}

    # ---- explainability (global) ----
    sample = te[FEATURES].sample(min(5000, len(te)), random_state=1)
    contrib = lgbm.booster_.predict(sample, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), FEATURES).sort_values(ascending=False)
    metrics["mean_abs_shap"] = {k: round(float(v), 4) for k, v in imp.items()}

    ss = pd.concat([g.sample(min(len(g), 500), random_state=1) for _, g in te[te.is_fraud == 1].groupby("fraud_type")] + [te[te.is_fraud == 0].sample(min(5000, int((te.is_fraud == 0).sum())), random_state=1)])
    sc = lgbm.booster_.predict(ss[FEATURES], pred_contrib=True)
    shap_df = pd.concat([ss[["txn_id", "fraud_type", "is_fraud"] + FEATURES].reset_index(drop=True),
                         pd.DataFrame(sc[:, :-1], columns=[f"shap_{f}" for f in FEATURES]), pd.DataFrame({"shap_base": sc[:, -1]})], axis=1)
    shap_df.to_parquet(DATA / "shap_sample.parquet", index=False)

    # ---- drift ----
    metrics["drift_psi_train_vs_test"] = psi_report(tr, te, FEATURES)

    # ---- score everything + graph analytics ----
    raw_all = lgbm.predict_proba(data[FEATURES])[:, 1]
    data["risk_score"] = apply_calibrator(cal, raw_all)
    data["decision"] = decide_array(data.risk_score.to_numpy(), ths)
    rings = detect_rings(df)
    metrics["rings"] = evaluate_rings(df, rings)
    cands = rank_mule_candidates(data[data.split == "test"])
    truth = set(users[users.is_mule].user_id) | {m for r in df[df.ring_id >= 0].groupby("ring_id") for m in set(r[1].sender_id) | set(r[1].receiver_id)}
    metrics["mule_ranking"] = evaluate_mule_ranking(cands, truth, 20)
    metrics["fairness"] = segment_report(data[data.split == "test"], users)

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, v in scores.items():
        pr, rc, _ = precision_recall_curve(te.is_fraud, v)
        ax[0].plot(rc, pr, label=f"{k} (AP={metrics['models'][k]['pr_auc']:.2f})")
    ax[0].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision"); ax[0].legend(fontsize=7)
    ax[1].plot([1e-4, 1], [1e-4, 1], "k--", lw=1, label="perfect")
    ax[1].plot(rel.predicted.clip(lower=1e-4), rel.observed.clip(lower=1e-4), "o-", label="model")
    ax[1].set(title="Reliability (calibrated, log-log)", xlabel="Predicted probability", ylabel="Observed fraud rate", xscale="log", yscale="log"); ax[1].legend(fontsize=7)
    top = imp.head(10)[::-1]; ax[2].barh([FRIENDLY[i] for i in top.index], top.values)
    ax[2].set(title="Mean |SHAP| (top 10)"); ax[2].tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(REPORTS / "figures/model_overview.png", dpi=130); plt.close(fig)

    # ---- artifacts ----
    json.dump(rings, open(DATA / "rings.json", "w"), indent=1)
    cands.to_parquet(DATA / "mule_candidates.parquet", index=False)
    json.dump(metrics, open(REPORTS / "metrics.json", "w"), indent=2)
    joblib.dump(dict(lgbm=lgbm, calibrator=cal, thresholds=th, conformal=conf_q, features=FEATURES, version=MODEL_VERSION,
                     trained_at=metrics["trained_at"], costs=COSTS), MODELS / "artifacts.joblib")
    f32 = data.select_dtypes("float64").columns
    data[f32] = data[f32].astype("float32")                      # halve dashboard memory at large scale
    data.to_parquet(DATA / "scored.parquet", index=False)
    log.info("done | PR-AUC %.3f | savings %.1f%% | rings %s | mule P@20 %.2f", metrics["models"]["LightGBM"]["pr_auc"],
             metrics["business_impact_test"]["savings_pct"], metrics["rings"], metrics["mule_ranking"]["precision_at_k"])
    return metrics


if __name__ == "__main__":
    main()
