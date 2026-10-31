"""Statistical rigor: cluster-bootstrap CIs, multi-seed stability, and cost-assumption sensitivity.

Run:  python -m src.robustness      (writes reports/robustness.json, ~5 min on one CPU core)
"""
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import COSTS, DATA, LGBM_PARAMS, REPORTS, WARMUP_DAYS
from .costs import decide_array, expected_cost, optimize_thresholds
from .dataset import load_modeling_data
from .features import FEATURES, build_features
from .generate_data import PRESETS, generate
from .logger import get_logger
from .rules import rule_flags

log = get_logger("robustness")


def recall_at_k(y, s, k):
    k = int(max(k, 1))
    idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
    return float(y[idx].sum() / max(y.sum(), 1)), float(y[idx].mean())


def _summ(x, point):
    x = np.asarray(x, float)
    return dict(point=round(float(point), 4), ci95=[round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)])


def cluster_bootstrap(clusters, y, scores: dict, flags, n_boot=400, seed=0):
    """Resample whole SENDERS (fraud arrives in bursts, so rows are not independent) and recompute metrics.
    Returns CIs for PR-AUC, recall at the rules' flag rate (matched), and paired differences vs. the first model."""
    codes, uniq = pd.factorize(clusters)
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes))]
    by = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    names = list(scores)
    ap = {n: [] for n in names}; rec = {n: [] for n in names}; rules_rec, rules_prec = [], []
    for _ in range(n_boot):
        idx = np.concatenate([by[i] for i in rng.integers(0, len(by), len(by))])
        yy = y[idx]
        k = flags[idx].sum()
        if yy.sum() == 0 or k == 0:
            continue
        rules_rec.append(yy[flags[idx]].sum() / yy.sum()); rules_prec.append(yy[flags[idx]].mean())
        for n in names:
            s = scores[n][idx]
            ap[n].append(average_precision_score(yy, s)); rec[n].append(recall_at_k(yy, s, k)[0])
    full_k = flags.sum()
    out = {"n_boot": len(rules_rec), "n_clusters": int(len(uniq)), "n_rows": int(len(y)), "n_fraud": int(y.sum())}
    for n in names:
        out[n] = dict(pr_auc=_summ(ap[n], average_precision_score(y, scores[n])),
                      recall_at_rules_flag_rate=_summ(rec[n], recall_at_k(y, scores[n], full_k)[0]))
    out["rules"] = dict(recall=_summ(rules_rec, y[flags].sum() / y.sum()), precision=_summ(rules_prec, y[flags].mean()), flag_rate=round(float(flags.mean()), 4))
    base = names[0]
    for n in names[1:]:
        d = np.array(ap[base]) - np.array(ap[n])
        out[f"paired_diff_pr_auc_{base}_minus_{n}"] = _summ(d, average_precision_score(y, scores[base]) - average_precision_score(y, scores[n]))
    d = np.array(rec[base]) - np.array(rules_rec)
    out[f"paired_diff_recall_{base}_minus_rules"] = _summ(d, recall_at_k(y, scores[base], full_k)[0] - y[flags].sum() / y.sum())
    return out


def bootstrap_section():
    df, data = load_modeling_data()
    tr, te = data[data.split == "train"], data[data.split == "test"]
    lgb = LGBMClassifier(**LGBM_PARAMS).fit(tr[FEATURES], tr.is_fraud)
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(tr[FEATURES], tr.is_fraud)
    scores = {"LightGBM": lgb.predict_proba(te[FEATURES])[:, 1], "LogisticRegression": lr.predict_proba(te[FEATURES])[:, 1]}
    return cluster_bootstrap(te.sender_id.to_numpy(), te.is_fraud.to_numpy(), scores, rule_flags(te))


def multi_seed_section(seeds=(11, 12, 13, 14, 15)):
    """Regenerate the world from scratch with different seeds: is the result an accident of one random draw?"""
    rows = []
    for sd in seeds:
        t0 = time.time()
        df, *_ = generate(**PRESETS['small'], seed=sd)
        ft = build_features(df)
        d = df.drop(columns=[c for c in ft.columns if c in df.columns and c != "txn_id"]).merge(ft, on="txn_id").sort_values("timestamp", kind="stable")
        d = d[d.timestamp >= d.timestamp.min() + pd.Timedelta(days=WARMUP_DAYS)].reset_index(drop=True)
        n = len(d); tr, te = d.iloc[: int(.75 * n)], d.iloc[int(.75 * n):]       # train+val -> test (chronological)
        lgb = LGBMClassifier(**LGBM_PARAMS).fit(tr[FEATURES], tr.is_fraud)
        lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(tr[FEATURES], tr.is_fraud)
        y, flags = te.is_fraud.to_numpy(), rule_flags(te); k = flags.sum()
        s_l, s_r = lgb.predict_proba(te[FEATURES])[:, 1], lr.predict_proba(te[FEATURES])[:, 1]
        rows.append(dict(seed=sd, fraud_rate=float(d.is_fraud.mean()), test_fraud=int(y.sum()),
                         lgbm_pr_auc=average_precision_score(y, s_l), logreg_pr_auc=average_precision_score(y, s_r),
                         lgbm_recall_at_rules_rate=recall_at_k(y, s_l, k)[0], rules_recall=float(y[flags].sum() / y.sum())))
        log.info("seed %d done in %.0fs: LGBM PR-AUC %.3f | recall %.2f vs rules %.2f", sd, time.time() - t0, rows[-1]["lgbm_pr_auc"],
                 rows[-1]["lgbm_recall_at_rules_rate"], rows[-1]["rules_recall"])
    t = pd.DataFrame(rows)
    return dict(per_seed=rows, mean={c: round(float(t[c].mean()), 4) for c in t.columns if c != "seed"},
                std={c: round(float(t[c].std(ddof=1)), 4) for c in t.columns if c != "seed"})


def cost_sensitivity_section():
    """How fragile is the headline 'savings %' to the cost assumptions I chose? One-at-a-time x0.5 / x2 plus a joint pessimistic case."""
    sc = pd.read_parquet(DATA / "scored.parquet")
    va, te = sc[sc.split == "val"], sc[sc.split == "test"]
    vb = va.iloc[len(va) // 2:]
    art = json.load(open(REPORTS / "metrics.json"))["thresholds"]
    base_th = (art["step_up"], art["review"], art["block"])
    p, y, a = te.risk_score.to_numpy(), te.is_fraud.to_numpy(), te.amount.to_numpy()
    none = float(a[y == 1].sum())

    def scen(costs):
        fixed = expected_cost(p, y, a, base_th, costs)
        th, _ = optimize_thresholds(vb.risk_score.to_numpy(), vb.is_fraud.to_numpy(), vb.amount.to_numpy(), costs)
        reopt_th = (th["step_up"], th["review"], th["block"])
        reopt = expected_cost(p, y, a, reopt_th, costs)
        flagged = float((decide_array(p, reopt_th) != "APPROVE").mean())
        return dict(savings_pct_fixed_thresholds=round(100 * (none - fixed) / none, 1), savings_pct_reoptimised=round(100 * (none - reopt) / none, 1),
                    flag_rate_reoptimised=round(flagged, 4), thresholds_reoptimised={k: round(v, 4) for k, v in th.items()})
    out = {"baseline": scen(COSTS)}
    for k in COSTS:
        if k.endswith("_catch") or k == "stepup_abandon":
            variants = {"-25%": max(COSTS[k] * .75, 0), "+25%": min(COSTS[k] * 1.25, 1.0)} if k.endswith("_catch") else {"x0.5": COSTS[k] * .5, "x2": COSTS[k] * 2}
        else:
            variants = {"x0.5": COSTS[k] * .5, "x2": COSTS[k] * 2}
        for lab, v in variants.items():
            out[f"{k} {lab}"] = scen({**COSTS, k: v})
    out["joint pessimistic"] = scen({**COSTS, "stepup_catch": .5, "review_catch": .8, "review_cost": 50.0, "block_fp_cost": 800.0, "stepup_abandon": .05})
    s = [v["savings_pct_reoptimised"] for v in out.values()]
    out["_range_savings_pct_reoptimised"] = [min(s), max(s)]
    return out


def main():
    t0 = time.time()
    res = {}
    log.info("cluster bootstrap..."); res["cluster_bootstrap_test"] = bootstrap_section()
    log.info("cost sensitivity..."); res["cost_sensitivity"] = cost_sensitivity_section()
    json.dump(res, open(REPORTS / "robustness.json", "w"), indent=2)
    log.info("multi-seed..."); res["multi_seed"] = multi_seed_section()
    json.dump(res, open(REPORTS / "robustness.json", "w"), indent=2)
    log.info("robustness done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
