import pandas as pd
import streamlit as st

from ui.common import J, REPORTS, metrics


def need(cmd):
    st.info(f"Not generated yet - run `{cmd}`")


def render():
    st.title("Evidence & Robustness")
    M = metrics()
    from ui.common import world
    if world().get("scale") == "small":
        st.info("This session runs the small simulated world. The bootstrap, multi-seed, backtest and tuning results below ship with the repository and were produced on the medium world (741k payments); the real-data results are independent of the world size.")
    t = st.tabs(["Real-data benchmark", "Uncertainty", "Stability & sensitivity", "Label delay & drift", "Serving latency", "Fairness", "Tuning", "Conformal"])
    with t[0]:
        r = J("real_data.json")
        if r:
            st.caption(f"ULB credit-card fraud: {r['dataset']['rows']:,} txns, {r['dataset']['fraud']} frauds, chronological split, frauds by split {r['dataset']['frauds_by_split']}")
            st.dataframe(pd.DataFrame(r["models"]).T)
            img = REPORTS / "figures/real_data.png"
            if img.exists():
                st.image(str(img))
            a, b = st.columns(2)
            a.subheader("Random vs chronological split"); a.json(r["split_comparison_lgbm_pr_auc"])
            b.subheader("Cost-based alerting"); b.dataframe(pd.DataFrame(r["cost_based_alerting_test"]).T)
        else:
            need("python -m src.realdata --download")
    with t[1]:
        r = J("robustness.json")
        if r:
            c = r["cluster_bootstrap_test"]
            st.caption(f"Cluster bootstrap over senders: {c['n_clusters']:,} clusters, {c['n_fraud']:,} frauds in test, {c['n_boot']} resamples. Point [95% CI].")
            rows = {k: dict(PR_AUC=f"{v['pr_auc']['point']} {v['pr_auc']['ci95']}", Recall_at_rules_flag_rate=f"{v['recall_at_rules_flag_rate']['point']} {v['recall_at_rules_flag_rate']['ci95']}") for k, v in c.items() if isinstance(v, dict) and "pr_auc" in v}
            rows["Hand-written rules"] = dict(PR_AUC="-", Recall_at_rules_flag_rate=f"{c['rules']['recall']['point']} {c['rules']['recall']['ci95']}")
            st.dataframe(pd.DataFrame(rows).T)
            st.json({k: v for k, v in c.items() if k.startswith("paired")})
        else:
            need("python -m src.robustness")
    with t[2]:
        r = J("robustness.json")
        if r and "multi_seed" in r:
            m = r["multi_seed"]; st.subheader("5 independently generated datasets (small preset)"); st.dataframe(pd.DataFrame(m["per_seed"])); st.write("mean", m["mean"]); st.write("std", m["std"])
        if r:
            st.subheader("Cost-assumption sensitivity"); cs = r["cost_sensitivity"]
            st.dataframe(pd.DataFrame({k: v for k, v in cs.items() if not k.startswith("_")}).T[["savings_pct_fixed_thresholds", "savings_pct_reoptimised", "flag_rate_reoptimised"]])
            st.caption(f"Savings range across scenarios (re-optimised thresholds): {cs['_range_savings_pct_reoptimised']}%")
        else:
            need("python -m src.robustness")
    with t[3]:
        r = J("backtest.json")
        if r:
            st.write(r["config"]); st.image(str(REPORTS / "figures/backtest.png")); st.json(r["summary_post_drift"])
            st.dataframe(pd.DataFrame(r["rows"])[["window", "policy", "pr_auc", "recall_all", "recall_novel_merchant_scam", "score_psi_vs_train", "max_feature_psi_vs_train"]])
        else:
            need("python -m src.backtest")
    with t[4]:
        r = J("latency.json")
        if r:
            st.dataframe(pd.DataFrame({k: v for k, v in r.items() if isinstance(v, dict)}).T); st.caption(r["note"])
        else:
            need("python -m src.benchmark")
    with t[5]:
        for dim, v in M.get("fairness", {}).items():
            st.subheader(dim)
            verdict = "SIGNIFICANT" if v["significant"] else "not significant (consistent with noise)"
            st.caption(f"adverse-impact ratio {v['adverse_impact_ratio']} | max gap {v['max_gap_pp']} pp | chi-square p = {v['homogeneity_p_value']} -> {verdict}")
            st.dataframe(pd.DataFrame(v["segments"]).T)
        st.caption("Segment attributes are randomly assigned in the simulation, so no disparity is expected by construction; the audit code is what transfers to real data.")
    with t[6]:
        r = J("tuning.json")
        if r:
            st.json(r)
        else:
            need("python -m src.tune")
    with t[7]:
        c = M.get("conformal")
        if c:
            st.json(c)
        else:
            need("python -m src.train")
