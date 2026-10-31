"""Segment audit: does the policy burden some customer groups more than others?

Reports, per segment of age band / city / account tenure, the share of GENUINE payments that get REVIEW or BLOCK (the harm metric),
the share getting step-up, and fraud recall (the protection metric), plus an adverse-impact ratio (worst / best genuine flag rate).
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HARM = ("REVIEW", "BLOCK_TRANSACTION")
ALPHA = 0.05 / 4        # Bonferroni across the four audited dimensions


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(float((c - h) / d), 5), round(float((c + h) / d), 5)]


def segment_report(scored: pd.DataFrame, users: pd.DataFrame, min_genuine=300) -> dict:
    d = scored.merge(users[["user_id", "age", "city"]].rename(columns={"user_id": "sender_id", "city": "home_city", "age": "user_age"}),
                     on="sender_id", how="left")
    d["age_band"] = pd.cut(d.user_age, [0, 29, 44, 59, 200], labels=["18-29", "30-44", "45-59", "60+"])
    prior = d.sort_values("timestamp").groupby("sender_id").cumcount()
    d["tenure"] = np.where(prior < 10, "new (<10 prior txns)", "established")
    out = {}
    for dim in ("age_band", "home_city", "tenure", "upi_app"):
        rows, table = {}, []
        for seg, g in d.groupby(dim, observed=True):
            gen, fr = g[g.is_fraud == 0], g[g.is_fraud == 1]
            if len(gen) < min_genuine:
                continue
            k = int(gen.decision.isin(HARM).sum())
            rows[str(seg)] = dict(n=int(len(g)), genuine_review_block_rate=round(k / len(gen), 5), rate_ci95=wilson(k, len(gen)), genuine_flagged=k,
                                  genuine_stepup_rate=round(float((gen.decision == "STEP_UP_AUTH").mean()), 5),
                                  fraud_recall=round(float((fr.decision != "APPROVE").mean()), 3) if len(fr) >= 10 else None, n_fraud=int(len(fr)))
            table.append([k, len(gen) - k])
        r = [v["genuine_review_block_rate"] for v in rows.values()]
        p_val = float(chi2_contingency(np.array(table))[1]) if len(table) > 1 and np.array(table)[:, 0].sum() > 0 else 1.0
        out[dim] = dict(segments=rows, adverse_impact_ratio=round(max(r) / max(min(r), 1e-9), 2) if len(r) > 1 and min(r) > 0 else None,
                        max_gap_pp=round(100 * (max(r) - min(r)), 3) if r else None, homogeneity_p_value=round(p_val, 4),
                        significant=bool(p_val < ALPHA))
    return out
