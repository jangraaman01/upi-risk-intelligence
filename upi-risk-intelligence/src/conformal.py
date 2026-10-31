"""Class-conditional split-conformal prediction on calibrated risk scores.

For each class we pick the score quantile that guarantees a chosen coverage (e.g. 90% of true frauds fall inside the 'fraud' set).
A payment whose prediction set contains BOTH classes (or neither) is *uncertain* and is a natural candidate for human review.
"""
import numpy as np


def _q(scores, alpha):
    s = np.sort(np.asarray(scores, float)); n = len(s)
    if n == 0:
        return 1.0
    return float(s[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])


def fit_class_conditional(p, y, alpha_fraud=0.10, alpha_genuine=0.02):
    p, y = np.asarray(p, float), np.asarray(y)
    return dict(alpha_fraud=alpha_fraud, alpha_genuine=alpha_genuine, q_fraud=_q(1 - p[y == 1], alpha_fraud), q_genuine=_q(p[y == 0], alpha_genuine),
                n_fraud=int((y == 1).sum()), n_genuine=int((y == 0).sum()))


def prediction_sets(p, q):
    p = np.asarray(p, float)
    return p <= q["q_genuine"], (1 - p) <= q["q_fraud"]          # (genuine in set, fraud in set)


def evaluate_sets(p, y, q):
    y = np.asarray(y); in0, in1 = prediction_sets(p, q)
    unc = in0 == in1
    return dict(coverage_fraud=round(float(in1[y == 1].mean()), 4), coverage_genuine=round(float(in0[y == 0].mean()), 4),
                share_uncertain=round(float(unc.mean()), 4), share_confident_fraud=round(float((in1 & ~in0).mean()), 4),
                share_confident_genuine=round(float((in0 & ~in1).mean()), 4),
                fraud_in_uncertain=round(float(y[unc].mean()), 4) if unc.any() else 0.0,
                precision_confident_fraud=round(float(y[in1 & ~in0].mean()), 4) if (in1 & ~in0).any() else None)
