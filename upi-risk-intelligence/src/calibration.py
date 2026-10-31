"""Platt scaling on the log-odds of the raw model score so risk scores behave like probabilities."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


def fit_calibrator(raw, y):
    return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw), y)


def apply_calibrator(cal, raw):
    return cal.predict_proba(_logit(raw))[:, 1]
