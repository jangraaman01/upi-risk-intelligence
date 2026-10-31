"""Population Stability Index (PSI) for feature drift. Rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 drift."""
import numpy as np
import pandas as pd


def psi(expected, actual, bins=10) -> float:
    e, a = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if e.nunique() <= 10:
        cats = sorted(set(e.unique()) | set(a.unique()))
        pe = np.array([(e == c).mean() for c in cats]); pa = np.array([(a == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pe = np.histogram(e, edges)[0] / len(e); pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_report(train: pd.DataFrame, live: pd.DataFrame, features) -> dict:
    out = {f: round(psi(train[f], live[f]), 4) for f in features}
    return {k: dict(psi=v, status="stable" if v < .1 else "watch" if v < .25 else "DRIFT") for k, v in out.items()}


def robust_anomalies(s: pd.Series, window: int = 14, z: float = 3.5) -> pd.DataFrame:
    """Flag points that deviate from the trailing median by more than `z` robust sigmas (MAD). Baseline uses PAST values only (no look-ahead)."""
    past = s.shift(1)
    med = past.rolling(window, min_periods=7).median()
    mad = (past - med).abs().rolling(window, min_periods=7).median()
    sigma = 1.4826 * mad.clip(lower=1e-9)
    score = (s - med) / sigma
    return pd.DataFrame({"value": s, "baseline": med, "score": score, "is_anomaly": score.abs() > z})
