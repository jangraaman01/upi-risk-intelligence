"""Cost-sensitive decision policy: choose (review, step-up, block) thresholds that minimise expected INR loss."""
import itertools
import numpy as np
from .config import COSTS

ACTIONS = ["APPROVE", "STEP_UP_AUTH", "REVIEW", "BLOCK_TRANSACTION"]   # increasing severity


def decide_array(p, th):
    s, r, b = th      # (step_up, review, block) thresholds, ascending
    return np.select([p >= b, p >= r, p >= s], ["BLOCK_TRANSACTION", "REVIEW", "STEP_UP_AUTH"], "APPROVE")


def expected_cost(p, y, amt, th, c=COSTS) -> float:
    s, r, b = th
    block = p >= b
    rev = (p >= r) & ~block
    step = (p >= s) & ~block & ~rev
    app = ~(block | step | rev)
    fraud = np.asarray(y) == 1
    cost = amt[app & fraud].sum()
    cost += rev.sum() * c["review_cost"] + amt[rev & fraud].sum() * (1 - c["review_catch"])
    cost += (step.sum() * c["stepup_cost"] + amt[step & fraud].sum() * (1 - c["stepup_catch"])
             + (step & ~fraud).sum() * c["stepup_abandon"] * c["abandon_cost"])
    cost += (block & ~fraud).sum() * c["block_fp_cost"]
    return float(cost)


def optimize_thresholds(p, y, amt, c=COSTS):
    p, y, amt = np.asarray(p), np.asarray(y), np.asarray(amt)
    cand = np.unique(np.r_[np.quantile(p, np.linspace(0.90, 0.9995, 18)), np.geomspace(1e-3, 0.95, 16)])
    best, best_th = np.inf, None
    for s, r, b in itertools.combinations(cand, 3):   # strictly ordered tiers: an analyst queue always exists
        v = expected_cost(p, y, amt, (s, r, b), c)
        if v < best:
            best, best_th = v, (float(s), float(r), float(b))
    return dict(step_up=best_th[0], review=best_th[1], block=best_th[2]), best


def optimize_thresholds_fast(p, y, amt, c=COSTS, n_bins=70):
    """Histogram-binned, fully vectorised version of `optimize_thresholds` (milliseconds instead of seconds) - used for interactive what-if sliders."""
    p, y, amt = np.asarray(p, float), np.asarray(y), np.asarray(amt, float)
    edges = np.unique(np.r_[0.0, np.quantile(p, np.linspace(0.6, 1, n_bins)), np.geomspace(1e-3, 0.99, 30), 1.0001])
    nb = len(edges) - 1
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, nb - 1)
    fraud = y == 1
    n_all = np.bincount(b, minlength=nb); n_leg = np.bincount(b[~fraud], minlength=nb)
    fa = np.bincount(b[fraud], weights=amt[fraud], minlength=nb)
    P = lambda x: np.r_[0, np.cumsum(x)]
    Pn, Pl, Pf = P(n_all), P(n_leg), P(fa)
    idx = np.arange(1, nb)
    I, J, K = np.meshgrid(idx, idx, idx, indexing="ij")
    m = (I < J) & (J < K)
    I, J, K = I[m], J[m], K[m]
    cost = (Pf[I] + (Pn[J] - Pn[I]) * c["stepup_cost"] + (Pf[J] - Pf[I]) * (1 - c["stepup_catch"])
            + (Pl[J] - Pl[I]) * c["stepup_abandon"] * c["abandon_cost"]
            + (Pn[K] - Pn[J]) * c["review_cost"] + (Pf[K] - Pf[J]) * (1 - c["review_catch"])
            + (Pl[-1] - Pl[K]) * c["block_fp_cost"])
    best = int(np.argmin(cost))
    return dict(step_up=float(edges[I[best]]), review=float(edges[J[best]]), block=float(edges[K[best]])), float(cost[best])
