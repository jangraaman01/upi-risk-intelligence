"""Leak-free behavioural features: every value uses only PAST transactions. Batch version (training); `online.py` is the streaming twin."""
import numpy as np
import pandas as pd

from .geo import haversine_km

FEATURES = ["log_amount", "amount_z", "amount_ratio", "log_time_gap",
            "sender_cnt_10m", "sender_cnt_1h", "sender_cnt_24h",
            "recv_cnt_1h", "recv_cnt_24h", "hour", "is_night", "new_receiver",
            "new_device", "is_p2p", "is_collect", "is_round",
            "geo_speed_log", "new_city", "near_limit", "sender_new_rcv_24h", "recv_new_payers_24h",
            "merch_cat", "recv_acct_age_log", "sender_acct_age_log", "dow"]

FRIENDLY = {
    "log_amount": "Transaction amount", "amount_z": "Amount vs. user's normal",
    "amount_ratio": "Amount / user's average", "log_time_gap": "Time since last payment",
    "sender_cnt_10m": "Payments in last 10 min", "sender_cnt_1h": "Payments in last hour",
    "sender_cnt_24h": "Payments in last 24h", "recv_cnt_1h": "Receiver inflow (1h)",
    "recv_cnt_24h": "Receiver inflow (24h)", "hour": "Hour of day",
    "is_night": "Night-time payment", "new_receiver": "First time paying this receiver",
    "new_device": "New device", "is_p2p": "Person-to-person", "is_collect": "Collect request",
    "is_round": "Round amount", "geo_speed_log": "Travel speed since last payment", "new_city": "Payment from a new city",
    "near_limit": "Just under a payment limit", "sender_new_rcv_24h": "New payees added in 24h",
    "recv_new_payers_24h": "Receiver's new payers in 24h", "merch_cat": "Merchant category",
    "recv_acct_age_log": "Receiver account age", "sender_acct_age_log": "Sender account age", "dow": "Day of week"}

_EPOCH = pd.Timestamp("1970-01-01")
_DAY = 86400.0


def near_limit(amt):
    amt = np.asarray(amt, float)
    return (((amt >= 9000) & (amt < 10000)) | ((amt >= 45000) & (amt < 50000)) | ((amt >= 90000) & (amt < 100000))).astype(int)


def _roll_count(ts, groups, window):
    out = np.zeros(len(ts), dtype=np.int32)
    for idx in groups.values():
        t = ts[idx]
        out[idx] = np.arange(len(t)) - np.searchsorted(t, t - window, side="left")
    return out


def _roll_flag_count(ts, groups, flag, window):
    """For each row: number of EARLIER rows in the same group within `window` seconds whose flag is 1."""
    out = np.zeros(len(ts), dtype=np.int32)
    for idx in groups.values():
        t = ts[idx]
        pref = np.r_[0, np.cumsum(flag[idx])]
        out[idx] = pref[np.arange(len(t))] - pref[np.searchsorted(t, t - window, side="left")]
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["timestamp", "txn_id"], kind="stable").reset_index(drop=True)
    ts = (df["timestamp"] - _EPOCH).dt.total_seconds().to_numpy()
    amt = df["amount"].to_numpy(float)
    la = np.log1p(amt)
    lat, lon = df["lat"].to_numpy(float), df["lon"].to_numpy(float)
    n = len(df)
    sg = df.groupby("sender_id").indices
    rg = df.groupby("receiver_id").indices
    nr = (df.groupby(["sender_id", "receiver_id"]).cumcount() == 0).to_numpy()

    z = np.zeros(n); ratio = np.ones(n); gap = np.full(n, 3 * _DAY); n_prior = np.zeros(n, int); speed = np.zeros(n)
    for idx in sg.values():
        x, a, t = la[idx], amt[idx], ts[idx]
        k = np.arange(len(idx))
        with np.errstate(divide="ignore", invalid="ignore"):
            m = (np.cumsum(x) - x) / k
            v = (np.cumsum(x * x) - x * x) / k - m * m
            ma = (np.cumsum(a) - a) / k
            zz = (x - m) / np.maximum(np.sqrt(np.maximum(v, 0)), 0.4)
            rr = a / ma
        zz = np.where(k >= 3, zz, 0.0); rr = np.where(k >= 1, rr, 1.0)
        z[idx] = np.clip(zz, -10, 10); ratio[idx] = np.clip(rr, 0, 50); n_prior[idx] = k
        if len(idx) > 1:
            dt = np.diff(t)
            gap[idx[1:]] = dt
            d = haversine_km(lat[idx][:-1], lon[idx][:-1], lat[idx][1:], lon[idx][1:])
            speed[idx[1:]] = np.minimum(d / np.maximum(dt / 3600.0, 1 / 60), 3000)

    out = pd.DataFrame({"txn_id": df["txn_id"]})
    out["log_amount"] = la; out["amount_z"] = z; out["amount_ratio"] = ratio
    out["log_time_gap"] = np.log1p(gap)
    out["sender_cnt_10m"] = _roll_count(ts, sg, 600)
    out["sender_cnt_1h"] = _roll_count(ts, sg, 3600)
    out["sender_cnt_24h"] = _roll_count(ts, sg, _DAY)
    out["recv_cnt_1h"] = _roll_count(ts, rg, 3600)
    out["recv_cnt_24h"] = _roll_count(ts, rg, _DAY)
    hr = df["timestamp"].dt.hour.to_numpy()
    out["hour"] = hr; out["is_night"] = (hr < 6).astype(int)
    out["new_receiver"] = nr.astype(int)
    first_dev = (df.groupby(["sender_id", "device_id"]).cumcount() == 0).to_numpy()
    out["new_device"] = (first_dev & (n_prior > 0)).astype(int)
    out["is_p2p"] = (df["receiver_type"] == "P2P").astype(int)
    out["is_collect"] = (df["channel"] == "collect").astype(int)
    out["is_round"] = ((amt % 100 == 0) & (amt >= 100)).astype(int)
    out["geo_speed_log"] = np.log1p(speed)
    first_city = (df.groupby(["sender_id", "city"]).cumcount() == 0).to_numpy()
    out["new_city"] = (first_city & (n_prior > 0)).astype(int)
    out["near_limit"] = near_limit(amt)
    out["sender_new_rcv_24h"] = _roll_flag_count(ts, sg, nr.astype(int), _DAY)
    out["recv_new_payers_24h"] = _roll_flag_count(ts, rg, nr.astype(int), _DAY)
    out["merch_cat"] = df["merch_cat"].to_numpy()
    out["recv_acct_age_log"] = np.log1p(df["receiver_acct_age_days"].to_numpy(float))
    out["sender_acct_age_log"] = np.log1p(df["sender_acct_age_days"].to_numpy(float))
    out["dow"] = df["timestamp"].dt.dayofweek.to_numpy()
    return out
