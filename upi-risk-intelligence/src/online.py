"""Online (streaming) feature store.

`build_features` (batch) is used for training; serving needs the SAME numbers in milliseconds without re-scanning history.
This class keeps O(1)-sized running state per sender / receiver and is verified to match the batch code exactly
(see tests/test_online.py) - the classic guard against training/serving skew.
"""
import math
from bisect import bisect_left

import numpy as np
import pandas as pd

from .features import near_limit
from .geo import haversine_km

_EPOCH = pd.Timestamp("1970-01-01")
_DAY = 86400.0


def _sec(ts) -> float:
    return (pd.Timestamp(ts) - _EPOCH).total_seconds()


def _trim(lst, t):
    if len(lst) > 256:                          # keep only what a 24h window can still need
        del lst[: bisect_left(lst, t - _DAY)]


class OnlineFeatureStore:
    def __init__(self):
        self.snd: dict = {}                 # sender -> running stats + recent timestamps + last position
        self.rcv: dict = {}                 # receiver -> recent inbound timestamps
        self.rnr: dict = {}                 # receiver -> timestamps at which a NEW payer paid it
        self.pairs: set = set()             # (sender, receiver) seen
        self.devs: set = set()              # (sender, device) seen
        self.cities: set = set()            # (sender, city) seen
        self.max_ts = -math.inf

    # ---------- read-only feature computation ----------
    def features(self, sender, receiver, amount, channel, device, ts, receiver_type, lat, lon, city, merch_cat, recv_age, sender_age) -> dict:
        t, la = _sec(ts), math.log1p(amount)
        st = self.snd.get(sender)
        n = st["n"] if st else 0
        z, ratio, gap, speed = 0.0, 1.0, 3 * _DAY, 0.0
        if n >= 1:
            ratio = float(np.clip(amount / (st["sa"] / n), 0, 50))
            gap = t - st["last"]
            d = float(haversine_km(st["lat"], st["lon"], lat, lon))
            speed = min(d / max(gap / 3600.0, 1 / 60), 3000.0)
        if n >= 3:
            m = st["s1"] / n
            v = max(st["s2"] / n - m * m, 0.0)
            z = float(np.clip((la - m) / max(math.sqrt(v), 0.4), -10, 10))
        s_times = st["times"] if st else []
        s_nr = st["nr"] if st else []
        r_times = self.rcv.get(receiver, [])
        r_nr = self.rnr.get(receiver, [])
        cnt = lambda lst, w: len(lst) - bisect_left(lst, t - w)
        hour = pd.Timestamp(ts).hour
        return dict(log_amount=la, amount_z=z, amount_ratio=ratio, log_time_gap=math.log1p(gap),
                    sender_cnt_10m=cnt(s_times, 600), sender_cnt_1h=cnt(s_times, 3600), sender_cnt_24h=cnt(s_times, _DAY),
                    recv_cnt_1h=cnt(r_times, 3600), recv_cnt_24h=cnt(r_times, _DAY),
                    hour=hour, is_night=int(hour < 6), new_receiver=int((sender, receiver) not in self.pairs),
                    new_device=int(n > 0 and (sender, device) not in self.devs), is_p2p=int(receiver_type == "P2P"),
                    is_collect=int(channel == "collect"), is_round=int(amount % 100 == 0 and amount >= 100),
                    geo_speed_log=math.log1p(speed), new_city=int(n > 0 and (sender, city) not in self.cities),
                    near_limit=int(near_limit([amount])[0]), sender_new_rcv_24h=cnt(s_nr, _DAY), recv_new_payers_24h=cnt(r_nr, _DAY),
                    merch_cat=int(merch_cat), recv_acct_age_log=math.log1p(recv_age), sender_acct_age_log=math.log1p(sender_age),
                    dow=pd.Timestamp(ts).dayofweek)

    def last_device(self, sender):
        st = self.snd.get(sender)
        return st["dev"] if st else None

    def last_city(self, sender):
        st = self.snd.get(sender)
        return st["city"] if st else None

    def has_sender(self, sender): return sender in self.snd
    def has_receiver(self, receiver): return receiver in self.rcv

    # ---------- state update (commit a transaction) ----------
    def update(self, sender, receiver, amount, device, ts, lat, lon, city):
        t, la = _sec(ts), math.log1p(amount)
        st = self.snd.setdefault(sender, dict(n=0, s1=0.0, s2=0.0, sa=0.0, last=None, dev=None, city=None, lat=0.0, lon=0.0, times=[], nr=[]))
        new_pair = (sender, receiver) not in self.pairs
        st["n"] += 1; st["s1"] += la; st["s2"] += la * la; st["sa"] += amount
        st["last"], st["dev"], st["city"], st["lat"], st["lon"] = t, device, city, lat, lon
        st["times"].append(t); _trim(st["times"], t)
        rl = self.rcv.setdefault(receiver, []); rl.append(t); _trim(rl, t)
        if new_pair:
            st["nr"].append(t); _trim(st["nr"], t)
            rn = self.rnr.setdefault(receiver, []); rn.append(t); _trim(rn, t)
        self.pairs.add((sender, receiver)); self.devs.add((sender, device)); self.cities.add((sender, city))
        self.max_ts = max(self.max_ts, t)

    # ---------- bulk helpers ----------
    @classmethod
    def from_history(cls, df: pd.DataFrame) -> "OnlineFeatureStore":
        store = cls()
        d = df.sort_values(["timestamp", "txn_id"], kind="stable")
        for s, r, a, dev, ts, la_, lo_, c in zip(d.sender_id, d.receiver_id, d.amount, d.device_id, d.timestamp, d.lat, d.lon, d.city):
            store.update(s, r, float(a), dev, ts, float(la_), float(lo_), c)
        return store

    @classmethod
    def replay(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Stream a dataframe through the store (features computed BEFORE each update) - for parity testing."""
        store, rows = cls(), []
        d = df.sort_values(["timestamp", "txn_id"], kind="stable")
        for r_ in d.itertuples(index=False):
            rows.append({"txn_id": r_.txn_id, **store.features(r_.sender_id, r_.receiver_id, float(r_.amount), r_.channel, r_.device_id, r_.timestamp,
                                                               r_.receiver_type, float(r_.lat), float(r_.lon), r_.city, r_.merch_cat,
                                                               r_.receiver_acct_age_days, r_.sender_acct_age_days)})
            store.update(r_.sender_id, r_.receiver_id, float(r_.amount), r_.device_id, r_.timestamp, float(r_.lat), float(r_.lon), r_.city)
        return pd.DataFrame(rows)

    @property
    def max_timestamp(self):
        return None if self.max_ts == -math.inf else _EPOCH + pd.Timedelta(seconds=self.max_ts)
