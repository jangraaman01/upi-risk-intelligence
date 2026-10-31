import numpy as np
import pandas as pd
from src.features import FEATURES, build_features
from src.online import OnlineFeatureStore


def test_online_store_matches_batch_features_exactly(small_df):
    """Training/serving parity: streaming features must equal batch features for EVERY row and feature."""
    batch = build_features(small_df).set_index("txn_id")
    online = OnlineFeatureStore.replay(small_df).set_index("txn_id")
    assert list(online.index) == list(batch.index)
    for f in FEATURES:
        np.testing.assert_allclose(online[f].to_numpy(float), batch[f].to_numpy(float), rtol=1e-6, atol=1e-9, err_msg=f)


def _f(store, s, r, amt, ts, rtype="MERCHANT", ch="qr", dev="D1", lat=19.07, lon=72.87, city="Mumbai"):
    return store.features(s, r, amt, ch, dev, ts, rtype, lat, lon, city, 1, 500, 800)


def _u(store, s, r, amt, ts, dev="D1", lat=19.07, lon=72.87, city="Mumbai"):
    store.update(s, r, amt, dev, ts, lat, lon, city)


def test_store_is_read_only_until_update():
    s = OnlineFeatureStore()
    ts = pd.Timestamp("2026-01-01 10:00")
    a, b = _f(s, "U1", "M1", 100.0, ts), _f(s, "U1", "M1", 100.0, ts)
    assert a == b and a["new_receiver"] == 1
    _u(s, "U1", "M1", 100.0, ts)
    c = _f(s, "U1", "M1", 100.0, ts + pd.Timedelta(minutes=5))
    assert c["new_receiver"] == 0 and c["sender_cnt_10m"] == 1 and c["recv_cnt_1h"] == 1


def test_window_trimming_does_not_change_counts():
    s = OnlineFeatureStore(); t0 = pd.Timestamp("2026-01-01")
    for i in range(1000):                                  # 33h of traffic: forces the >256 trimming branch
        _u(s, "U1", "M1", 50.0, t0 + pd.Timedelta(minutes=2 * i))
    f = _f(s, "U1", "M1", 50.0, t0 + pd.Timedelta(minutes=2000))
    assert (f["sender_cnt_10m"], f["sender_cnt_1h"], f["sender_cnt_24h"]) == (5, 30, 720)
    assert (f["recv_cnt_1h"], f["recv_cnt_24h"]) == (30, 720)


def test_geo_speed_new_city_and_fan_in():
    s = OnlineFeatureStore(); t0 = pd.Timestamp("2026-01-01 12:00")
    _u(s, "U1", "M1", 100.0, t0)                                           # Mumbai
    f = _f(s, "U1", "M2", 100.0, t0 + pd.Timedelta(minutes=10), lat=28.61, lon=77.21, city="Delhi")   # ~1,150 km in 10 minutes
    assert f["new_city"] == 1 and np.expm1(f["geo_speed_log"]) > 3000 * 0.99      # capped at 3000 km/h
    for k in range(5):                                                      # five different payers hit one receiver
        _u(s, f"P{k}", "R1", 100.0, t0 + pd.Timedelta(minutes=k))
    assert _f(s, "P9", "R1", 100.0, t0 + pd.Timedelta(minutes=9))["recv_new_payers_24h"] == 5
    for k in range(4):                                                      # one sender adds four new payees
        _u(s, "U7", f"X{k}", 9500.0, t0 + pd.Timedelta(minutes=k))
    g = _f(s, "U7", "X9", 9500.0, t0 + pd.Timedelta(minutes=9), rtype="P2P")
    assert g["sender_new_rcv_24h"] == 4 and g["near_limit"] == 1
