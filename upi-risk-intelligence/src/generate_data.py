"""Synthetic UPI world (v2).

Legitimate behaviour: per-user spend level / activity / peak hour / contacts / favourite merchants, home city + occasional trips,
13 merchant categories (amount and hour profiles), weekend / salary-week / festival calendar effects, devices and UPI apps,
account ages. Hard legitimate look-alikes (big purchases, new phones, bursts, night-shift users, shops receiving P2P,
monthly rent, festival gifting) make fraud non-trivial to separate.

Fraud typologies (each labelled in `fraud_type`):
  account_takeover     SIM-swap style: new device, often night, often from another city, bursts to young mule accounts
  velocity_attack      6-24 small payments to gaming / gift-card / e-commerce merchants within minutes
  collect_scam         older victim approves a large 'collect' request from a mule
  mule_ring            young accounts passing near-round amounts around a cycle
  structuring          many payments just under the Rs 10k / 50k limits to distinct mule accounts within hours
  impossible_travel    payment from a city hundreds of km away minutes after the victim's genuine local payment
  dormant_reactivation inactive (30+ days) account suddenly sends large transfers
  investment_scam      escalating payments over days to a fake 'investment' merchant
Opt-in `drift_scenario` adds a late, novel tactic (`novel_merchant_scam`) for drift / label-delay backtests.
"""
import numpy as np
import pandas as pd

from .config import ROOT, START
from .geo import CITY_LAT, CITY_LON, CITY_NAMES, CITY_P, far_cities

CATS = ["p2p", "grocery", "food_delivery", "fuel", "ecommerce", "utilities", "travel", "entertainment",
        "pharmacy", "education", "gaming", "electronics", "gift_cards", "investment"]
_CAT_P = np.array([.20, .15, .08, .12, .07, .06, .06, .06, .04, .05, .04, .012, .004]); _CAT_P = _CAT_P / _CAT_P.sum()   # categories 1..13
_CAT_MULT = np.array([1.0, .8, .6, 1.5, 1.4, 2.0, 3.0, .7, .7, 4.0, .9, 5.0, 1.2, 3.0])
_APPS = np.array(["app_a", "app_b", "app_c"])

PRESETS = {                      # approx. rows: small ~0.2M, medium ~0.9M, large ~3.5M
    "small": dict(n_users=4000, n_merchants=400, days=60),
    "medium": dict(n_users=12000, n_merchants=700, days=90),
    "large": dict(n_users=40000, n_merchants=1500, days=120),
}


def generate(n_users=4000, n_merchants=400, days=60, fraud_scale=None, seed=7, drift_scenario=False, drift_start=0.7):
    """Returns (transactions, users, merchants). fraud_scale defaults to world size so fraud prevalence stays comparable across presets."""
    rng = np.random.default_rng(seed)
    fs = fraud_scale if fraud_scale is not None else (n_users / 4000) * (days / 60)
    far = far_cities(600)
    nC = len(CITY_NAMES)

    # ---------------- static world ----------------
    uid = np.array([f"U{i:05d}" for i in range(n_users)])
    mid = np.array([f"M{i:04d}" for i in range(n_merchants)])
    age = rng.integers(18, 76, n_users)
    avg_amt = rng.lognormal(6.0, 0.6, n_users)
    rate = rng.gamma(2.0, 0.35, n_users)
    peak = rng.choice([9, 13, 19, 21], n_users)
    home = rng.choice(nC, n_users, p=CITY_P)
    acct_age0 = rng.integers(60, 3000, n_users)
    app = rng.choice(_APPS, n_users, p=[.5, .35, .15])
    dev_os = rng.choice(["android", "ios"], n_users, p=[.8, .2])
    dev_a = np.array([f"D{i:05d}a" for i in range(n_users)])
    dev_b = np.array([f"D{i:05d}b" for i in range(n_users)])
    contacts = rng.integers(0, n_users, (n_users, 8))

    pop = 1 / np.arange(1, n_merchants + 1) ** 0.8; pop /= pop.sum()
    m_cat = rng.choice(np.arange(1, 14), n_merchants, p=_CAT_P)
    m_cat[-10:] = 13; m_cat[-22:-10] = 12            # rarest merchants: 'investment' and 'gift_cards' (scam-prone, almost no genuine traffic)
    m_city = rng.choice(nC, n_merchants, p=CITY_P)
    m_age0 = rng.integers(300, 3500, n_merchants)
    fav = rng.choice(n_merchants, (n_users, 6), p=pop)

    n_mules = max(40, int(150 * n_users / 4000))
    mules = rng.choice(n_users, n_mules, replace=False)
    pool = rng.permutation(np.setdiff1d(np.arange(n_users), mules))
    dormant = rng.random(n_users) < 0.03
    # Account lifecycles (so account age is STATIONARY over time, like real onboarding): mule accounts are opened in waves and live ~75 days,
    # ring accounts are opened shortly before their ring runs, and genuine new customers keep joining. age(day) = acct_age0 + day.
    start_day = np.zeros(n_users, int); end_day = np.full(n_users, 10 ** 6)
    n_periods = max(1, int(np.ceil(days / 30)))
    start_day[mules] = 30 * rng.integers(0, n_periods, len(mules)) + rng.integers(0, 10, len(mules))
    acct_age0[mules] = rng.integers(5, 40, len(mules)) - start_day[mules]; end_day[mules] = start_day[mules] + 75
    n_rings = max(1, int(8 * fs)); ring_day = rng.integers(5, max(6, days - 3), n_rings)
    for rid in range(n_rings):
        nodes = pool[rid * 12: rid * 12 + 10]
        sd = np.maximum(ring_day[rid] - rng.integers(0, 10, len(nodes)), 0)
        start_day[nodes] = sd; acct_age0[nodes] = rng.integers(5, 60, len(nodes)) - sd; end_day[nodes] = sd + 75
    new_cust = (rng.random(n_users) < 0.06) & (end_day == 10 ** 6)
    sd = rng.integers(0, max(1, days - 5), n_users)
    start_day[new_cust] = sd[new_cust]; acct_age0[new_cust] = rng.integers(1, 60, int(new_cust.sum())) - sd[new_cust]

    def active_mules(d):
        a = mules[(start_day[mules] <= d) & (d < start_day[mules] + 45)]
        return a if len(a) else mules
    rate = np.where(dormant, rate * 0.04, rate)
    trip_on = rng.random(n_users) < 0.15
    trip_start = rng.integers(5, max(6, days - 8), n_users); trip_len = rng.integers(2, 7, n_users)
    trip_city = (home + rng.integers(1, nC, n_users)) % nC

    d_idx = np.arange(days); dates = START + pd.to_timedelta(d_idx, unit="D")
    dow = dates.dayofweek.to_numpy(); dom = dates.day.to_numpy()
    w = np.ones(days); w[dow >= 5] *= 1.15; w[dom <= 7] *= 1.25
    fest = int(days * 0.45); w[fest:fest + 5] *= 1.6
    p_day = w / w.sum()

    # ---------------- legitimate traffic ----------------
    counts = rng.poisson(rate * days)
    u = np.repeat(np.arange(n_users), counts); n = len(u)
    day = rng.choice(days, n, p=p_day)
    hour = np.where(rng.random(n) < 0.03, rng.integers(0, 6, n), np.clip(np.round(rng.normal(peak[u], 3.5)), 6, 23)).astype(int)
    is_p2p = rng.random(n) < 0.38
    p2p_r = contacts[u, rng.integers(0, 8, n)]; p2p_r = np.where(p2p_r == u, (u + 1) % n_users, p2p_r)
    m_r = np.where(rng.random(n) < 0.85, fav[u, rng.integers(0, 6, n)], np.where(rng.random(n) < 0.3, rng.integers(0, n_merchants, n), rng.choice(n_merchants, p=pop, size=n)))
    cat = np.where(is_p2p, 0, m_cat[m_r])
    gaming_night = (cat == 10) & (rng.random(n) < 0.3)
    hour = np.where(gaming_night, rng.integers(22, 30, n) % 24, hour)
    ts = START + pd.to_timedelta(day * 86400 + hour * 3600 + rng.integers(0, 3600, n), unit="s")
    in_trip = trip_on[u] & (day >= trip_start[u]) & (day < trip_start[u] + trip_len[u])
    city_t = np.where(in_trip, trip_city[u], home[u])
    receiver = np.where(is_p2p, uid[p2p_r], mid[m_r])
    amount = avg_amt[u] * rng.lognormal(0, 0.5, n) * _CAT_MULT[cat]
    rnd = is_p2p & (rng.random(n) < 0.4)
    amount = np.round(np.clip(np.where(rnd, np.maximum(np.round(amount / 100) * 100, 100), amount), 1, 150000), 2)
    channel = np.where(is_p2p, np.where(rng.random(n) < 0.04, "collect", "p2p"), np.where(rng.random(n) < 0.7, "qr", "intent"))
    r = rng.random(n)
    device = np.where(r < 0.965, dev_a[u], np.where(r < 0.999, dev_b[u], "D_new"))
    device = np.where(device == "D_new", np.char.add("D_new", np.arange(n).astype(str)), device)
    keep = (day >= start_day[u]) & (day < end_day[u])                 # accounts only transact while they exist / are active
    legit = pd.DataFrame(dict(
        timestamp=ts, sender_id=uid[u], receiver_id=receiver, receiver_type=np.where(is_p2p, "P2P", "MERCHANT"), amount=amount, channel=channel,
        device_id=device, sender_age=age[u], city=CITY_NAMES[city_t], lat=CITY_LAT[city_t] + rng.normal(0, .01, n), lon=CITY_LON[city_t] + rng.normal(0, .01, n),
        merchant_category=np.array(CATS)[cat], merch_cat=cat, receiver_acct_age_days=np.maximum(1, np.where(is_p2p, acct_age0[p2p_r], m_age0[m_r]) + day),
        sender_acct_age_days=acct_age0[u] + day, upi_app=app[u], device_os=dev_os[u], is_fraud=0, fraud_type="none", ring_id=-1))
    legit = legit[keep].reset_index(drop=True)

    # ---------------- helpers for injected events ----------------
    rows = []

    def t_at(d, h):
        return START + pd.Timedelta(days=int(d), hours=int(h), minutes=int(rng.integers(0, 60)))

    def mk(t, s, rcv, rtype, amt, ch, dev, label, ftype="none", ring=-1, city=None, cat_=0):
        d = (t - START).days; c = home[s] if city is None else city
        ridx = int(rcv[1:])
        rage = max(1, (acct_age0[ridx] if rtype == "P2P" else m_age0[ridx]) + d)
        rows.append(dict(timestamp=t, sender_id=uid[s], receiver_id=rcv, receiver_type=rtype, amount=round(float(np.clip(amt, 1, 200000)), 2), channel=ch,
                         device_id=dev, sender_age=int(age[s]), city=CITY_NAMES[c], lat=CITY_LAT[c] + rng.normal(0, .01), lon=CITY_LON[c] + rng.normal(0, .01),
                         merchant_category=CATS[cat_], merch_cat=int(cat_), receiver_acct_age_days=int(rage), sender_acct_age_days=int(max(1, acct_age0[s] + d)),
                         upi_app=app[s], device_os=dev_os[s], is_fraud=int(label), fraud_type=ftype, ring_id=ring))

    def pick_mule(v, d):
        act = active_mules(d)
        m = rng.choice(act)
        return m if m != v else (act[0] if act[0] != v else act[-1])

    def rand_merchant(cats):
        return rng.choice(np.where(np.isin(m_cat, cats))[0])

    # ---------------- hard legitimate look-alikes ----------------
    for _ in range(int(900 * fs)):                    # big one-off purchase / fees to a NEW payee
        v = rng.integers(n_users); t = t_at(rng.integers(1, days), rng.integers(9, 22))
        if rng.random() < .5:
            mk(t, v, uid[rng.integers(n_users)], "P2P", avg_amt[v] * rng.uniform(4, 14), "collect" if rng.random() < .3 else "p2p", dev_a[v], 0)
        else:
            mi = rand_merchant([5, 6, 9, 11]); mk(t, v, mid[mi], "MERCHANT", avg_amt[v] * rng.uniform(4, 14), "qr", dev_a[v], 0, cat_=m_cat[mi])
    for i in range(int(250 * fs)):                    # new phone: new device, otherwise normal
        v = rng.integers(n_users); t = t_at(rng.integers(5, days), rng.integers(8, 23))
        for _ in range(rng.integers(2, 6)):
            t += pd.Timedelta(minutes=int(rng.integers(10, 600))); mi = rng.choice(n_merchants, p=pop)
            mk(t, v, mid[mi], "MERCHANT", avg_amt[v] * rng.lognormal(0, .5), "qr", f"D_phone{i}", 0, cat_=m_cat[mi])
    for _ in range(int(300 * fs)):                    # bursts: festival shopping
        v = rng.integers(n_users); t = t_at(rng.integers(2, days), rng.integers(10, 21))
        for _ in range(rng.integers(5, 14)):
            t += pd.Timedelta(seconds=int(rng.integers(20, 240))); mi = rng.choice(n_merchants, p=pop)
            mk(t, v, mid[mi], "MERCHANT", avg_amt[v] * rng.uniform(.2, 1.2), "qr", dev_a[v], 0, cat_=m_cat[mi])
    for _ in range(int(400 * fs)):                    # genuine gift-card and investment purchases (so these categories are not a giveaway)
        v = rng.integers(n_users); mi = rand_merchant([12, 13])
        mk(t_at(rng.integers(1, days), rng.integers(9, 22)), v, mid[mi], "MERCHANT", avg_amt[v] * rng.uniform(1, 4), "intent" if rng.random() < .6 else "qr", dev_a[v], 0, cat_=m_cat[mi])
    for _ in range(int(500 * fs)):                    # night owls
        v = rng.integers(n_users); mi = rng.choice(n_merchants, p=pop)
        mk(t_at(rng.integers(1, days), rng.integers(0, 6)), v, mid[mi], "MERCHANT", avg_amt[v] * rng.lognormal(0, .5), "intent", dev_a[v], 0, cat_=m_cat[mi])
    for _ in range(int(60 * fs)):                     # small shops accepting P2P
        shop = rng.integers(n_users); t0 = t_at(rng.integers(2, days), rng.integers(10, 20))
        for _ in range(rng.integers(8, 25)):
            c = rng.integers(n_users)
            if c != shop:
                mk(t0 + pd.Timedelta(minutes=int(rng.integers(0, 120))), c, uid[shop], "P2P", avg_amt[c] * rng.lognormal(0, .5), "p2p", dev_a[c], 0)
    for v in np.where(rng.random(n_users) < 0.12)[0]:  # monthly rent to a landlord, early in each month
        ll = (v + 1 + rng.integers(0, n_users - 1)) % n_users; amt = round(avg_amt[v] * rng.uniform(8, 20), -2)
        for mstart in range(0, days, 30):
            dd = min(mstart + rng.integers(0, 5), days - 1); mk(t_at(dd, rng.integers(9, 20)), v, uid[ll], "P2P", amt, "p2p", dev_a[v], 0)
    for v in np.where(rng.random(n_users) < 0.02)[0]:  # festival gifting: many small round payments to relatives
        t = t_at(fest + rng.integers(0, 5), rng.integers(9, 18))
        for c in rng.choice(contacts[v], size=int(rng.integers(6, 13)), replace=True):
            if c != v:
                t += pd.Timedelta(minutes=int(rng.integers(3, 20))); mk(t, v, uid[c], "P2P", float(rng.choice([251, 501, 1001, 1100, 2100])), "p2p", dev_a[v], 0)

    # ---------------- fraud ----------------
    for i in range(int(120 * fs)):                    # account takeover (SIM-swap)
        v = rng.integers(n_users); camo = rng.random() < .4
        h = rng.choice([0, 1, 2, 3, 4, 5]) if (not camo and rng.random() < .8) else rng.integers(7, 22)
        d_ = int(rng.integers(3, days)); t = t_at(d_, h); m = pick_mule(v, d_)
        c = rng.choice(far[home[v]]) if rng.random() < (.35 if camo else .65) else home[v]
        dev = dev_a[v] if (camo and rng.random() < .25) else f"D_ato{i}"
        for _ in range(rng.integers(1, 3) if camo else rng.integers(3, 8)):
            t += pd.Timedelta(seconds=int(rng.integers(30, 600)))
            mk(t, v, uid[m], "P2P", avg_amt[v] * rng.lognormal(.2 if camo else 1.3, .5), "p2p", dev, 1, "account_takeover", city=c)
    vel_m = np.where(np.isin(m_cat, [4, 10, 12]))[0]
    for i in range(int(40 * fs)):                     # velocity attack
        v = rng.integers(n_users); t = t_at(rng.integers(3, days), rng.integers(8, 22)); dev = dev_a[v] if rng.random() < .6 else f"D_bot{i}"; slow = rng.random() < .35
        for _ in range(rng.integers(6, 25)):
            t += pd.Timedelta(seconds=int(rng.integers(5, 90) if slow is False else rng.integers(40, 240))); mi = rng.choice(vel_m) if rng.random() < .6 else rng.choice(n_merchants, p=pop)
            mk(t, v, mid[mi], "MERCHANT", avg_amt[v] * rng.uniform(.1, 1.0), "intent", dev, 1, "velocity_attack", cat_=m_cat[mi])
    older = np.where(age >= 45)[0]
    for _ in range(int(180 * fs)):                    # collect-request scam
        v = rng.integers(n_users) if rng.random() < .4 else rng.choice(older); d_ = int(rng.integers(3, days)); t = t_at(d_, rng.integers(10, 20)); m = pick_mule(v, d_) if rng.random() < .7 else rng.integers(n_users)           # 30%: paid to an ordinary (old) account
        if m == v:
            continue
        mk(t, v, uid[m], "P2P", avg_amt[v] * rng.uniform(1.5 if m not in mules else 2.5, 14), "collect", dev_a[v], 1, "collect_scam")
        if rng.random() < .4:
            mk(t + pd.Timedelta(minutes=int(rng.integers(5, 30))), v, uid[m], "P2P", avg_amt[v] * rng.uniform(3, 8), "collect", dev_a[v], 1, "collect_scam")
    for rid in range(n_rings):                        # mule rings
        k = rng.integers(5, 11); nodes = pool[rid * 12: rid * 12 + k]
        if len(nodes) < 5:
            break
        t0 = t_at(ring_day[rid], rng.integers(0, 24)); tidy = rng.random() < .7; span = rng.uniform(24, 60) if tidy else rng.uniform(60, 110); base = rng.choice([5000, 9000, 15000, 25000])
        for _ in range(rng.integers(40, 100)):
            i = rng.integers(k); j = (i + 1) % k if rng.random() < .85 else rng.integers(k)
            j = (i + 1) % k if j == i else j
            mk(t0 + pd.Timedelta(seconds=float(rng.uniform(0, span * 3600))), nodes[i], uid[nodes[j]], "P2P", (round(base * rng.uniform(.9, 1.1), -1) if tidy else round(base * rng.uniform(.7, 1.3), 2)),
               "p2p", dev_a[nodes[i]], 1, "mule_ring", ring=rid)
    for i in range(int(25 * fs)):                     # structuring just under the limit
        d_ = int(rng.integers(3, days)); v = rng.choice(active_mules(d_)) if rng.random() < .6 else rng.integers(n_users); thr = 50000 if rng.random() < .25 else 10000; camo = rng.random() < .3
        t = t_at(d_, rng.integers(8, 22)); dev = dev_a[v] if rng.random() < .5 else f"D_str{i}"
        for m in rng.choice(active_mules(d_), size=rng.integers(8, 21), replace=True):
            if m == v:
                continue
            t += pd.Timedelta(seconds=int(rng.integers(1200, 3600) if camo else rng.integers(120, 900)))
            mk(t, v, uid[m], "P2P", thr * (rng.uniform(.75, .94) if camo else rng.uniform(.90, .999)), "p2p", dev, 1, "structuring")
    ecom = np.where(np.isin(m_cat, [5, 11, 12]))[0]
    for i in range(int(30 * fs)):                     # impossible travel
        v = rng.integers(n_users); t = t_at(rng.integers(3, days), rng.integers(10, 18)); mi = rng.choice(n_merchants, p=pop)
        mk(t, v, mid[mi], "MERCHANT", avg_amt[v] * rng.lognormal(0, .4), "qr", dev_a[v], 0, cat_=m_cat[mi])   # genuine local payment (anchor)
        t += pd.Timedelta(minutes=int(rng.integers(3, 26) if rng.random() < .7 else rng.integers(60, 180))); c = rng.choice(far[home[v]])
        for _ in range(rng.integers(2, 5)):
            t += pd.Timedelta(seconds=int(rng.integers(30, 400))); mi = rng.choice(ecom)
            mk(t, v, mid[mi], "MERCHANT", avg_amt[v] * rng.uniform(2, 6), "intent", f"D_it{i}", 1, "impossible_travel", city=c, cat_=m_cat[mi])
    dorm_ids = np.where(dormant)[0]
    dts = legit[legit.sender_id.isin(uid[dorm_ids])].groupby("sender_id").timestamp.apply(lambda s: np.sort(s.to_numpy()))
    for i in range(int(25 * fs)):                     # dormant account reactivation
        v = rng.choice(dorm_ids); past = dts.get(uid[v], np.array([], dtype="datetime64[ns]"))
        for _ in range(8):
            t = t_at(rng.integers(int(days * .35), days), rng.integers(9, 21)); t64 = t.to_datetime64(); before = past[past < t64]
            if len(before) == 0 or (t64 - before.max()) > np.timedelta64(30, "D"):
                m = pick_mule(v, (t - START).days); dev = dev_a[v] if rng.random() < .5 else f"D_dor{i}"
                for _ in range(rng.integers(1, 4)):
                    t += pd.Timedelta(minutes=int(rng.integers(2, 30))); mk(t, v, uid[m], "P2P", avg_amt[v] * rng.uniform(1.5 if rng.random() < .4 else 3, 10), "p2p", dev, 1, "dormant_reactivation")
                break
    inv_m = np.where(m_cat == 13)[0]; mid_age = np.where((age >= 30) & (age <= 70))[0]
    for _ in range(int(35 * fs)):                     # investment scam: escalating payments over days
        v = rng.choice(mid_age); mi = rng.choice(inv_m); d0 = rng.integers(3, max(4, days - 10)); amt = avg_amt[v] * rng.uniform(1, 3)
        for k in range(rng.integers(4, 10)):
            mk(t_at(min(d0 + k + rng.integers(0, 2), days - 1), rng.integers(10, 20)), v, mid[mi], "MERCHANT", min(amt, 80000), "intent", dev_a[v], 1, "investment_scam", cat_=13)
            amt *= rng.uniform(1.2, 1.6)
    if drift_scenario:                                # NOVEL tactic appearing late
        cand7 = np.where(m_cat == 7)[0]
        scam_m = rng.choice(cand7 if len(cand7) else np.arange(n_merchants), min(15, len(cand7) or n_merchants), replace=False)
        for _ in range(int(150 * fs)):
            v = rng.integers(n_users); t = t_at(rng.integers(int(days * drift_start), days), rng.integers(10, 20))
            for _ in range(rng.integers(1, 4)):
                t += pd.Timedelta(minutes=int(rng.integers(2, 20))); mi = rng.choice(scam_m)
                mk(t, v, mid[mi], "MERCHANT", avg_amt[v] * rng.uniform(2, 6), "intent", dev_a[v], 1, "novel_merchant_scam", cat_=7)

    df = pd.concat([legit, pd.DataFrame(rows)], ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])
    users = pd.DataFrame(dict(user_id=uid, age=age, city=CITY_NAMES[home], home_lat=CITY_LAT[home], home_lon=CITY_LON[home], avg_amt=avg_amt.round(2),
                              acct_age_days=acct_age0, is_mule=np.isin(np.arange(n_users), mules), is_dormant=dormant, upi_app=app, device_os=dev_os))
    merchants = pd.DataFrame(dict(merchant_id=mid, category=np.array(CATS)[m_cat], merch_cat=m_cat, city=CITY_NAMES[m_city], acct_age_days=m_age0, popularity=pop))
    return df, users, merchants


if __name__ == "__main__":
    df, users, merchants = generate()
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_parquet(ROOT / "data/transactions.parquet", index=False); users.to_parquet(ROOT / "data/users.parquet", index=False)
    merchants.to_parquet(ROOT / "data/merchants.parquet", index=False)
    print(f"{len(df):,} transactions | fraud rate {df.is_fraud.mean():.2%}")
    print(df[df.is_fraud == 1].fraud_type.value_counts().to_string())
