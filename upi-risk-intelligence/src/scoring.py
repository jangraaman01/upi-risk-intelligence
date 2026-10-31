"""RiskEngine: shared by the FastAPI backend and the Streamlit dashboard."""
import pickle

import numpy as np
import pandas as pd

import joblib

from .calibration import apply_calibrator
from .config import DATA, MODELS, START
from .features import FEATURES, FRIENDLY, build_features
from .geo import CITY_INDEX, CITY_LAT, CITY_LON
from .logger import get_logger
from .online import OnlineFeatureStore
from .validation import validate_transactions

log = get_logger("engine")
_COLS = ["txn_id", "timestamp", "sender_id", "receiver_id", "receiver_type", "amount", "channel", "device_id",
         "city", "lat", "lon", "merch_cat", "receiver_acct_age_days", "sender_acct_age_days"]


def load_artifacts():
    return joblib.load(MODELS / "artifacts.joblib")


def explain_frame(model, X: pd.DataFrame, top=5):
    """SHAP contributions (log-odds of the raw model) for the first row of X."""
    c = model.booster_.predict(X[FEATURES].iloc[:1], pred_contrib=True)[0][:-1]
    return [dict(feature=FEATURES[i], label=FRIENDLY[FEATURES[i]], value=float(X[FEATURES[i]].iloc[0]), contribution=round(float(c[i]), 3))
            for i in np.argsort(-np.abs(c))[:top]]


class RiskEngine:
    def __init__(self):
        art = load_artifacts()
        self.model, self.calibrator, self.th = art["lgbm"], art["calibrator"], art["thresholds"]
        self.conformal = art.get("conformal")
        self.version, self.trained_at = art["version"], art["trained_at"]
        self.history = validate_transactions(pd.read_parquet(DATA / "transactions.parquet"))
        self.users = pd.read_parquet(DATA / "users.parquet").set_index("user_id")
        self.merchants = pd.read_parquet(DATA / "merchants.parquet").set_index("merchant_id")
        self._by_s = self.history.groupby("sender_id").indices
        self._by_r = self.history.groupby("receiver_id").indices
        self.store = self._load_store()
        log.info("engine ready: model v%s, %d history rows, store up to %s", self.version, len(self.history), self.store.max_timestamp)

    def _load_store(self) -> OnlineFeatureStore:
        """Streaming state is rebuilt from history (slow for big worlds) and snapshotted; the snapshot is reused while the history is unchanged."""
        snap, key = DATA / "store.pkl", (len(self.history), str(self.history.timestamp.max()))
        if snap.exists():
            try:
                saved = pickle.load(open(snap, "rb"))
                if saved["key"] == key:
                    return saved["store"]
            except Exception:                                          # corrupt / incompatible snapshot -> rebuild
                pass
        store = OnlineFeatureStore.from_history(self.history)
        pickle.dump(dict(key=key, store=store), open(snap, "wb"), protocol=5)
        return store

    def decide(self, p):
        t = self.th
        return ("BLOCK_TRANSACTION" if p >= t["block"] else "REVIEW" if p >= t["review"] else "STEP_UP_AUTH" if p >= t["step_up"] else "APPROVE")

    def score_features(self, X: pd.DataFrame):
        return apply_calibrator(self.calibrator, self.model.predict_proba(X[FEATURES])[:, 1])

    def explain(self, X: pd.DataFrame, top=5):
        return explain_frame(self.model, X, top)

    # ---------- attribute lookups ----------
    def _receiver_attrs(self, receiver_id, ts):
        d = (ts - START).days
        if receiver_id.startswith("M"):
            if receiver_id in self.merchants.index:
                m = self.merchants.loc[receiver_id]
                return "MERCHANT", int(m.merch_cat), int(m.acct_age_days) + d
            return "MERCHANT", 1, 365 + d
        return "P2P", 0, int(self.users.acct_age_days.get(receiver_id, 365)) + d

    def _city_xy(self, city):
        i = CITY_INDEX.get(city, 0)
        return float(CITY_LAT[i]), float(CITY_LON[i])

    def score_transaction(self, sender_id, receiver_id, amount, channel="p2p", device_id=None, timestamp=None, city=None, commit=False, explain=True):
        """Score one payment.

        live mode          : timestamp omitted or >= latest ingested event -> O(1) lookups in the OnlineFeatureStore (milliseconds).
        point-in-time mode : timestamp earlier than the latest event -> features recomputed from history strictly before that
                             timestamp (no look-ahead); slower, used for replays / what-if analysis.
        commit=True records the payment into the live state (as a real gateway would after processing it).
        """
        st = self.store
        ts = pd.Timestamp(timestamp) if timestamp else (st.max_timestamp or self.history.timestamp.max()) + pd.Timedelta(minutes=1)
        rtype, mcat, rage = self._receiver_attrs(receiver_id, ts)
        warnings = []
        if not st.has_sender(sender_id):
            warnings.append("unknown sender: cold-start scoring (no behavioural history)")
        if not st.has_receiver(receiver_id):
            warnings.append("unknown receiver")
        live = st.max_timestamp is None or ts >= st.max_timestamp
        if live:
            last_dev, last_city = st.last_device(sender_id), st.last_city(sender_id)
        else:                                  # point-in-time: defaults come from the sender's state AT that timestamp, not today's
            h0 = self.history.iloc[self._by_s.get(sender_id, np.array([], int))]
            h0 = h0[h0.timestamp < ts]
            last_dev = h0.device_id.iloc[-1] if len(h0) else None
            last_city = h0.city.iloc[-1] if len(h0) else None
        dev = device_id or last_dev or "D_unknown"
        city = city or last_city or (self.users.city.get(sender_id) if sender_id in self.users.index else None) or "Mumbai"
        lat, lon = self._city_xy(city)
        s_age = int(self.users.acct_age_days.get(sender_id, 365)) + (ts - START).days
        if live:
            row = pd.DataFrame([st.features(sender_id, receiver_id, float(amount), channel, dev, ts, rtype, lat, lon, city, mcat, rage, s_age)])
        else:
            warnings.append("point-in-time replay: timestamp precedes latest ingested event")
            h = self.history
            s_idx, r_idx = self._by_s.get(sender_id, np.array([], int)), self._by_r.get(receiver_id, np.array([], int))
            sub = h.iloc[np.union1d(s_idx, r_idx).astype(int)]
            sub = sub[sub.timestamp < ts]
            new = pd.DataFrame([dict(txn_id="~NEW", timestamp=ts, sender_id=sender_id, receiver_id=receiver_id, receiver_type=rtype, amount=float(amount),
                                     channel=channel, device_id=dev, city=city, lat=lat, lon=lon, merch_cat=mcat, receiver_acct_age_days=rage, sender_acct_age_days=s_age)])
            f = build_features(pd.concat([sub[_COLS], new], ignore_index=True))
            row = f[f.txn_id == "~NEW"]
        p = float(self.score_features(row)[0])
        if commit:
            st.update(sender_id, receiver_id, float(amount), dev, ts, lat, lon, city)
        return dict(risk_score=round(p, 5), decision=self.decide(p), reasons=self.explain(row) if explain else [], warnings=warnings,
                    model_version=self.version, timestamp=str(ts), mode="live" if live else "point_in_time", city=city)

    def what_if(self, sender_id, receiver_id, amount, channel="p2p", device_id=None, timestamp=None, city=None):
        """Counterfactual analysis: re-score the payment under realistic alternatives (smaller amount, usual device, home city, ...).
        Exact (it re-runs the real feature pipeline), but descriptive only - it does not claim causality about fraudsters."""
        base_args = dict(sender_id=sender_id, receiver_id=receiver_id, amount=float(amount), channel=channel, device_id=device_id, timestamp=timestamp, city=city)
        base = self.score_transaction(**base_args, explain=False)
        ts = pd.Timestamp(base["timestamp"])
        scenarios = [(f"Amount x{f}", dict(amount=amount * f)) for f in (0.75, 0.5, 0.25, 0.1)]
        usual_dev = self.store.last_device(sender_id)
        if usual_dev and usual_dev != device_id:
            scenarios.append(("Paid from the usual device", dict(device_id=usual_dev)))
        if sender_id in self.users.index and self.users.city.get(sender_id) != base["city"]:
            scenarios.append(("Paid from the home city", dict(city=self.users.city.get(sender_id))))
        if channel == "collect":
            scenarios.append(("Normal payment instead of a collect request", dict(channel="p2p")))
        scenarios.append(("Paid at 2 pm instead", dict(timestamp=str(ts.replace(hour=14, minute=0, second=0)))))
        out = []
        for name, ch in scenarios:
            r = self.score_transaction(**{**base_args, **ch}, explain=False)
            out.append(dict(scenario=name, risk_score=r["risk_score"], decision=r["decision"], delta=round(r["risk_score"] - base["risk_score"], 5)))
        return dict(base=base, scenarios=sorted(out, key=lambda x: x["risk_score"]))
