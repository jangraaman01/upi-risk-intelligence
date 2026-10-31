"""Single source of truth for building the modelling table (used by training AND the analysis notebook)."""
import numpy as np
import pandas as pd

from .config import DATA, WARMUP_DAYS
from .features import build_features
from .logger import get_logger
from .validation import validate_transactions

log = get_logger("dataset")


def load_modeling_data():
    """Returns (raw_df, data): `data` has features, a post-warm-up filter and a chronological train/val/test `split` (60/15/25)."""
    df = validate_transactions(pd.read_parquet(DATA / "transactions.parquet"))
    feats = build_features(df)
    data = df.drop(columns=[c for c in feats.columns if c in df.columns and c != "txn_id"]).merge(feats, on="txn_id")
    data = data.sort_values("timestamp", kind="stable")
    data = data[data.timestamp >= data.timestamp.min() + pd.Timedelta(days=WARMUP_DAYS)].reset_index(drop=True)   # drop warm-up
    log.info("excluded %d-day warm-up; %d rows remain", WARMUP_DAYS, len(data))
    n = len(data)
    data["split"] = np.where(np.arange(n) < .60 * n, "train", np.where(np.arange(n) < .75 * n, "val", "test"))
    return df, data
