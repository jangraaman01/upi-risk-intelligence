"""Schema / data-quality checks run before training and scoring."""
import pandas as pd

REQUIRED = ["txn_id", "timestamp", "sender_id", "receiver_id", "receiver_type", "amount", "channel", "device_id",
            "city", "lat", "lon", "merch_cat", "receiver_acct_age_days", "sender_acct_age_days"]
CHANNELS = {"p2p", "collect", "qr", "intent"}


class DataValidationError(ValueError):
    pass


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise DataValidationError("timestamp must be datetime64")
    if df[REQUIRED].isna().any().any():
        raise DataValidationError("null values in required columns")
    if df["txn_id"].duplicated().any():
        raise DataValidationError("duplicate txn_id")
    if (df["amount"] <= 0).any():
        raise DataValidationError("non-positive amounts")
    bad = set(df["channel"].unique()) - CHANNELS
    if bad:
        raise DataValidationError(f"unknown channels: {bad}")
    if not set(df["receiver_type"].unique()) <= {"P2P", "MERCHANT"}:
        raise DataValidationError("receiver_type must be P2P or MERCHANT")
    return df
