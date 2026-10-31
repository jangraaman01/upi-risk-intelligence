import numpy as np
import pandas as pd
import pytest
from src.features import FEATURES, build_features
from src.validation import DataValidationError, validate_transactions


def test_generator_schema_and_fraud_types(small_df):
    validate_transactions(small_df)
    assert 0.002 < small_df.is_fraud.mean() < 0.15
    assert {"account_takeover", "velocity_attack", "collect_scam", "mule_ring"} <= set(small_df.fraud_type)
    assert small_df.timestamp.is_monotonic_increasing


def test_features_complete_and_finite(small_df):
    f = build_features(small_df)
    assert list(f.columns[:1]) == ["txn_id"] and set(FEATURES) <= set(f.columns)
    assert len(f) == len(small_df)
    assert np.isfinite(f[FEATURES].to_numpy(float)).all()


def test_features_have_no_future_leakage(small_df):
    """A transaction's features must not change when LATER transactions are added to the data."""
    cut = small_df.timestamp.quantile(0.5)
    full = build_features(small_df).set_index("txn_id")
    part = build_features(small_df[small_df.timestamp <= cut]).set_index("txn_id")
    pd.testing.assert_frame_equal(full.loc[part.index, FEATURES], part[FEATURES], check_dtype=False)


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.drop(columns=["amount"]), "missing"),
    (lambda d: d.assign(amount=-1.0), "non-positive"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate"),
    (lambda d: d.assign(channel="cheque"), "unknown channels"),
])
def test_validation_rejects_bad_data(small_df, mutate, msg):
    with pytest.raises(DataValidationError, match=msg):
        validate_transactions(mutate(small_df.copy()))
