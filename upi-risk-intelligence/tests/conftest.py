import pytest
from src.generate_data import generate


@pytest.fixture(scope="session")
def small_df():
    df, users, merchants = generate(n_users=300, n_merchants=40, days=20, fraud_scale=0.3, seed=1)
    return df
