"""Central configuration: paths, seeds, model hyper-parameters and the business cost model."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, REPORTS, LOGS = ROOT / "data", ROOT / "models", ROOT / "reports", ROOT / "logs"
AUDIT_DB = Path(os.getenv("UPI_AUDIT_DB", DATA / "audit.db"))
CASES_DB = Path(os.getenv("UPI_CASES_DB", DATA / "cases.db"))
START = pd.Timestamp("2026-01-01")      # first day of every simulated world (used for account-age arithmetic)
MODEL_VERSION = "2.0.0"
SEED = 7
WARMUP_DAYS = 14   # initial history-building period excluded from training/evaluation (avoids cold-start drift)

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)

# Business cost model (INR). Used to choose decision thresholds by minimising expected total cost.
COSTS = dict(review_cost=20.0,      # analyst time per manually reviewed transaction
             review_catch=0.80,     # share of fraud an analyst stops
             stepup_cost=3.0,       # OTP / biometric friction per transaction
             stepup_catch=0.90,     # share of fraud stopped by step-up auth
             stepup_abandon=0.02,   # share of genuine users who abandon after step-up
             abandon_cost=50.0,     # value lost per abandoned genuine payment
             block_fp_cost=400.0)   # churn / complaint / support cost of wrongly blocking a genuine payment
