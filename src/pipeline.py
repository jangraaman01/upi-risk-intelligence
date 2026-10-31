"""One-command pipeline:  python -m src.pipeline [--scale small|medium|large] [--seed N]

small  ~0.2M transactions (~1 min)   medium ~0.75M (default, ~5 min)   large ~3M+ (much more RAM/time; not benchmarked)
"""
import argparse
import json

from .config import DATA
from .generate_data import PRESETS, generate
from .logger import get_logger
from .train import main as train

log = get_logger("pipeline")


def run(scale: str = "medium", seed: int = 7):
    """Simulate the world, save it, train and evaluate. Also used by the console's first-launch bootstrap."""
    DATA.mkdir(exist_ok=True)
    df, users, merchants = generate(**PRESETS[scale], seed=seed)
    df.to_parquet(DATA / "transactions.parquet", index=False)
    users.to_parquet(DATA / "users.parquet", index=False)
    merchants.to_parquet(DATA / "merchants.parquet", index=False)
    json.dump(dict(scale=scale, seed=seed, rows=len(df), **PRESETS[scale]), open(DATA / "world.json", "w"))
    log.info("generated %s world: %d transactions, %d users, %d merchants", scale, len(df), len(users), len(merchants))
    return train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(PRESETS), default="medium")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    run(a.scale, a.seed)
