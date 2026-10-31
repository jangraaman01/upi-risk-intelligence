import logging
import sys
from .config import LOGS


def get_logger(name: str = "upi") -> logging.Logger:
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGS.mkdir(exist_ok=True)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / "risk_engine.log")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg
