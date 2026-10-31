"""A reasonable hand-written rule set - the baseline any ML system must beat."""
import numpy as np


def rule_flags(d) -> np.ndarray:
    return (((d.amount_ratio > 4) & (d.new_receiver == 1)) | (d.sender_cnt_10m >= 6)
            | ((d.is_night == 1) & (d.new_device == 1) & (d.amount_ratio > 2))).to_numpy()
