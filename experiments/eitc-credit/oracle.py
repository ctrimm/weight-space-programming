"""Stipulated EITC oracle (PROTOCOL.md). Ground truth."""
from __future__ import annotations

import numpy as np


def eitc_oracle(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x <= 17_880.0,
        0.40 * x,
        np.where(
            x <= 23_350.0,
            7_152.0,
            np.where(x < 57_310.0, 7_152.0 - 0.2106 * (x - 23_350.0), 0.0),
        ),
    )
