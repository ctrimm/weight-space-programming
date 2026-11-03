"""Stipulated SNAP allotment oracle (PROTOCOL.md). Ground truth."""
from __future__ import annotations

import numpy as np

STD_DEDUCTION = 209.0
MAX_ALLOTMENT = 546.0
SHELTER_CAP = 744.0


def snap_oracle(E: np.ndarray, S: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    A = 0.80 * E - STD_DEDUCTION
    X = np.maximum(0.0, S - 0.5 * A)
    Xc = np.minimum(X, SHELTER_CAP)
    N = np.maximum(0.0, A - Xc)
    return np.maximum(0.0, MAX_ALLOTMENT - 0.30 * N)


def snap_region(E: np.ndarray, S: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    A = 0.80 * E - STD_DEDUCTION
    X = np.maximum(0.0, S - 0.5 * A)
    Xc = np.minimum(X, SHELTER_CAP)
    base = np.where(X <= 0, 0, np.where(X <= SHELTER_CAP, 1, 2))
    N_raw = A - Xc
    B_raw = MAX_ALLOTMENT - 0.30 * np.maximum(0.0, N_raw)
    region = np.where(N_raw < 0, base + 10, base)
    return np.where(B_raw <= 0, region + 20, region)
