"""Stipulated Medicaid patient-liability oracle (PROTOCOL.md). Ground truth."""
from __future__ import annotations

import numpy as np

MMMNA_MIN = 2_643.75
PNA = 50.0


def medicaid_oracle(I: np.ndarray, SI: np.ndarray, P: np.ndarray) -> np.ndarray:
    I = np.asarray(I, dtype=np.float64)
    SI = np.asarray(SI, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    A = np.maximum(0.0, MMMNA_MIN - SI)
    return np.maximum(0.0, I - PNA - A - P)


def medicaid_region(I: np.ndarray, SI: np.ndarray, P: np.ndarray) -> np.ndarray:
    """0: allowance > 0, liability > 0; 1: allowance = 0, liability > 0;
    2: allowance > 0, liability = 0; 3: allowance = 0, liability = 0."""
    I = np.asarray(I, dtype=np.float64)
    SI = np.asarray(SI, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    A = np.maximum(0.0, MMMNA_MIN - SI)
    L = np.maximum(0.0, I - PNA - A - P)
    return np.where(A > 0, np.where(L > 0, 0, 2), np.where(L > 0, 1, 3))
