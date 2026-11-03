"""Stipulated rule oracles for the weight-space programming test.

These implement PROTOCOL.md exactly. They are the ground truth the
networks are scored against -- the ceiling arm, not a competitor.
"""
from __future__ import annotations

import numpy as np

# --- Rule A: EITC TY2025, single/HOH, 2 children, AGI = earned income ---
EITC_PHASE_IN_RATE = 0.40
EITC_PHASE_IN_END = 17_880.0
EITC_MAX = 7_152.0
EITC_PHASEOUT_START = 23_350.0
EITC_PHASEOUT_RATE = 0.2106
EITC_PHASEOUT_END = 57_310.0


def eitc_oracle(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    credit = np.where(
        x <= EITC_PHASE_IN_END,
        EITC_PHASE_IN_RATE * x,
        np.where(
            x <= EITC_PHASEOUT_START,
            EITC_MAX,
            np.where(
                x < EITC_PHASEOUT_END,
                EITC_MAX - EITC_PHASEOUT_RATE * (x - EITC_PHASEOUT_START),
                0.0,
            ),
        ),
    )
    return credit


# --- Rule B: SNAP FY2026 allotment, household of 2, earned income only ---
SNAP_STD_DEDUCTION = 209.0
SNAP_MAX_ALLOTMENT = 546.0
SNAP_SHELTER_CAP = 744.0


def snap_oracle(E: np.ndarray, S: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    A = 0.80 * E - SNAP_STD_DEDUCTION
    X = np.maximum(0.0, S - 0.5 * A)
    Xc = np.minimum(X, SNAP_SHELTER_CAP)
    N = np.maximum(0.0, A - Xc)
    B = np.maximum(0.0, SNAP_MAX_ALLOTMENT - 0.30 * N)
    return B


def snap_region(E: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Region id per point, for error breakdown (see PROTOCOL.md).
    0: no excess shelter (X=0)
    1: excess shelter, uncapped
    2: excess shelter, capped
    Crossed with benefit state appended as +10 if N floored at 0 is the
    binding floor, +20 if benefit is 0.
    """
    E = np.asarray(E, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    A = 0.80 * E - SNAP_STD_DEDUCTION
    X = np.maximum(0.0, S - 0.5 * A)
    Xc = np.minimum(X, SNAP_SHELTER_CAP)
    base = np.where(X <= 0, 0, np.where(X <= SNAP_SHELTER_CAP, 1, 2))
    N_raw = A - Xc
    B_raw = SNAP_MAX_ALLOTMENT - 0.30 * np.maximum(0.0, N_raw)
    region = base.copy()
    region = np.where(N_raw < 0, region + 10, region)
    region = np.where(B_raw <= 0, region + 20, region)
    return region
