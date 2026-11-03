"""Stipulated SSI payment oracle (PROTOCOL.md). Ground truth."""
from __future__ import annotations

import numpy as np

FBR = 994.0
GENERAL_EXCLUSION = 20.0
EARNED_EXCLUSION = 65.0


def ssi_oracle(E: np.ndarray, U: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    CU = np.maximum(0.0, U - GENERAL_EXCLUSION)
    R = np.maximum(0.0, GENERAL_EXCLUSION - U)
    CE = np.maximum(0.0, E - EARNED_EXCLUSION - R) / 2.0
    return np.maximum(0.0, FBR - (CU + CE))


def ssi_region(E: np.ndarray, U: np.ndarray) -> np.ndarray:
    """0: U<=20 (spill active), 1: U>20; +2 if earned excess > 0; +4 if payment = 0."""
    E = np.asarray(E, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    CU = np.maximum(0.0, U - GENERAL_EXCLUSION)
    R = np.maximum(0.0, GENERAL_EXCLUSION - U)
    CE = np.maximum(0.0, E - EARNED_EXCLUSION - R) / 2.0
    region = np.where(U <= GENERAL_EXCLUSION, 0, 1)
    region = np.where(CE > 0, region + 2, region)
    return np.where(CU + CE >= FBR, region + 4, region)
