"""SSI oracles, parameterized by FBR. Arithmetic cores only (see
../ssi-payment/PROTOCOL.md). $k scaling lives in mlp.py."""
import numpy as np

FBR_STALE = 943.0    # TY2024 individual rate, per SSA
FBR_CURRENT = 994.0  # CY2026 individual rate, per SSA


def ssi_oracle_fbr(E, U, fbr):
    E = np.asarray(E, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    CU = np.maximum(U - 20.0, 0.0)
    R = np.maximum(20.0 - U, 0.0)
    CE = np.maximum(E - 65.0 - R, 0.0) / 2.0
    return np.maximum(fbr - (CU + CE), 0.0)


def ssi_oracle_stale(E, U):
    return ssi_oracle_fbr(E, U, FBR_STALE)


def ssi_oracle_current(E, U):
    return ssi_oracle_fbr(E, U, FBR_CURRENT)


def theoretical_delta(E, U):
    E = np.asarray(E, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    C = np.maximum(U - 20.0, 0.0) + np.maximum(E - 65.0 - np.maximum(20.0 - U, 0.0), 0.0) / 2.0
    return np.maximum(FBR_CURRENT - C, 0.0) - np.maximum(FBR_STALE - C, 0.0)
