"""Medically-needy share of cost oracle (CA, individual). See PROTOCOL.md."""
import numpy as np

MNIL_INDIVIDUAL = 600.0


def soc_oracle(E, U, M):
    E = np.asarray(E, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    CU = np.maximum(U - 20.0, 0.0)
    R = np.maximum(20.0 - U, 0.0)
    CE = np.maximum(E - 65.0 - R, 0.0) / 2.0
    return np.maximum(CU + CE - MNIL_INDIVIDUAL - M, 0.0)
