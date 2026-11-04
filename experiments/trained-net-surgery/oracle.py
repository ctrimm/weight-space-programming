"""SSI couple payment oracle. See PROTOCOL.md."""
import numpy as np

FBR_COUPLE = 1491.0


def couple_oracle(E1, U1, E2, U2):
    E = np.asarray(E1, dtype=np.float64) + np.asarray(E2, dtype=np.float64)
    U = np.asarray(U1, dtype=np.float64) + np.asarray(U2, dtype=np.float64)
    CU = np.maximum(U - 20.0, 0.0)
    R = np.maximum(20.0 - U, 0.0)
    CE = np.maximum(E - 65.0 - R, 0.0) / 2.0
    return np.maximum(FBR_COUPLE - (CU + CE), 0.0)
