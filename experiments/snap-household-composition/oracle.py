"""Stipulated rule oracle — SNAP allotment, household of 2, member-level.
See PROTOCOL.md. Vectorized."""
from __future__ import annotations

import numpy as np


def snapcomp_oracle(E1, U1, E2, U2, S):
    E1 = np.asarray(E1, dtype=np.float64); U1 = np.asarray(U1, dtype=np.float64)
    E2 = np.asarray(E2, dtype=np.float64); U2 = np.asarray(U2, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    E = E1 + E2
    U = U1 + U2
    A = 0.80 * E + U - 209.0
    X = np.minimum(np.maximum(0.0, S - 0.5 * A), 744.0)
    N = np.maximum(0.0, A - X)
    return np.maximum(0.0, 546.0 - 0.30 * N)
