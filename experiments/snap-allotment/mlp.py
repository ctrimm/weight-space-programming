"""Generic MLP forward pass. Deliberately rule-free: this file knows
nothing about any program's rules. If a rule appears here, the test is
void. Weights live in .npz files with keys W1,b1,...,Wn,bn; every
layer but the last is followed by ReLU. Input/output scaling constants
are architecture (see each experiment's PROTOCOL.md), applied by the
wrappers below.
"""
from __future__ import annotations

import numpy as np


def load_weights(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def forward(w: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    h = X
    i = 1
    while f"W{i}" in w:
        h = h @ w[f"W{i}"].T + w[f"b{i}"]
        if f"W{i + 1}" in w:
            h = np.maximum(h, 0.0)
        i += 1
    return h[:, 0]


def eitc_forward(w, x_dollars):
    X = (np.asarray(x_dollars, dtype=np.float64) / 10_000.0).reshape(-1, 1)
    return 1_000.0 * forward(w, X)


def snap_forward(w, E, S):
    X = np.stack([np.asarray(E, dtype=np.float64) / 1_000.0,
                  np.asarray(S, dtype=np.float64) / 1_000.0], axis=1)
    return 1_000.0 * forward(w, X)


def ssi_forward(w, E, U):
    X = np.stack([np.asarray(E, dtype=np.float64) / 1_000.0,
                  np.asarray(U, dtype=np.float64) / 1_000.0], axis=1)
    return 1_000.0 * forward(w, X)


def medicaid_forward(w, I, SI, P):
    X = np.stack([np.asarray(I, dtype=np.float64) / 1_000.0,
                  np.asarray(SI, dtype=np.float64) / 1_000.0,
                  np.asarray(P, dtype=np.float64) / 1_000.0], axis=1)
    return 1_000.0 * forward(w, X)
