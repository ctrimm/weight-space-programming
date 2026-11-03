"""Scoring per PROTOCOL.md. Arms: zero_shot, trained."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mlp import load_weights, couple_forward  # noqa: E402
from oracle import couple_oracle  # noqa: E402

RNG_SEED = 20261102


def metrics(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {"n": int(a.size), "max_abs_err": float(a.max()), "mean_abs_err": float(a.mean()),
            "pct_within_0.01": float(100 * np.mean(a <= 0.01)),
            "pct_within_1": float(100 * np.mean(a <= 1.0))}


def points():
    rng = np.random.default_rng(RNG_SEED)
    n = 25_000
    E1 = rng.uniform(0, 2000, n); U1 = rng.uniform(0, 1500, n)
    E2 = rng.uniform(0, 2000, n); U2 = rng.uniform(0, 1500, n)
    spans = {"random": slice(0, n)}
    nk = 5_000
    kE1 = rng.uniform(0, 2000, nk); kU1 = rng.uniform(0, 1500, nk)
    kE2 = rng.uniform(0, 2000, nk); kU2 = rng.uniform(0, 1500, nk)
    which = rng.integers(0, 3, nk)
    kU2 = np.where(which == 0, np.clip(20.0 - kU1 + rng.normal(0, 8, nk), 0, 1500), kU2)
    R = np.maximum(20.0 - (kU1 + kU2), 0.0)
    kE2 = np.where(which == 1, np.clip(65.0 + R - kE1 + rng.normal(0, 12, nk), 0, 2000), kE2)
    C = np.maximum(kU1 + kU2 - 20, 0) + np.maximum(kE1 + kE2 - 65 - np.maximum(20 - kU1 - kU2, 0), 0) / 2
    kU1 = np.where(which == 2, np.clip(1491.0 + 20.0 - C - kU2 + rng.normal(0, 15, nk), 0, 1500), kU1)
    E1 = np.concatenate([E1, kE1]); U1 = np.concatenate([U1, kU1])
    E2 = np.concatenate([E2, kE2]); U2 = np.concatenate([U2, kU2])
    spans["kink"] = slice(n, n + nk)
    Eg, Ug = np.meshgrid(np.arange(0.0, 3000.0 + 25.0, 25.0), np.arange(0.0, 1500.0 + 25.0, 25.0))
    g = Eg.size
    z = np.zeros(g)
    E1 = np.concatenate([E1, Eg.ravel(), z]); U1 = np.concatenate([U1, Ug.ravel(), z])
    E2 = np.concatenate([E2, z, Eg.ravel()]); U2 = np.concatenate([U2, z, Ug.ravel()])
    spans["slice_member2_zero"] = slice(n + nk, n + nk + g)
    spans["slice_member1_zero"] = slice(n + nk + g, n + nk + 2 * g)
    return E1, U1, E2, U2, spans


def main() -> None:
    arms = {"zero_shot": "zero_shot/couple_weights.npz", "trained": "trained/couple_weights.npz"}
    which = sys.argv[1] if len(sys.argv) > 1 else "zero_shot"
    base = Path(__file__).parent
    w = load_weights(str(base / arms[which]))
    E1, U1, E2, U2, spans = points()
    err = couple_forward(w, E1, U1, E2, U2) - couple_oracle(E1, U1, E2, U2)
    out = {"arm": which, "all": metrics(err)}
    for name, sl in spans.items():
        out[name] = metrics(err[sl])
    (base / "results" / f"{which}_scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
