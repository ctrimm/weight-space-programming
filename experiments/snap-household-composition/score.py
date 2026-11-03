"""Scoring per PROTOCOL.md. Arms: zero_shot, trained. Also computes
the composition-invariance and should-change-control metrics for
any arm. Usage: python3 score.py <arm> [weights_path]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mlp import load_weights, snapcomp_forward  # noqa: E402
from oracle import snapcomp_oracle  # noqa: E402

SEED = 20261102
SHARES = np.array([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])


def metrics(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {"n": int(a.size), "max_abs_err": float(a.max()), "mean_abs_err": float(a.mean()),
            "pct_within_0.01": float(100 * np.mean(a <= 0.01)),
            "pct_within_1": float(100 * np.mean(a <= 1.0))}


def accuracy_points():
    rng = np.random.default_rng(SEED)
    n = 25_000
    E1 = rng.uniform(0, 3000, n); U1 = rng.uniform(0, 2000, n)
    E2 = rng.uniform(0, 3000, n); U2 = rng.uniform(0, 2000, n)
    S = rng.uniform(0, 2500, n)
    spans = {"random": slice(0, n)}
    nk = 5_000
    kE1 = rng.uniform(0, 3000, nk); kU1 = rng.uniform(0, 2000, nk)
    kE2 = rng.uniform(0, 3000, nk); kU2 = rng.uniform(0, 2000, nk)
    kS = rng.uniform(0, 2500, nk)
    A = 0.8 * (kE1 + kE2) + (kU1 + kU2) - 209.0
    which = rng.integers(0, 4, nk)
    kS = np.where(which == 0, np.clip(0.5 * A + rng.normal(0, 40, nk), 0, 2500), kS)
    kS = np.where(which == 1, np.clip(0.5 * A + 744 + rng.normal(0, 40, nk), 0, 2500), kS)
    kU2 = np.where(which == 2, np.clip(209 - 0.8 * (kE1 + kE2) - kU1 + rng.normal(0, 30, nk), 0, 2000), kU2)
    kE2 = np.where(which == 3, np.clip((1820 - (kU1 + kU2) + 209) / 0.8 - kE1 + rng.normal(0, 60, nk), 0, 3000), kE2)
    E1 = np.concatenate([E1, kE1]); U1 = np.concatenate([U1, kU1])
    E2 = np.concatenate([E2, kE2]); U2 = np.concatenate([U2, kU2])
    S = np.concatenate([S, kS])
    spans["kink"] = slice(n, n + nk)
    return E1, U1, E2, U2, S, spans


def composition_classes():
    """Yields (E1,U1,E2,U2,S) blocks, one per totals class."""
    for E_tot in (0, 250, 1000, 2500, 4000, 6000):
        for U_tot in (0, 100, 500, 2000, 4000):
            for S_v in (0.0, 800.0, 2000.0):
                A, B = np.meshgrid(SHARES, SHARES, indexing="ij")
                a, b = A.ravel(), B.ravel()
                yield (a * E_tot, b * U_tot, (1 - a) * E_tot, (1 - b) * U_tot,
                       np.full(a.size, S_v))


def score(w) -> dict:
    E1, U1, E2, U2, S, spans = accuracy_points()
    err = snapcomp_forward(w, E1, U1, E2, U2, S) - snapcomp_oracle(E1, U1, E2, U2, S)
    out = {"accuracy": metrics(err)}
    for name, sl in spans.items():
        out[name] = metrics(err[sl])
    ranges = []
    for block in composition_classes():
        o = snapcomp_forward(w, *block)
        ranges.append(float(o.max() - o.min()))
    ranges = np.array(ranges)
    out["invariance"] = {"max_range": float(ranges.max()), "mean_range": float(ranges.mean()),
                         "pct_classes_over_0.01": float(100 * np.mean(ranges > 0.01)),
                         "n_classes": int(ranges.size)}
    # should-change control: $100 from E1 to U1
    rng = np.random.default_rng(SEED + 1)
    n = 2_000
    cE1 = rng.uniform(100, 3000, n); cU1 = rng.uniform(0, 1900, n)
    cE2 = rng.uniform(0, 3000, n); cU2 = rng.uniform(0, 2000, n)
    cS = rng.uniform(0, 2500, n)
    d_net = (snapcomp_forward(w, cE1 - 100, cU1 + 100, cE2, cU2, cS)
             - snapcomp_forward(w, cE1, cU1, cE2, cU2, cS))
    d_or = (snapcomp_oracle(cE1 - 100, cU1 + 100, cE2, cU2, cS)
            - snapcomp_oracle(cE1, cU1, cE2, cU2, cS))
    derr = np.abs(d_net - d_or)
    out["should_change_control"] = {
        "oracle_delta_mean": float(np.mean(d_or)), "oracle_delta_min": float(np.min(d_or)),
        "delta_max_abs_err": float(derr.max()), "delta_mean_abs_err": float(derr.mean())}
    return out


def main() -> None:
    base = Path(__file__).parent
    which = sys.argv[1] if len(sys.argv) > 1 else "zero_shot"
    arms = {"zero_shot": "zero_shot/snapcomp_weights.npz", "trained": "trained/snapcomp_weights.npz"}
    path = sys.argv[2] if len(sys.argv) > 2 else str(base / arms[which])
    w = load_weights(path)
    out = {"arm": which, **score(w)}
    (base / "results").mkdir(exist_ok=True)
    (base / "results" / f"{which}_scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
