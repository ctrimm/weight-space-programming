"""Scoring per PROTOCOL.md. Written AFTER the zero-shot weights were
frozen and hashed (FREEZE.json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mlp import load_weights, medicaid_forward  # noqa: E402
from oracle import medicaid_oracle, medicaid_region  # noqa: E402

RNG_SEED = 20261102


def metrics(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {
        "n": int(a.size),
        "max_abs_err": float(a.max()),
        "mean_abs_err": float(a.mean()),
        "pct_within_0.01": float(100.0 * np.mean(a <= 0.01)),
        "pct_within_1": float(100.0 * np.mean(a <= 1.0)),
    }


def medicaid_points():
    I_grid, SI_grid = np.meshgrid(
        np.arange(0.0, 6_000.0 + 50.0, 50.0), np.arange(0.0, 5_000.0 + 50.0, 50.0)
    )
    P_levels = np.array([0.0, 150.0, 300.0, 450.0, 600.0])
    I_g = np.repeat(I_grid.ravel(), len(P_levels))
    SI_g = np.repeat(SI_grid.ravel(), len(P_levels))
    P_g = np.tile(P_levels, I_grid.size)
    parts_I, parts_SI, parts_P = [I_g], [SI_g], [P_g]
    spans = {}
    start = I_g.size
    rng = np.random.default_rng(RNG_SEED)
    parts_I.append(rng.uniform(0.0, 6_000.0, 20_000))
    parts_SI.append(rng.uniform(0.0, 5_000.0, 20_000))
    parts_P.append(rng.uniform(0.0, 600.0, 20_000))
    spans["random"] = slice(start, start + 20_000)
    start += 20_000
    n_kink = 5_000
    which = rng.integers(0, 2, n_kink)
    I_k = rng.uniform(0.0, 6_000.0, n_kink)
    SI_k = rng.uniform(0.0, 5_000.0, n_kink)
    P_k = rng.uniform(0.0, 600.0, n_kink)
    # plane 0: SI = MMMNA (allowance boundary)
    SI_k = np.where(which == 0, 2_643.75 + rng.normal(0.0, 20.0, n_kink), SI_k)
    # plane 1: liability-zero plane I = 50 + A + P
    A_k = np.maximum(0.0, 2_643.75 - SI_k)
    I_k = np.where(which == 1, 50.0 + A_k + P_k + rng.normal(0.0, 20.0, n_kink), I_k)
    parts_I.append(np.clip(I_k, 0.0, 6_000.0))
    parts_SI.append(np.clip(SI_k, 0.0, 5_000.0))
    parts_P.append(np.clip(P_k, 0.0, 600.0))
    spans["kink"] = slice(start, start + n_kink)
    return np.concatenate(parts_I), np.concatenate(parts_SI), np.concatenate(parts_P), spans


def main() -> None:
    arms = {"zero_shot": "zero_shot/medicaid_weights.npz", "trained": "trained/medicaid_weights.npz"}
    which = sys.argv[1] if len(sys.argv) > 1 else "zero_shot"
    base = Path(__file__).parent
    w = load_weights(str(base / arms[which]))
    I, SI, P, spans = medicaid_points()
    err = medicaid_forward(w, I, SI, P) - medicaid_oracle(I, SI, P)
    n_grid = 121 * 101 * 5
    out = {"arm": which, "grid": metrics(err[:n_grid]),
           "random": metrics(err[spans["random"]]),
           "kink": metrics(err[spans["kink"]]),
           "all": metrics(err)}
    reg = medicaid_region(I[:n_grid], SI[:n_grid], P[:n_grid])
    names = {0: "allowance>0, liability>0", 1: "allowance=0, liability>0",
             2: "allowance>0, liability=0", 3: "allowance=0, liability=0"}
    out["regions"] = {}
    for r in sorted(np.unique(reg)):
        out["regions"][f"{int(r)}:{names.get(int(r), '?')}"] = metrics(err[:n_grid][reg == r])
    dest = base / "results" / f"{which}_scores.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
