"""Scoring per PROTOCOL.md. Loads any .npz arm, generic forward pass."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mlp import load_weights, snap_forward  # noqa: E402
from oracle import snap_oracle, snap_region  # noqa: E402

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


def snap_points():
    E_grid, S_grid = np.meshgrid(
        np.arange(300.0, 4_000.0 + 25.0, 25.0), np.arange(0.0, 2_500.0 + 25.0, 25.0)
    )
    parts_E = [E_grid.ravel()]
    parts_S = [S_grid.ravel()]
    spans = {}
    start = parts_E[0].size
    rng = np.random.default_rng(RNG_SEED)
    parts_E.append(rng.uniform(300.0, 4_000.0, 20_000))
    parts_S.append(rng.uniform(0.0, 2_500.0, 20_000))
    spans["random"] = slice(start, start + 20_000)
    start += 20_000
    n_kink = 5_000
    E_k = rng.uniform(300.0, 4_000.0, n_kink)
    A_k = 0.8 * E_k - 209.0
    which = rng.integers(0, 4, n_kink)
    S_k = np.empty(n_kink)
    jit = rng.normal(0.0, 15.0, n_kink)
    S_k = np.where(which == 0, 0.5 * A_k + jit, S_k)
    S_k = np.where(which == 1, 0.5 * A_k + 744.0 + jit, S_k)
    S_k = np.where(which == 2, 1.5 * A_k + jit, S_k)
    S_k = np.where(which == 3, rng.uniform(0.0, 2_500.0, n_kink), S_k)
    E_k = np.where(which == 3, (1820.0 + 209.0) / 0.8 + rng.normal(0.0, 40.0, n_kink), E_k)
    parts_E.append(np.clip(E_k, 300.0, 4_000.0))
    parts_S.append(np.clip(S_k, 0.0, 2_500.0))
    spans["kink"] = slice(start, start + n_kink)
    return np.concatenate(parts_E), np.concatenate(parts_S), spans


def main() -> None:
    arms = {
        "zero_shot": "zero_shot/snap_weights.npz",
        "trained": "trained/snap_weights.npz",
        "trained_v1": "trained_v1/snap_weights.npz",
    }
    which = sys.argv[1] if len(sys.argv) > 1 else "zero_shot"
    base = Path(__file__).parent
    w = load_weights(str(base / arms[which]))
    E, S, spans = snap_points()
    err = snap_forward(w, E, S) - snap_oracle(E, S)
    n_grid = 149 * 101
    out = {"arm": which, "snap_grid": metrics(err[:n_grid]),
           "snap_random": metrics(err[spans["random"]]),
           "snap_kink": metrics(err[spans["kink"]]),
           "snap_all": metrics(err)}
    reg = snap_region(E[:n_grid], S[:n_grid])
    names = {0: "no excess shelter", 1: "excess shelter uncapped", 2: "excess shelter capped",
             11: "uncapped, N floored", 12: "capped, N floored",
             20: "no excess, benefit 0", 21: "uncapped, benefit 0", 22: "capped, benefit 0",
             31: "uncapped, N floored", 32: "capped, N floored"}
    out["snap_regions"] = {}
    for r in sorted(np.unique(reg)):
        out["snap_regions"][f"{int(r)}:{names.get(int(r), '?')}"] = metrics(err[:n_grid][reg == r])
    dest = base / "results" / f"{which}_scores.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
