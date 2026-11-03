"""Scoring per PROTOCOL.md. Written AFTER the zero-shot weights were
frozen and hashed (FREEZE.json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mlp import load_weights, ssi_forward  # noqa: E402
from oracle import ssi_oracle, ssi_region  # noqa: E402

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


def ssi_points():
    E_grid, U_grid = np.meshgrid(
        np.arange(0.0, 3_000.0 + 25.0, 25.0), np.arange(0.0, 1_500.0 + 25.0, 25.0)
    )
    parts_E = [E_grid.ravel()]
    parts_U = [U_grid.ravel()]
    spans = {}
    start = parts_E[0].size
    rng = np.random.default_rng(RNG_SEED)
    parts_E.append(rng.uniform(0.0, 3_000.0, 20_000))
    parts_U.append(rng.uniform(0.0, 1_500.0, 20_000))
    spans["random"] = slice(start, start + 20_000)
    start += 20_000
    n_kink = 5_000
    which = rng.integers(0, 3, n_kink)
    E_k = rng.uniform(0.0, 3_000.0, n_kink)
    U_k = rng.uniform(0.0, 1_500.0, n_kink)
    # plane 0: U = 20 (spill boundary)
    U_k = np.where(which == 0, 20.0 + rng.normal(0.0, 8.0, n_kink), U_k)
    # plane 1: earned-excess boundary E = 65 + max(0, 20 - U)
    E_k = np.where(which == 1, 65.0 + np.maximum(0.0, 20.0 - U_k) + rng.normal(0.0, 12.0, n_kink), E_k)
    # plane 2: payment-zero boundary C = 994 -> given U, solve E
    CU = np.maximum(0.0, U_k - 20.0)
    R = np.maximum(0.0, 20.0 - U_k)
    E_star = 65.0 + R + 2.0 * np.maximum(0.0, 994.0 - CU)
    E_k = np.where(which == 2, E_star + rng.normal(0.0, 15.0, n_kink), E_k)
    parts_E.append(np.clip(E_k, 0.0, 3_000.0))
    parts_U.append(np.clip(U_k, 0.0, 1_500.0))
    spans["kink"] = slice(start, start + n_kink)
    return np.concatenate(parts_E), np.concatenate(parts_U), spans


def main() -> None:
    arms = {"zero_shot": "zero_shot/ssi_weights.npz", "trained": "trained/ssi_weights.npz"}
    which = sys.argv[1] if len(sys.argv) > 1 else "zero_shot"
    base = Path(__file__).parent
    w = load_weights(str(base / arms[which]))
    E, U, spans = ssi_points()
    err = ssi_forward(w, E, U) - ssi_oracle(E, U)
    n_grid = 121 * 61
    out = {"arm": which, "grid": metrics(err[:n_grid]),
           "random": metrics(err[spans["random"]]),
           "kink": metrics(err[spans["kink"]]),
           "all": metrics(err)}
    reg = ssi_region(E[:n_grid], U[:n_grid])
    names = {0: "U<=20 spill, no earned excess", 1: "U>20, no earned excess",
             2: "U<=20 spill, earned excess", 3: "U>20, earned excess",
             4: "spill, no excess, payment 0", 5: "U>20, no excess, payment 0",
             6: "spill, excess, payment 0", 7: "U>20, excess, payment 0"}
    out["regions"] = {}
    for r in sorted(np.unique(reg)):
        out["regions"][f"{int(r)}:{names.get(int(r), '?')}"] = metrics(err[:n_grid][reg == r])
    dest = base / "results" / f"{which}_scores.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
