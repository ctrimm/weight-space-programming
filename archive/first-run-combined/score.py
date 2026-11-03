"""Scoring per PROTOCOL.md. Written AFTER the zero-shot weights were
frozen and hashed (FREEZE.json). Loads any .npz arm, runs the generic
forward pass, reports errors against the oracle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mlp import load_weights, eitc_forward, snap_forward  # noqa: E402
from oracle import eitc_oracle, snap_oracle, snap_region  # noqa: E402

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


def eitc_points() -> np.ndarray:
    grid = np.arange(0.0, 70_000.0 + 25.0, 25.0)
    probes = []
    for bp in (17_880.0, 23_350.0, 57_310.0):
        for d in (-25.0, -5.0, -1.0, 0.0, 1.0, 5.0, 25.0):
            probes.append(bp + d)
    return np.unique(np.concatenate([grid, np.array(probes)]))


def snap_points() -> tuple[np.ndarray, np.ndarray, dict[str, slice]]:
    E_grid, S_grid = np.meshgrid(
        np.arange(300.0, 4_000.0 + 25.0, 25.0), np.arange(0.0, 2_500.0 + 25.0, 25.0)
    )
    parts_E = [E_grid.ravel()]
    parts_S = [S_grid.ravel()]
    spans = {}
    start = parts_E[0].size
    rng = np.random.default_rng(RNG_SEED)
    # uniform random
    parts_E.append(rng.uniform(300.0, 4_000.0, 20_000))
    parts_S.append(rng.uniform(0.0, 2_500.0, 20_000))
    spans["random"] = slice(start, start + 20_000)
    start += 20_000
    # kink-jittered: pick a kink plane, place points near it
    n_kink = 5_000
    E_k = rng.uniform(300.0, 4_000.0, n_kink)
    A_k = 0.8 * E_k - 209.0
    which = rng.integers(0, 4, n_kink)
    S_k = np.empty(n_kink)
    jit = rng.normal(0.0, 15.0, n_kink)
    # plane 0: S = 0.5A (X=0); plane 1: S = 0.5A + 744 (cap binds)
    S_k = np.where(which == 0, 0.5 * A_k + jit, S_k)
    S_k = np.where(which == 1, 0.5 * A_k + 744.0 + jit, S_k)
    # plane 2: N=0 with uncapped excess -> S = 1.5A ; plane 3: benefit=0 -> N=1820, uncapped, S small: use E jitter instead
    S_k = np.where(which == 2, 1.5 * A_k + jit, S_k)
    S_k = np.where(which == 3, rng.uniform(0.0, 2_500.0, n_kink), S_k)
    E_k = np.where(which == 3, (1820.0 + 209.0) / 0.8 + rng.normal(0.0, 40.0, n_kink), E_k)
    parts_E.append(np.clip(E_k, 300.0, 4_000.0))
    parts_S.append(np.clip(S_k, 0.0, 2_500.0))
    spans["kink"] = slice(start, start + n_kink)
    return np.concatenate(parts_E), np.concatenate(parts_S), spans


def main() -> None:
    arms = {
        "zero_shot": ("zero_shot/eitc_weights.npz", "zero_shot/snap_weights.npz"),
        "trained": ("trained/eitc_weights.npz", "trained/snap_weights.npz"),
    }
    which = sys.argv[1] if len(sys.argv) > 1 else "zero_shot"
    eitc_path, snap_path = arms[which]
    base = Path(__file__).parent
    out: dict = {"arm": which}

    w = load_weights(str(base / eitc_path))
    x = eitc_points()
    err = eitc_forward(w, x) - eitc_oracle(x)
    out["eitc_all"] = metrics(err)

    w = load_weights(str(base / snap_path))
    E, S, spans = snap_points()
    pred = snap_forward(w, E, S)
    truth = snap_oracle(E, S)
    err = pred - truth
    n_grid = 149 * 101
    out["snap_grid"] = metrics(err[:n_grid])
    out["snap_random"] = metrics(err[spans["random"]])
    out["snap_kink"] = metrics(err[spans["kink"]])
    out["snap_all"] = metrics(err)
    # region breakdown on grid points
    reg = snap_region(E[:n_grid], S[:n_grid])
    names = {
        0: "no excess shelter",
        1: "excess shelter uncapped",
        2: "excess shelter capped",
        10: "no excess, N floored",
        11: "uncapped, N floored",
        12: "capped, N floored",
        20: "no excess, benefit 0",
        21: "uncapped, benefit 0",
        22: "capped, benefit 0",
        30: "no excess, N floored (benefit max region edge)",
        31: "uncapped, N floored",
        32: "capped, N floored",
    }
    out["snap_regions"] = {}
    for r in sorted(np.unique(reg)):
        m = metrics(err[:n_grid][reg == r])
        out["snap_regions"][f"{int(r)}:{names.get(int(r), '?')}"] = m

    dest = base / "results" / f"{which}_scores.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
