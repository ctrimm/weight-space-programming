"""Scoring per PROTOCOL.md. Loads any .npz arm, generic forward pass."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mlp import load_weights, eitc_forward  # noqa: E402
from oracle import eitc_oracle  # noqa: E402


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


def main() -> None:
    arms = {
        "zero_shot": "zero_shot/eitc_weights.npz",
        "zero_shot_attempt2": "zero_shot/eitc_weights_attempt2.npz",
        "trained": "trained/eitc_weights.npz",
        "trained_v1": "trained_v1/eitc_weights.npz",
    }
    which = sys.argv[1] if len(sys.argv) > 1 else "zero_shot"
    base = Path(__file__).parent
    w = load_weights(str(base / arms[which]))
    x = eitc_points()
    out = {"arm": which, "eitc_all": metrics(eitc_forward(w, x) - eitc_oracle(x))}
    dest = base / "results" / f"{which}_scores.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
