"""Scoring per PROTOCOL.md. Arms: zero_shot, trained."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mlp import load_weights, soc_forward  # noqa: E402
from oracle import soc_oracle  # noqa: E402

RNG_SEED = 20261102


def metrics(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {"n": int(a.size), "max_abs_err": float(a.max()), "mean_abs_err": float(a.mean()),
            "pct_within_0.01": float(100 * np.mean(a <= 0.01)),
            "pct_within_1": float(100 * np.mean(a <= 1.0))}


def points():
    Eg, Ug = np.meshgrid(np.arange(0.0, 3000.0 + 50.0, 50.0), np.arange(0.0, 3000.0 + 50.0, 50.0))
    parts = []
    for Mv in (0.0, 250.0, 500.0, 1000.0, 2000.0):
        parts.append((Eg.ravel(), Ug.ravel(), np.full(Eg.size, Mv)))
    E = np.concatenate([p[0] for p in parts]); U = np.concatenate([p[1] for p in parts]); M = np.concatenate([p[2] for p in parts])
    spans = {"grid": slice(0, E.size)}
    rng = np.random.default_rng(RNG_SEED)
    start = E.size
    E = np.concatenate([E, rng.uniform(0, 3000, 20_000)]); U = np.concatenate([U, rng.uniform(0, 3000, 20_000)]); M = np.concatenate([M, rng.uniform(0, 2000, 20_000)])
    spans["random"] = slice(start, start + 20_000); start += 20_000
    Ek = rng.uniform(0, 3000, 5_000); Uk = rng.uniform(0, 3000, 5_000); Mk = rng.uniform(0, 2000, 5_000)
    which = rng.integers(0, 3, 5_000)
    Uk = np.where(which == 0, 20.0 + rng.normal(0, 8, 5_000), Uk)
    Ck = np.maximum(Uk - 20, 0) + np.maximum(Ek - 65 - np.maximum(20 - Uk, 0), 0) / 2
    Mk = np.where(which == 1, np.maximum(Ck - 600 + rng.normal(0, 15, 5_000), 0), Mk)
    Ek = np.where(which == 2, 65.0 + np.maximum(0.0, 20.0 - Uk) + rng.normal(0, 12, 5_000), Ek)
    E = np.concatenate([E, np.clip(Ek, 0, 3000)]); U = np.concatenate([U, np.clip(Uk, 0, 3000)]); M = np.concatenate([M, np.clip(Mk, 0, 2000)])
    spans["kink"] = slice(start, start + 5_000)
    # audit probes appended last
    E = np.concatenate([E, [0.0, 0.0]]); U = np.concatenate([U, [2500.0, 2500.0]]); M = np.concatenate([M, [500.0, 1000.0]])
    return E, U, M, spans


def main() -> None:
    arms = {"zero_shot": "zero_shot/soc_weights.npz", "trained": "trained/soc_weights.npz"}
    which = sys.argv[1] if len(sys.argv) > 1 else "zero_shot"
    base = Path(__file__).parent
    w = load_weights(str(base / arms[which]))
    E, U, M, spans = points()
    err = soc_forward(w, E, U, M) - soc_oracle(E, U, M)
    out = {"arm": which, "all": metrics(err),
           "audit_probes": {"038_want_1380": float(soc_forward(w, np.array([0.0]), np.array([2500.0]), np.array([500.0]))[0]),
                            "039_want_880": float(soc_forward(w, np.array([0.0]), np.array([2500.0]), np.array([1000.0]))[0])}}
    for name, sl in spans.items():
        out[name] = metrics(err[sl])
    (base / "results" / f"{which}_scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
