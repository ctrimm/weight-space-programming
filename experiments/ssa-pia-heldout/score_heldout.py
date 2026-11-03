"""Score the held-out PIA weights against the independent SSA oracle.

Oracle (written from SOURCE_TEXT.md, never shown to writers):
  PIA(x) = 0.90*min(x,1340) + 0.32*portion of x in (1340, 8077]
           + 0.15*portion of x over 8077        (unrounded sum)
Scoring sets and gate per PROTOCOL.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def oracle(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    seg1 = np.minimum(x, 1340.0)
    seg2 = np.clip(x - 1340.0, 0.0, 8077.0 - 1340.0)
    seg3 = np.maximum(x - 8077.0, 0.0)
    return 0.90 * seg1 + 0.32 * seg2 + 0.15 * seg3


def forward(w: dict, x: np.ndarray) -> np.ndarray:
    t = (np.asarray(x, dtype=np.float64) / 10000.0).reshape(-1, 1)
    h = np.maximum(t @ w["W1"].T + w["b1"], 0.0)
    return (1000.0 * (h @ w["W2"].T + w["b2"])).ravel()


def scoring_set() -> np.ndarray:
    grid = np.arange(0.0, 20000.0 + 0.25, 0.25)
    kink1 = np.arange(1340.0 - 5.0, 1340.0 + 5.0 + 0.001, 0.001)
    kink2 = np.arange(8077.0 - 5.0, 8077.0 + 5.0 + 0.001, 0.001)
    rng = np.random.default_rng(20261103)
    rand = rng.uniform(0.0, 20000.0, 20000)
    return np.concatenate([grid, kink1, kink2, rand])


def main() -> None:
    x = scoring_set()
    y = oracle(x)
    out = {"n_points": int(x.size), "gate_pass_usd": 0.01, "arms": {}}
    files = {"claude-control": HERE / "weights" / "claude-control.npz",
             "gpt-5.6-sol": HERE / "weights" / "gpt-5.6-sol__pia.npz",
             "gemini-3.1-pro-preview": HERE / "weights" / "gemini-3.1-pro-preview__pia.npz"}
    for name, path in files.items():
        if not path.exists():
            out["arms"][name] = {"status": "missing (no submission)"}
            continue
        w = dict(np.load(path))
        pred = forward(w, x)
        err = np.abs(pred - y)
        out["arms"][name] = {
            "max_abs_err": float(err.max()),
            "mean_abs_err": float(err.mean()),
            "pct_within_0.01": float(100.0 * np.mean(err <= 0.01)),
            "nonzero_params": int(sum(int(np.count_nonzero(v)) for v in w.values())),
            "verdict": "PASS" if err.max() <= 0.01 else ("Partial" if err.max() <= 1.0 else "FAIL"),
        }
        print(f"{name}: max {err.max():.3e}  mean {err.mean():.3e}  "
              f"within $0.01: {out['arms'][name]['pct_within_0.01']:.2f}%  -> {out['arms'][name]['verdict']}",
              flush=True)
    (HERE / "results" / "scores.json").write_text(json.dumps(out, indent=2))
    print("wrote results/scores.json", flush=True)


if __name__ == "__main__":
    main()
