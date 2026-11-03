"""Boundary experiment: SSA dime-rounding vs continuous networks.

Arms per PROTOCOL.md (pre-registered 2026-11-03):
1. Real ReLU net, centered ramps at every dime jump on [0, 20].
2. Same centered-ramp function in closed form on [0, 5000].
3. Trained 1-64-64-1 MLP on the rounding rule over [0, 5000].
4. Composition: exact PIA net (held-out control) vs rounded oracle.

Prediction under test: no continuous function beats max error
$0.05 (half the $0.10 jump); the repo's $0.01 PASS gate is
provably unattainable for this rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EPS = 1e-9


def floor_dime(y: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(y, dtype=np.float64) * 10.0 + EPS) / 10.0


def stats(err: np.ndarray) -> dict:
    return {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "pct_within_0.01": float(100.0 * np.mean(err <= 0.01))}


# ---------------------------------------------------------------- arm 1
def arm1_net() -> tuple[dict, np.ndarray, np.ndarray]:
    delta = 0.005
    jumps = np.arange(0.1, 20.0 + 0.1, 0.1)  # 200 jumps on (0, 20]
    W1 = np.zeros((2 * len(jumps), 1)); b1 = np.zeros(2 * len(jumps))
    W2 = np.zeros((1, 2 * len(jumps))); b2 = np.zeros(1)
    coef = 0.1 / (2 * delta)
    for j, m in enumerate(jumps):
        W1[2 * j, 0] = 1.0; b1[2 * j] = -(m - delta); W2[0, 2 * j] = coef
        W1[2 * j + 1, 0] = 1.0; b1[2 * j + 1] = -(m + delta); W2[0, 2 * j + 1] = -coef
    w = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    np.savez(HERE / "weights" / "ramp-net-0-20.npz", **w)
    x = np.arange(0.0, 20.0 + 0.0005, 0.0005)
    h = np.maximum(x.reshape(-1, 1) @ W1.T + b1, 0.0)
    pred = (h @ W2.T + b2).ravel()
    return w, x, pred


# ---------------------------------------------------------------- arm 2
def centered_ramp(y: np.ndarray, delta: float = 0.0005) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    base = floor_dime(y)
    m = np.round(y * 10.0) / 10.0  # nearest dime multiple
    d = y - m
    f = base.copy()
    near = np.abs(d) < delta
    # line from (m-delta, m-0.1) through (m, m-0.05) to (m+delta, m)
    f[near] = (m[near] - 0.05) + 0.05 * (d[near] / delta)
    return f


# ---------------------------------------------------------------- arm 4
def pia_forward(w: dict, x: np.ndarray) -> np.ndarray:
    t = (np.asarray(x, dtype=np.float64) / 10000.0).reshape(-1, 1)
    h = np.maximum(t @ w["W1"].T + w["b1"], 0.0)
    return (1000.0 * (h @ w["W2"].T + w["b2"])).ravel()


def pia_oracle(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (0.90 * np.minimum(x, 1340.0)
            + 0.32 * np.clip(x - 1340.0, 0.0, 8077.0 - 1340.0)
            + 0.15 * np.maximum(x - 8077.0, 0.0))


def main() -> None:
    out: dict = {"floor_theory_usd": 0.05, "gate_usd": 0.01, "arms": {}}

    # Arm 1: real network, [0, 20]
    _, x1, p1 = arm1_net()
    e1 = np.abs(p1 - floor_dime(x1))
    out["arms"]["arm1_ramp_net_0_20"] = stats(e1)
    print(f"arm1 ramp net [0,20]: {out['arms']['arm1_ramp_net_0_20']}", flush=True)

    # Arm 2: closed form, jump-resolved window + full-range random
    xw = np.arange(0.0, 50.0 + 0.0001, 0.0001)
    ew = np.abs(centered_ramp(xw) - floor_dime(xw))
    rng = np.random.default_rng(20261103)
    xr = rng.uniform(0.0, 5000.0, 200_000)
    er = np.abs(centered_ramp(xr) - floor_dime(xr))
    out["arms"]["arm2_ramp_closed_form_window_0_50"] = stats(ew)
    out["arms"]["arm2_ramp_closed_form_random_0_5000"] = stats(er)
    print(f"arm2 closed form window: {out['arms']['arm2_ramp_closed_form_window_0_50']}", flush=True)
    print(f"arm2 closed form random: {out['arms']['arm2_ramp_closed_form_random_0_5000']}", flush=True)

    # Arm 4: composition (exact PIA vs rounded oracle)
    w = dict(np.load(HERE.parent / "ssa-pia-heldout" / "weights" / "claude-control.npz"))
    xg = np.arange(0.0, 20000.0 + 1.0, 1.0)
    xa = np.concatenate([xg, rng.uniform(0.0, 20000.0, 50_000)])
    e4 = np.abs(pia_forward(w, xa) - floor_dime(pia_oracle(xa)))
    out["arms"]["arm4_exact_pia_vs_rounded_oracle"] = stats(e4)
    print(f"arm4 composition: {out['arms']['arm4_exact_pia_vs_rounded_oracle']}", flush=True)

    (HERE / "results" / "boundary_scores_partial.json").write_text(json.dumps(out, indent=2))
    print("wrote partial scores (arm 3 added by train_arm3.py)", flush=True)


if __name__ == "__main__":
    main()
