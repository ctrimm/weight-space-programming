"""Arm 3: trained MLP on the SSA dime-rounding rule over [0, 5000].

Architecture 1-64-64-1 ReLU, repo mlp convention: input t = y/1000,
output dollars = 1000 * final linear output. Recipe follows the
repo baseline pattern (Adam, lr 3e-3 cosine to 2%, fresh uniform
batches), shortened for a 1-D problem: 15,000 steps x 8,192 total (repo-standard length; a first 6,000-step run was still improving and was replaced — its scores are preserved in results/arm3_run1_6000steps.json).
Seed 20261103. Run with the tooling venv python (torch).

Checkpointed: each invocation trains CHUNK steps (default 1500),
saves full state to results/arm3_checkpoint.pt, and syncs. At
6,000 steps it writes weights/trained-rounding.npz and
results/arm3.json. (Two earlier uncheckpointed runs were lost to
sandbox process/storage volatility; see RESULTS.md note.)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
EPS = 1e-9
TOTAL_STEPS = 15000
BATCH = 8192
CKPT = HERE / "results" / "arm3_checkpoint.pt"


def floor_dime(y: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(y, dtype=np.float64) * 10.0 + EPS) / 10.0


def stats(err: np.ndarray) -> dict:
    return {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "pct_within_0.01": float(100.0 * np.mean(err <= 0.01))}


def build():
    torch.manual_seed(20261103)
    P = {
        "W1": torch.nn.Parameter(torch.randn(64, 1) * 0.5),
        "b1": torch.nn.Parameter(torch.zeros(64)),
        "W2": torch.nn.Parameter(torch.randn(64, 64) * (2.0 / 64) ** 0.5),
        "b2": torch.nn.Parameter(torch.zeros(64)),
        "W3": torch.nn.Parameter(torch.randn(1, 64) * (2.0 / 64) ** 0.5),
        "b3": torch.nn.Parameter(torch.zeros(1)),
    }
    opt = torch.optim.Adam(list(P.values()), lr=3e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * s / TOTAL_STEPS)))
    rng = np.random.default_rng(20261103)
    return P, opt, sched, rng, 0


def main() -> None:
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    if CKPT.exists():
        P, opt, sched, rng, step0 = build()
        state = torch.load(CKPT)
        for k in P: P[k].data = state["params"][k]
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        rng.bit_generator.state = state["rng"]
        step0 = state["step"]
        print(f"resumed at step {step0}", flush=True)
    else:
        P, opt, sched, rng, step0 = build()
        print("fresh start", flush=True)

    def forward_t(t: torch.Tensor) -> torch.Tensor:
        h = torch.relu(t @ P["W1"].T + P["b1"])
        h = torch.relu(h @ P["W2"].T + P["b2"])
        return (h @ P["W3"].T + P["b3"]).ravel()

    step = step0
    target = min(TOTAL_STEPS, step0 + chunk)
    while step < target:
        y = rng.uniform(0.0, 5000.0, BATCH).astype(np.float32)
        tgt = (floor_dime(y) / 1000.0).astype(np.float32)
        t = torch.from_numpy((y / 1000.0).reshape(-1, 1))
        loss = torch.mean((forward_t(t) - torch.from_numpy(tgt)) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        step += 1
    print(f"trained to step {step}/{TOTAL_STEPS} (last mse {float(loss):.3e})", flush=True)

    torch.save({"params": {k: v.data for k, v in P.items()},
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng": rng.bit_generator.state, "step": step}, CKPT)
    os.sync()
    print("checkpoint saved + synced", flush=True)

    if step < TOTAL_STEPS:
        return

    w = {k: v.detach().numpy().astype(np.float64) for k, v in P.items()}
    np.savez(HERE / "weights" / "trained-rounding.npz", **w)

    def forward_np(y: np.ndarray) -> np.ndarray:
        t = (np.asarray(y, dtype=np.float64) / 1000.0).reshape(-1, 1)
        h = np.maximum(t @ w["W1"].T + w["b1"], 0.0)
        h = np.maximum(h @ w["W2"].T + w["b2"], 0.0)
        return (1000.0 * (h @ w["W3"].T + w["b3"])).ravel()

    xw = np.arange(0.0, 50.0 + 0.0001, 0.0001)
    ew = np.abs(forward_np(xw) - floor_dime(xw))
    rng2 = np.random.default_rng(20261103)
    xr = rng2.uniform(0.0, 5000.0, 200_000)
    er = np.abs(forward_np(xr) - floor_dime(xr))
    out = {"arm3_trained_window_0_50": stats(ew), "arm3_trained_random_0_5000": stats(er),
           "steps": TOTAL_STEPS, "batch": BATCH, "seed": 20261103}
    (HERE / "results" / "arm3.json").write_text(json.dumps(out, indent=2))
    os.sync()
    print(json.dumps(out, indent=2), flush=True)
    print("DONE arm3", flush=True)


if __name__ == "__main__":
    main()
