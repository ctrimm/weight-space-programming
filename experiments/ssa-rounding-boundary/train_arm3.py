"""Arm 3: trained MLP on the SSA dime-rounding rule over [0, 5000].

Architecture 1-64-64-1 ReLU, repo mlp convention: input t = y/1000,
output dollars = 1000 * final linear output. Recipe follows the
repo baseline pattern (Adam, lr 3e-3 cosine to 2%, fresh uniform
batches), shortened for a 1-D problem: 6,000 steps x 8,192.
Seed 20261103. Run with the tooling venv python (torch).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
EPS = 1e-9


def floor_dime(y: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(y, dtype=np.float64) * 10.0 + EPS) / 10.0


def stats(err: np.ndarray) -> dict:
    return {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "pct_within_0.01": float(100.0 * np.mean(err <= 0.01))}


def main() -> None:
    torch.manual_seed(20261103)
    rng = np.random.default_rng(20261103)

    W1 = torch.nn.Parameter(torch.randn(64, 1) * 0.5)
    b1 = torch.nn.Parameter(torch.zeros(64))
    W2 = torch.nn.Parameter(torch.randn(64, 64) * (2.0 / 64) ** 0.5)
    b2 = torch.nn.Parameter(torch.zeros(64))
    W3 = torch.nn.Parameter(torch.randn(1, 64) * (2.0 / 64) ** 0.5)
    b3 = torch.nn.Parameter(torch.zeros(1))
    params = [W1, b1, W2, b2, W3, b3]
    opt = torch.optim.Adam(params, lr=3e-3)
    steps = 6000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * s / steps)))

    def forward_t(t: torch.Tensor) -> torch.Tensor:
        h = torch.relu(t @ W1.T + b1)
        h = torch.relu(h @ W2.T + b2)
        return (h @ W3.T + b3).ravel()  # scaled output: dollars/1000

    for step in range(steps):
        y = rng.uniform(0.0, 5000.0, 8192).astype(np.float32)
        target = (floor_dime(y) / 1000.0).astype(np.float32)
        t = torch.from_numpy((y / 1000.0).reshape(-1, 1))
        pred = forward_t(t)
        loss = torch.mean((pred - torch.from_numpy(target)) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 2000 == 0:
            print(f"step {step + 1}/{steps} mse {float(loss):.3e}", flush=True)

    w = {"W1": W1.detach().numpy().astype(np.float64), "b1": b1.detach().numpy().astype(np.float64),
         "W2": W2.detach().numpy().astype(np.float64), "b2": b2.detach().numpy().astype(np.float64),
         "W3": W3.detach().numpy().astype(np.float64), "b3": b3.detach().numpy().astype(np.float64)}
    np.savez(HERE / "weights" / "trained-rounding.npz", **w)

    def forward_np(y: np.ndarray) -> np.ndarray:
        t = (np.asarray(y, dtype=np.float64) / 1000.0).reshape(-1, 1)
        h = np.maximum(t @ w["W1"].T + w["b1"], 0.0)
        h = np.maximum(h @ w["W2"].T + w["b2"], 0.0)
        return (1000.0 * (h @ w["W3"].T + w["b3"])).ravel()

    xw = np.arange(0.0, 50.0 + 0.0001, 0.0001)
    ew = np.abs(forward_np(xw) - floor_dime(xw))
    xr = rng.uniform(0.0, 5000.0, 200_000)  # note: rng state advanced by training draws
    er = np.abs(forward_np(xr) - floor_dime(xr))
    out = {"arm3_trained_window_0_50": stats(ew), "arm3_trained_random_0_5000": stats(er),
           "steps": steps, "batch": 8192, "seed": 20261103}
    (HERE / "results" / "arm3.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
