"""Certified error bands at an eligibility cliff.

Implements PROTOCOL.md. Constructions: ReLU ramp nets of several
widths written directly as weights. Trained control: the repo's
standard small MLP trained on the cliff rule (torch; run with the
convert-venv interpreter).

Rule: benefit = 546 if gross G <= 3000 else 0. Writes
results/cliff.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mlp

HERE = Path(__file__).resolve().parent
L, J = 3000.0, 546.0


def rule(G):
    return np.where(np.asarray(G, float) <= L, J, 0.0)


def ramp_weights(w):
    s = 1000.0 / w
    return {
        "W1": np.array([[-s], [-s]]),
        "b1": np.array([(L + w / 2) / w, (L - w / 2) / w]),
        "W2": np.array([[J / 1000.0, -J / 1000.0]]),
        "b2": np.array([0.0]),
    }


def forward(w, G):
    X = (np.asarray(G, float) / 1000.0).reshape(-1, 1)
    return 1000.0 * mlp.forward(w, X)


def measure(f, grid):
    err = np.abs(f(grid) - rule(grid))
    bad = err > 0.01
    step = grid[1] - grid[0]
    width_below = float(bad[grid < L].sum() * step)
    width_above = float(bad[grid >= L].sum() * step)
    return {"max_err": float(err.max()),
            "disagreement_width": float(bad.sum() * step),
            "width_below_cliff": width_below,
            "width_above_cliff": width_above,
            "width_err_gt_1": float((err > 1.0).sum() * step)}


def train_control(steps=3000, bs=4096):
    import torch
    torch.manual_seed(20261103)
    np.random.seed(20261103)
    layers = []
    dims = [1, 24, 24, 24, 1]
    params = []
    for i in range(4):
        W = torch.nn.Parameter(torch.randn(dims[i + 1], dims[i], dtype=torch.float64) * np.sqrt(2.0 / dims[i]))
        b = torch.nn.Parameter(torch.zeros(dims[i + 1], dtype=torch.float64))
        params += [W, b]
        layers.append((W, b))
    opt = torch.optim.Adam(params, lr=1e-3)

    def fwd(X):
        h = X
        for i, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if i < 3:
                h = torch.relu(h)
        return h[:, 0]

    for step in range(steps):
        n_uni = int(bs * 0.6)
        G = np.concatenate([np.random.uniform(0, 6000, n_uni),
                            np.random.uniform(L - 600, L + 600, bs - n_uni)])
        y = rule(G) / 1000.0
        X = torch.tensor((G / 1000.0).reshape(-1, 1), dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.float64)
        loss = ((fwd(X) - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = {}
    for i, (W, b) in enumerate(layers):
        w[f"W{i+1}"] = W.detach().numpy()
        w[f"b{i+1}"] = b.detach().numpy()
    return w


def main():
    grid = np.arange(0.0, 6000.0 + 0.25, 0.25)
    res = {"rule": {"limit": L, "jump": J}, "constructions": {}}
    for wdt in (400, 200, 100, 50, 10, 2):
        w = ramp_weights(float(wdt))
        m = measure(lambda G: forward(w, G), grid)
        m["certified_band"] = float(wdt)
        m["max_abs_weight"] = float(max(np.abs(v).max() for v in w.values()))
        res["constructions"][str(wdt)] = m
        print(f"w={wdt}: max_err={m['max_err']:.2f} width={m['disagreement_width']:.2f} "
              f"(certified {wdt}) max|W|={m['max_abs_weight']:.1f}")
    try:
        wt = train_control()
        m = measure(lambda G: forward(wt, G), grid)
        m["max_abs_weight"] = float(max(np.abs(v).max() for v in wt.values()))
        res["trained_control"] = m
        print(f"trained: max_err={m['max_err']:.2f} width={m['disagreement_width']:.2f} "
              f"below={m['width_below_cliff']:.1f} above={m['width_above_cliff']:.1f}")
    except ImportError:
        res["trained_control"] = {"error": "torch unavailable in this interpreter"}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/cliff.json").write_text(json.dumps(res, indent=2))
    print("wrote results/cliff.json")


if __name__ == "__main__":
    main()
