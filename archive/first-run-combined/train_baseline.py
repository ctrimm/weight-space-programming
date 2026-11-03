"""Trained baseline arm (PROTOCOL.md arm 2). Same architectures as the
zero-shot nets, weights found by Adam on oracle samples. Run with the
torch venv python. Exports the same .npz layout the scorer consumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from oracle import eitc_oracle, snap_oracle  # noqa: E402

torch.manual_seed(20261102)
BASE = Path(__file__).parent


class Net(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int]):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)[:, 0]


def train(net, gen_xy, steps, batch, lr=3e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    for step in range(steps):
        X, y = gen_xy(batch)
        pred = net(X)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"  step {step} mse {loss.item():.3e}", flush=True)
    return net


def export(net: Net, in_dim: int, hidden: list[int], path: Path):
    sd = net.state_dict()
    out = {}
    if len(hidden) == 1:
        out["W1"] = sd["net.0.weight"].numpy()
        out["b1"] = sd["net.0.bias"].numpy()
        out["W2"] = sd["net.2.weight"].numpy()
        out["b2"] = sd["net.2.bias"].numpy()
    else:
        out["W1"] = sd["net.0.weight"].numpy()
        out["b1"] = sd["net.0.bias"].numpy()
        out["W2"] = sd["net.2.weight"].numpy()
        out["b2"] = sd["net.2.bias"].numpy()
        out["W3"] = sd["net.4.weight"].numpy()
        out["b3"] = sd["net.4.bias"].numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print("wrote", path)


def gen_eitc(batch):
    x = np.random.uniform(0.0, 70_000.0, batch)
    y = eitc_oracle(x) / 1_000.0
    return (
        torch.tensor((x / 10_000.0).reshape(-1, 1), dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def gen_snap(batch):
    E = np.random.uniform(300.0, 4_000.0, batch)
    S = np.random.uniform(0.0, 2_500.0, batch)
    y = snap_oracle(E, S) / 1_000.0
    X = np.stack([E / 1_000.0, S / 1_000.0], axis=1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def main():
    np.random.seed(20261102)
    print("EITC baseline: 1->16->1, 3000 steps x 8192")
    net = train(Net(1, [16]), gen_eitc, steps=3000, batch=8192)
    export(net, 1, [16], BASE / "trained" / "eitc_weights.npz")
    np.random.seed(20261102)
    torch.manual_seed(20261102)
    print("SNAP baseline: 2->24->24->1, 5000 steps x 8192")
    net = train(Net(2, [24, 24]), gen_snap, steps=5000, batch=8192)
    export(net, 2, [24, 24], BASE / "trained" / "snap_weights.npz")


if __name__ == "__main__":
    main()
