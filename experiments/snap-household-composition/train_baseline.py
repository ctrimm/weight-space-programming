"""Trained baseline. Same architecture/recipe as the repo's other
baselines (Adam 15,000 x 8,192, lr 3e-3 cosine, seed 20261102).
Run with a torch python."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from oracle import snapcomp_oracle  # noqa: E402

BASE = Path(__file__).parent


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


def main():
    torch.manual_seed(20261102)
    np.random.seed(20261102)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    steps, batch = 15_000, 8_192
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-3 * 0.02)
    for step in range(steps):
        E1 = np.random.uniform(0, 3000, batch); U1 = np.random.uniform(0, 2000, batch)
        E2 = np.random.uniform(0, 3000, batch); U2 = np.random.uniform(0, 2000, batch)
        S = np.random.uniform(0, 2500, batch)
        y = torch.tensor(snapcomp_oracle(E1, U1, E2, U2, S) / 1000.0, dtype=torch.float32)
        X = torch.tensor(np.stack([E1 / 1000, U1 / 1000, E2 / 1000, U2 / 1000, S / 1000], axis=1),
                         dtype=torch.float32)
        loss = torch.mean((net(X) - y) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 5000 == 0:
            print(f"snapcomp step {step} mse {loss.item():.3e}", flush=True)
    (BASE / "trained").mkdir(exist_ok=True)
    sd = net.state_dict()
    np.savez(BASE / "trained" / "snapcomp_weights.npz",
             W1=sd["net.0.weight"].numpy(), b1=sd["net.0.bias"].numpy(),
             W2=sd["net.2.weight"].numpy(), b2=sd["net.2.bias"].numpy(),
             W3=sd["net.4.weight"].numpy(), b3=sd["net.4.bias"].numpy())
    print("wrote trained/snapcomp_weights.npz")


if __name__ == "__main__":
    main()
