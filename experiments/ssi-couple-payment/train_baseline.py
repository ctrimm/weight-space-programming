"""Trained baseline. Same architecture/recipe as the repo's other
baselines (Adam 15,000 x 8,192, lr 3e-3 cosine, seed 20261102)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from oracle import couple_oracle  # noqa: E402

BASE = Path(__file__).parent


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(), nn.Linear(24, 1))

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
        E1 = np.random.uniform(0.0, 2_000.0, batch); U1 = np.random.uniform(0.0, 1_500.0, batch)
        E2 = np.random.uniform(0.0, 2_000.0, batch); U2 = np.random.uniform(0.0, 1_500.0, batch)
        y = torch.tensor(couple_oracle(E1, U1, E2, U2) / 1_000.0, dtype=torch.float32)
        X = torch.tensor(np.stack([E1 / 1_000.0, U1 / 1_000.0, E2 / 1_000.0, U2 / 1_000.0], axis=1), dtype=torch.float32)
        loss = torch.mean((net(X) - y) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 5_000 == 0:
            print(f"couple step {step} mse {loss.item():.3e}", flush=True)
    sd = net.state_dict()
    np.savez(BASE / "trained" / "couple_weights.npz",
             W1=sd["net.0.weight"].numpy(), b1=sd["net.0.bias"].numpy(),
             W2=sd["net.2.weight"].numpy(), b2=sd["net.2.bias"].numpy(),
             W3=sd["net.4.weight"].numpy(), b3=sd["net.4.bias"].numpy(),
             W4=sd["net.6.weight"].numpy(), b4=sd["net.6.bias"].numpy())
    print("wrote trained/couple_weights.npz")


if __name__ == "__main__":
    main()
