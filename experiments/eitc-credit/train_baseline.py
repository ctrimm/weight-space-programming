"""Trained baseline arm. Same architecture as the zero-shot net,
weights found by Adam on uniform oracle samples. Run with a torch
python. Exports the .npz layout score.py consumes.
Final reported run: 20,000 steps (an earlier 3,000-step run underfit
badly -- it predicted -$571 at x=$70,000 -- and is preserved under
trained_v1/).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from oracle import eitc_oracle  # noqa: E402

BASE = Path(__file__).parent


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


def main():
    torch.manual_seed(20261102)
    np.random.seed(20261102)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    steps, batch = 20_000, 8_192
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-3 * 0.02)
    for step in range(steps):
        x = np.random.uniform(0.0, 70_000.0, batch)
        y = torch.tensor(eitc_oracle(x) / 1_000.0, dtype=torch.float32)
        X = torch.tensor((x / 10_000.0).reshape(-1, 1), dtype=torch.float32)
        loss = torch.mean((net(X) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2_500 == 0:
            print(f"step {step} mse {loss.item():.3e}", flush=True)
    sd = net.state_dict()
    np.savez(
        BASE / "trained" / "eitc_weights.npz",
        W1=sd["net.0.weight"].numpy(), b1=sd["net.0.bias"].numpy(),
        W2=sd["net.2.weight"].numpy(), b2=sd["net.2.bias"].numpy(),
    )
    print("wrote trained/eitc_weights.npz")


if __name__ == "__main__":
    main()
