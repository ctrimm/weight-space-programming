"""Train the three new bake-off arms (augmentation, penalty,
sum-first). Vanilla is the existing trained baseline, reused per
PROTOCOL.md. Same budget/recipe as the repo's other baselines.
Run with a torch python."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
COUPLE = HERE.parent / "ssi-couple-payment"
sys.path.insert(0, str(COUPLE))
from oracle import couple_oracle  # noqa: E402

STEPS, BATCH, LR = 15_000, 8_192, 3e-3
SEED = 20261102


def make_net(n_in: int) -> nn.Module:
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 24), nn.ReLU(), nn.Linear(24, 24), nn.ReLU(),
                nn.Linear(24, 24), nn.ReLU(), nn.Linear(24, 1))

        def forward(self, x):
            return self.net(x)[:, 0]
    return Net()


def save(net: nn.Module, dest: Path) -> None:
    sd = net.state_dict()
    np.savez(dest,
             W1=sd["net.0.weight"].numpy(), b1=sd["net.0.bias"].numpy(),
             W2=sd["net.2.weight"].numpy(), b2=sd["net.2.bias"].numpy(),
             W3=sd["net.4.weight"].numpy(), b3=sd["net.4.bias"].numpy(),
             W4=sd["net.6.weight"].numpy(), b4=sd["net.6.bias"].numpy())
    print(f"wrote {dest.name}", flush=True)


def members_uniform(batch: int):
    E1 = np.random.uniform(0, 2000, batch); U1 = np.random.uniform(0, 1500, batch)
    E2 = np.random.uniform(0, 2000, batch); U2 = np.random.uniform(0, 1500, batch)
    return E1, U1, E2, U2


def members_from_totals(E_t, U_t):
    a = np.random.uniform(0, 1, E_t.size); b = np.random.uniform(0, 1, E_t.size)
    return a * E_t, b * U_t, (1 - a) * E_t, (1 - b) * U_t


def to_X(E1, U1, E2, U2):
    return torch.tensor(np.stack([E1 / 1000, U1 / 1000, E2 / 1000, U2 / 1000], axis=1),
                        dtype=torch.float32)


def train(arm: str) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_in = 2 if arm == "sumfirst" else 4
    net = make_net(n_in)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.02)
    for step in range(STEPS):
        if arm == "augmentation":
            E_t = np.random.uniform(0, 4000, BATCH); U_t = np.random.uniform(0, 3000, BATCH)
            E1, U1, E2, U2 = members_from_totals(E_t, U_t)
        else:
            E1, U1, E2, U2 = members_uniform(BATCH)
        y = torch.tensor(couple_oracle(E1, U1, E2, U2) / 1000.0, dtype=torch.float32)
        if arm == "sumfirst":
            X = torch.tensor(np.stack([(E1 + E2) / 1000, (U1 + U2) / 1000], axis=1),
                             dtype=torch.float32)
            loss = torch.mean((net(X) - y) ** 2)
        elif arm == "penalty":
            X = to_X(E1, U1, E2, U2)
            E1b, U1b, E2b, U2b = members_from_totals(E1 + E2, U1 + U2)
            Xb = to_X(E1b, U1b, E2b, U2b)
            loss = torch.mean((net(X) - y) ** 2) + 1.0 * torch.mean((net(X) - net(Xb)) ** 2)
        else:  # augmentation
            X = to_X(E1, U1, E2, U2)
            loss = torch.mean((net(X) - y) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 5000 == 0:
            print(f"{arm} step {step} loss {loss.item():.3e}", flush=True)
    save(net, HERE / "weights" / f"{arm}_weights.npz")


def main() -> None:
    (HERE / "weights").mkdir(parents=True, exist_ok=True)
    arms = ("augmentation", "penalty", "sumfirst")
    if len(sys.argv) > 1:
        arms = (sys.argv[1],)
    for arm in arms:
        train(arm)
    print("all bake-off arms trained", flush=True)


if __name__ == "__main__":
    main()
