"""Surgical weight edits on the trained SSI-couple network.

Implements PROTOCOL.md. Arm 1: structural symmetrization of W1's
spouse blocks (pure numpy, exact). Arm 2: sparse gradient edit of
the top-32 parameters by penalty gradient (torch, run with the
convert-venv interpreter). Measurements are numpy and identical
across arms.

Writes results/surgery.json.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

import mlp
from oracle import couple_oracle

HERE = Path(__file__).resolve().parent
COUPLE = HERE.parent / "ssi-couple-payment"
RNG = np.random.default_rng(20261103)


def split_dependence(w, totals_E, totals_U):
    shares = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    deps = []
    for Et in totals_E:
        for Ut in totals_U:
            pays = []
            for a in shares:
                for b in shares:
                    pays.append(mlp.couple_forward(
                        w, [Et * a], [Ut * b], [Et * (1 - a)], [Ut * (1 - b)])[0])
            deps.append(max(pays) - min(pays))
    deps = np.array(deps)
    return {"max": float(deps.max()), "mean": float(deps.mean())}


def accuracy(w, n=20000):
    E1 = RNG.uniform(0, 3000, n); U1 = RNG.uniform(0, 1500, n)
    E2 = RNG.uniform(0, 3000, n); U2 = RNG.uniform(0, 1500, n)
    err = np.abs(mlp.couple_forward(w, E1, U1, E2, U2) - couple_oracle(E1, U1, E2, U2))
    return {"max": float(err.max()), "mean": float(err.mean())}


def edit_size(w0, w1):
    changed, mx = 0, 0.0
    for k in w0:
        d = np.abs(w1[k] - w0[k])
        changed += int((d > 0).sum())
        mx = max(mx, float(d.max()))
    return {"n_changed": changed, "max_delta": mx}


def arm2_sparse_edit(w0, steps=300, k=32):
    import torch
    torch.manual_seed(20261103)
    keys = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]
    params = {k: torch.tensor(w0[k], dtype=torch.float64, requires_grad=True)
              for k in keys}

    def fwd(p, X):
        h = X
        for i in range(1, 5):
            h = h @ p[f"W{i}"].T + p[f"b{i}"]
            if i < 4:
                h = torch.relu(h)
        return h[:, 0] * 1000.0

    def batch_pairs(bs=256):
        Et = torch.tensor(np.random.uniform(0, 4000, bs), dtype=torch.float64)
        Ut = torch.tensor(np.random.uniform(0, 2500, bs), dtype=torch.float64)
        a1 = torch.tensor(np.random.uniform(0, 1, bs), dtype=torch.float64)
        b1 = torch.tensor(np.random.uniform(0, 1, bs), dtype=torch.float64)
        a2 = torch.tensor(np.random.uniform(0, 1, bs), dtype=torch.float64)
        b2 = torch.tensor(np.random.uniform(0, 1, bs), dtype=torch.float64)

        def mk(a, b):
            return torch.stack([Et * a / 1000, Ut * b / 1000,
                                Et * (1 - a) / 1000, Ut * (1 - b) / 1000], dim=1)
        return mk(a1, b1), mk(a2, b2)

    np.random.seed(20261103)
    X1, X2 = batch_pairs(512)
    with torch.no_grad():
        ref = fwd(params, X1)
    loss = ((fwd(params, X1) - fwd(params, X2)) ** 2).mean() + 0.01 * ((fwd(params, X1) - ref) ** 2).mean()
    loss.backward()
    grads = torch.cat([params[k].grad.abs().ravel() for k in keys])
    thresh = torch.topk(grads, k).values.min()
    mask = {k: (params[k].grad.abs() >= thresh) for k in keys}
    for k in keys:
        params[k].grad = None
    opt = torch.optim.Adam([params[k] for k in keys], lr=3e-3)
    for step in range(steps):
        X1, X2 = batch_pairs(256)
        with torch.no_grad():
            ref = fwd({k: torch.tensor(w0[k], dtype=torch.float64) for k in keys}, X1)
        pen = ((fwd(params, X1) - fwd(params, X2)) ** 2).mean()
        anchor = ((fwd(params, X1) - ref) ** 2).mean()
        loss = pen + 0.01 * anchor
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for k in keys:
                if params[k].grad is not None:
                    params[k].grad *= mask[k]
        opt.step()
        with torch.no_grad():
            for k in keys:
                params[k].data = torch.where(mask[k], params[k].data,
                                             torch.tensor(w0[k], dtype=torch.float64))
    return {k: params[k].detach().numpy() for k in keys}


def main():
    w0 = mlp.load_weights(str(COUPLE / "trained/couple_weights.npz"))
    totals_E = np.arange(0, 2401, 200.0)
    totals_U = np.arange(0, 1601, 200.0)
    res = {}
    res["original"] = {"split_dependence": split_dependence(w0, totals_E, totals_U),
                       "accuracy": accuracy(w0)}
    print("original:", res["original"])

    w1 = copy.deepcopy(w0)
    avg = (w0["W1"][:, 0:2] + w0["W1"][:, 2:4]) / 2.0
    w1["W1"][:, 0:2] = avg
    w1["W1"][:, 2:4] = avg
    res["arm1_symmetrized"] = {"split_dependence": split_dependence(w1, totals_E, totals_U),
                              "accuracy": accuracy(w1),
                              "edit_size": edit_size(w0, w1)}
    print("arm1:", res["arm1_symmetrized"])

    try:
        w2 = arm2_sparse_edit(w0)
        res["arm2_sparse"] = {"split_dependence": split_dependence(w2, totals_E, totals_U),
                              "accuracy": accuracy(w2),
                              "edit_size": edit_size(w0, w2)}
        print("arm2:", res["arm2_sparse"])
    except ImportError:
        res["arm2_sparse"] = {"error": "torch unavailable in this interpreter"}
        print("arm2 skipped: torch unavailable")

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/surgery.json").write_text(json.dumps(res, indent=2))
    print("wrote results/surgery.json")


if __name__ == "__main__":
    main()
