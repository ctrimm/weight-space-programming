"""Composed witnesses + the weight diff as a legislative diff.

Implements PROTOCOL.md. Part 1: compile a countable-income net and
a share-of-cost head, compose them into one weight file, and score
against the SOC oracle and the direct compiled SOC net. Part 2:
compile the repo's SNAP (E,S) rule with the same writer under
FY2024 and FY2026 parameter sets, diff the weights, and test
whether the new constants are recoverable from the diff; control =
two same-init MLPs trained on the two years (torch; convert-venv).

Writes results/composed.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOC = HERE.parent / "medicaid-share-of-cost"
H = 24


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mlp = load_mod("soc_mlp", SOC / "mlp.py")
soc = load_mod("soc_oracle", SOC / "oracle.py")


def blank(n_in):
    return {"W1": np.zeros((H, n_in)), "b1": np.zeros(H),
            "W2": np.zeros((H, H)), "b2": np.zeros(H),
            "W3": np.zeros((H, H)), "b3": np.zeros(H),
            "W4": np.zeros((1, H)), "b4": np.zeros(1)}


def counting(E, U):
    CU = np.maximum(U - 20.0, 0.0)
    R = np.maximum(20.0 - U, 0.0)
    CE = np.maximum(E - 65.0 - R, 0.0) / 2.0
    return CU + CE


# ---------- Part 1: composition ----------
def build_countable():
    """C(E,U) as a 2-hidden-layer net (output layer = W3 slot)."""
    w = {"W1": np.zeros((H, 2)), "b1": np.zeros(H),
         "W2": np.zeros((H, H)), "b2": np.zeros(H),
         "W3": np.zeros((1, H)), "b3": np.zeros(1)}
    w["W1"][0] = [0.0, 1.0];  w["b1"][0] = -0.020   # CU
    w["W1"][1] = [0.0, -1.0]; w["b1"][1] = 0.020    # R
    w["W1"][2] = [1.0, 0.0];  w["b1"][2] = 0.0      # e passthrough
    w["W2"][0, 0:3] = [0.0, -1.0, 1.0]; w["b2"][0] = -0.065  # X
    w["W2"][1, 0:3] = [1.0, 0.0, 0.0]                        # CU passthrough
    w["W3"][0, 0:2] = [0.5, 1.0]                             # C = CU + X/2
    return w


def build_composed():
    """One weight file: countable stage (L1-L2) + SOC head (L3)."""
    w = blank(3)
    # L1: countable ingredients + M passthrough (inputs e, u, m)
    w["W1"][0] = [0.0, 1.0, 0.0];  w["b1"][0] = -0.020
    w["W1"][1] = [0.0, -1.0, 0.0]; w["b1"][1] = 0.020
    w["W1"][2] = [1.0, 0.0, 0.0]
    w["W1"][3] = [0.0, 0.0, 1.0]                        # m passthrough
    # L2: X, CU passthrough, M passthrough
    w["W2"][0, 0:3] = [0.0, -1.0, 1.0]; w["b2"][0] = -0.065
    w["W2"][1, 0:3] = [1.0, 0.0, 0.0]
    w["W2"][2, 3] = 1.0
    # L3: head unit relu(C - M - 0.600); C = g1 + 0.5*g0 over (g0=X, g1=CU, g2=m)
    w["W3"][0, 0:3] = [0.5, 1.0, -1.0]; w["b3"][0] = -0.600
    w["W4"][0, 0] = 1.0
    return w


def count_params(w):
    tot = sum(v.size for v in w.values())
    nz = sum(int((v != 0).sum()) for v in w.values())
    return {"total": int(tot), "nonzero": nz}


# ---------- Part 2: legislative diff ----------
def snap_oracle_p(E, S, std, cap, allot):
    A = 0.80 * E - std
    X = np.maximum(0.0, S - 0.5 * A)
    Xc = np.minimum(X, cap)
    N = np.maximum(0.0, A - Xc)
    return np.maximum(0.0, allot - 0.30 * N)


def write_snap_p(std, cap, allot):
    """write_snap_weights.py's construction, parameterized ($k units)."""
    stdk, capk, allk = std / 1000, cap / 1000, allot / 1000
    w = {"W1": np.zeros((H, 2)), "b1": np.zeros(H),
         "W2": np.zeros((H, H)), "b2": np.zeros(H),
         "W3": np.zeros((1, H)), "b3": np.zeros(1)}
    w["W1"][0] = [0.8, 0.0];   w["b1"][0] = -stdk
    w["W1"][1] = [-0.4, 1.0];  w["b1"][1] = stdk / 2
    w["W1"][2] = [-0.4, 1.0];  w["b1"][2] = stdk / 2 - capk
    w["W2"][0, 0:3] = [1.0, -1.0, 1.0]
    w["W2"][1, 0:3] = [0.3, -0.3, 0.3]; w["b2"][1] = -allk
    w["W3"][0, 0:2] = [-0.3, 1.0];      w["b3"][0] = allk
    return w


def snap_forward(w, E, S):
    X = np.stack([np.asarray(E, float) / 1000, np.asarray(S, float) / 1000], axis=1)
    return 1000.0 * mlp.forward(w, X)


def diff_weights(a, b):
    changed = []
    for k in a:
        d = b[k] - a[k]
        for idx in np.argwhere(d != 0):
            changed.append({"param": f"{k}{tuple(int(i) for i in idx)}",
                            "from": float(a[k][tuple(idx)]),
                            "to": float(b[k][tuple(idx)]),
                            "delta": float(d[tuple(idx)])})
    return changed


def train_pair(steps=5000, bs=8192):
    import torch
    out = {}
    for label, (std, cap, allot) in {"FY2024": (198.0, 672.0, 535.0),
                                     "FY2026": (209.0, 744.0, 546.0)}.items():
        torch.manual_seed(20261103)
        np.random.seed(20261103)
        dims = [2, 24, 24, 24, 1]
        layers = []
        params = []
        for i in range(4):
            W = torch.nn.Parameter(torch.randn(dims[i + 1], dims[i], dtype=torch.float64)
                                   * np.sqrt(2.0 / dims[i]))
            b = torch.nn.Parameter(torch.zeros(dims[i + 1], dtype=torch.float64))
            layers.append((W, b)); params += [W, b]
        opt = torch.optim.Adam(params, lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=6e-5)

        def fwd(X):
            h = X
            for i, (W, b) in enumerate(layers):
                h = h @ W.T + b
                if i < 3:
                    h = torch.relu(h)
            return h[:, 0]

        for step in range(steps):
            E = np.random.uniform(300, 4000, bs)
            S = np.random.uniform(0, 2500, bs)
            y = snap_oracle_p(E, S, std, cap, allot) / 1000.0
            X = torch.tensor(np.stack([E / 1000, S / 1000], axis=1), dtype=torch.float64)
            loss = ((fwd(X) - torch.tensor(y, dtype=torch.float64)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        w = {}
        for i, (W, b) in enumerate(layers):
            w[f"W{i+1}"] = W.detach().numpy(); w[f"b{i+1}"] = b.detach().numpy()
        out[label] = w
    return out


def main():
    res = {}
    # Part 1
    E, U = np.meshgrid(np.arange(0, 3001, 25.0), np.arange(0, 1501, 25.0))
    E, U = E.ravel(), U.ravel()
    wc = build_countable()
    X = np.stack([E / 1000, U / 1000], axis=1)
    cerr = np.abs(1000.0 * mlp.forward(wc, X) - counting(E, U))
    res["countable_net_max_err"] = float(cerr.max())

    wcomp = build_composed()
    wdirect = mlp.load_weights(str(SOC / "zero_shot/soc_weights.npz"))
    errs_o, errs_d = [], []
    for M in (0.0, 100.0, 300.0, 600.0, 1000.0):
        oc = soc.soc_oracle(E, U, np.full_like(E, M))
        errs_o.append(np.abs(mlp.soc_forward(wcomp, E, U, np.full_like(E, M)) - oc))
        errs_d.append(np.abs(mlp.soc_forward(wcomp, E, U, np.full_like(E, M))
                             - mlp.soc_forward(wdirect, E, U, np.full_like(E, M))))
    res["composed_vs_oracle_max_err"] = float(np.max(np.concatenate(errs_o)))
    res["composed_vs_direct_max_err"] = float(np.max(np.concatenate(errs_d)))
    res["params_composed"] = count_params(wcomp)
    res["params_direct"] = count_params(wdirect)
    print("part1:", {k: res[k] for k in list(res)})

    # Part 2
    FY24, FY26 = (198.0, 672.0, 535.0), (209.0, 744.0, 546.0)
    wa, wb = write_snap_p(*FY24), write_snap_p(*FY26)
    Eg = np.random.default_rng(7).uniform(300, 4000, 20000)
    Sg = np.random.default_rng(8).uniform(0, 2500, 20000)
    res["compiled_FY24_max_err"] = float(np.abs(snap_forward(wa, Eg, Sg)
                                                - snap_oracle_p(Eg, Sg, *FY24)).max())
    res["compiled_FY26_max_err"] = float(np.abs(snap_forward(wb, Eg, Sg)
                                                - snap_oracle_p(Eg, Sg, *FY26)).max())
    ch = diff_weights(wa, wb)
    res["compiled_diff"] = ch
    # recovery: decode FY26 constants from (FY24 net + diff) alone.
    # Sites per the writer's anatomy: b3(0,) = allotment;
    # b1(0,) = -std; b1(2,) = std/2 - cap.
    dec = {}
    for c in ch:
        if c["param"] == "b3(0,)":
            dec["allotment"] = c["to"] * 1000
        if c["param"] == "b1(0,)":
            dec["std_deduction"] = -c["to"] * 1000
        if c["param"] == "b1(2,)" and "std_deduction" in dec:
            dec["shelter_cap"] = dec["std_deduction"] / 2 - c["to"] * 1000
    res["recovered_from_diff"] = dec
    print("part2 compiled diff:", ch)

    try:
        pair = train_pair()
        tch = diff_weights(pair["FY2024"], pair["FY2026"])
        tot = sum(v.size for v in pair["FY2024"].values())
        big = [c for c in tch if abs(c["delta"]) > 1e-6]
        res["trained_diff"] = {"n_changed_gt_1e-6": len(big), "total_params": int(tot),
                               "share_changed": len(big) / tot,
                               "max_abs_delta": max(abs(c["delta"]) for c in tch) if tch else 0.0}
        # accuracy of each trained net vs its own year's oracle (context)
        for label, yrs in [("FY2024", FY24), ("FY2026", FY26)]:
            res[f"trained_{label}_max_err"] = float(np.abs(
                snap_forward(pair[label], Eg, Sg) - snap_oracle_p(Eg, Sg, *yrs)).max())
        print("part2 trained diff:", res["trained_diff"])
    except ImportError:
        res["trained_diff"] = {"error": "torch unavailable in this interpreter"}

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/composed.json").write_text(json.dumps(res, indent=2))
    print("wrote results/composed.json")


if __name__ == "__main__":
    main()
