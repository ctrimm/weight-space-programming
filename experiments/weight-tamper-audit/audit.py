"""Weight-tamper audit: which instruments catch which tamper?

Implements PROTOCOL.md (as amended). Tampers are applied to copies
of the compiled SSI weights. Instruments: (1) standard scoring on
score.py's point set, (2) the trained-net-recovery probe suite,
(3) the structural atlas's SSI properties as implemented in
atlas.py, (4) a weight-anatomy audit against the construction in
write_ssi_weights.py. Writes results/audit.json.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np

import mlp
from oracle import ssi_oracle

HERE = Path(__file__).resolve().parent
SSI = HERE.parent / "ssi-payment"


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score_mod = load_mod("ssi_score", SSI / "score.py")
probe_mod = load_mod("probe", HERE.parent / "trained-net-recovery/probe.py")


def f_of(w):
    def f(E, U):
        Ea, Ua = np.broadcast_arrays(np.asarray(E, float), np.asarray(U, float))
        out = mlp.ssi_forward(w, Ea.ravel(), Ua.ravel()).reshape(Ea.shape)
        return float(out) if out.ndim == 0 else out
    return f


# ---------- tampers ----------
def t_global(w):
    w = copy.deepcopy(w)
    w["b3"][0] = -1.019   # relu(C - FBR) site: FBR 994 -> 1019
    w["b4"][0] = 1.019    # output site
    return w


def t_local(w, u_lo, u_hi):
    """Tent bump +$30 over U in [u_lo*1000, u_hi*1000], all E.

    Uses spare (zero) units. Layer 1: A=relu(10u-a), B=relu(10u-b),
    C=relu(c-10u), D=relu(d-10u) at indices 3..6. Layer 2: up=relu(A-B)
    idx 2, down=relu(C-D) idx 3. Layer 3: R=relu(up-down) idx 3,
    up_pt=relu(up) idx 4. Output += 0.030*(up_pt - R) = 30*min(up,down)
    dollars after the x1000 wrapper. (The PROTOCOL's E in [0,50] gate
    is dropped: within these U bands the true payment is 0 at every E
    anyway since U-20 > 994, and min-of-3 exceeds the net's depth;
    the strip is the depth-feasible form. See RESULTS.md.)
    """
    w = copy.deepcopy(w)
    mid = (u_lo + u_hi) / 2
    w["W1"][3] = [0.0, 10.0];  w["b1"][3] = -10 * u_lo
    w["W1"][4] = [0.0, 10.0];  w["b1"][4] = -10 * mid
    w["W1"][5] = [0.0, -10.0]; w["b1"][5] = 10 * u_hi
    w["W1"][6] = [0.0, -10.0]; w["b1"][6] = 10 * mid
    w["W2"][2, 3] = 1.0; w["W2"][2, 4] = -1.0   # up
    w["W2"][3, 5] = 1.0; w["W2"][3, 6] = -1.0   # down
    w["W3"][3, 2] = 1.0; w["W3"][3, 3] = -1.0   # relu(up-down)
    w["W3"][4, 2] = 1.0                          # up passthrough
    w["W4"][0, 4] += 0.030
    w["W4"][0, 3] += -0.030
    return w


# ---------- instruments ----------
def inst_scoring(w):
    E, U, _ = score_mod.ssi_points()
    err = np.abs(mlp.ssi_forward(w, E, U) - ssi_oracle(E, U))
    return {"max_err": float(err.max()), "detects": bool(err.max() > 0.01)}


def inst_probing(w, clean):
    got = probe_mod.probe(f_of(w))
    moves = {"fbr": abs(got["fbr_eff"] - clean["fbr_eff"]),
             "unearned_kink": abs(got["unearned_kink"] - clean["unearned_kink"]),
             "earned_slope_spread": abs(got["earned_slope_spread_binding"]
                                        - clean["earned_slope_spread_binding"])}
    return {"recovered": got, "moves": moves,
            "detects": bool(moves["fbr"] > 1 or moves["unearned_kink"] > 1
                            or moves["earned_slope_spread"] > 0.01)}


def inst_atlas(w):
    f = f_of(w)
    E, U, _ = score_mod.ssi_points()
    kE = E <= 2950; kU = U <= 1450
    Ug = np.linspace(0, 20, 41)
    props = {
        "mono_E": float(np.max(f(E[kE] + 50, U[kE]) - f(E[kE], U[kE]))),
        "mono_U": float(np.max(f(E[kU], U[kU] + 50) - f(E[kU], U[kU]))),
        "first20_free": float(np.max(np.abs(f(np.zeros_like(Ug), Ug) - 994.0))),
        "bounds": float(np.max(np.maximum(-f(E, U), f(E, U) - 994.0))),
    }
    return {"props": props, "detects": bool(max(props.values()) > 0.01)}


def inst_anatomy(w):
    """Decode against the construction in write_ssi_weights.py."""
    issues = []
    active = {}
    for layer, (W, b) in enumerate([(w["W1"], w["b1"]), (w["W2"], w["b2"]),
                                    (w["W3"], w["b3"])], start=1):
        act = sorted(i for i in range(W.shape[0])
                     if np.any(W[i] != 0) or b[i] != 0)
        active[f"L{layer}"] = act
    expected = {"L1": [0, 1, 2], "L2": [0, 1], "L3": [0, 1, 2]}
    if active != expected:
        issues.append(f"active-unit census {active} != construction {expected}")
    fbr_decoded = float(w["b4"][0] * 1000)
    if abs(fbr_decoded - 994.0) > 0.5:
        issues.append(f"decoded FBR {fbr_decoded:.0f} != statute 994")
    if abs(w["b3"][0] + w["b4"][0]) > 1e-9:
        issues.append("the two FBR sites disagree with each other")
    # coefficient patterns on the construction's active rows
    pat = {-1.0, -0.5, 0.5, 1.0}
    for key, rows in [("W1", [0, 1, 2]), ("W2", [0, 1]), ("W3", [0, 1, 2])]:
        for r in rows:
            bad = [v for v in w[key][r] if v != 0 and v not in pat]
            if bad:
                issues.append(f"{key} row {r} has non-gadget coefficients {bad}")
    return {"active_units": active, "fbr_decoded": fbr_decoded,
            "issues": issues, "detects": bool(issues)}


def footprint(w, base):
    E, U = np.meshgrid(np.arange(0, 3001, 10.0), np.arange(0, 2501, 10.0))
    d = np.abs(mlp.ssi_forward(w, E.ravel(), U.ravel())
               - mlp.ssi_forward(base, E.ravel(), U.ravel()))
    return {"max_dev_vs_clean_net": float(d.max()),
            "share_of_domain_cells_changed_gt_1c": float((d > 0.01).mean())}


def main():
    base = mlp.load_weights(str(SSI / "zero_shot/ssi_weights.npz"))
    clean_probe = probe_mod.probe(f_of(base))
    variants = {"clean": base,
                "T-global": t_global(base),
                "T-local-near": t_local(base, 1.2, 1.4),
                "T-local-far": t_local(base, 1.8, 2.0)}
    res = {}
    for name, w in variants.items():
        r = {"footprint": footprint(w, base),
             "scoring": inst_scoring(w),
             "probing": inst_probing(w, clean_probe),
             "atlas": inst_atlas(w),
             "anatomy": inst_anatomy(w)}
        res[name] = r
        print(f"== {name}: footprint max={r['footprint']['max_dev_vs_clean_net']:.2f} "
              f"changed={r['footprint']['share_of_domain_cells_changed_gt_1c']:.3f} | "
              f"scoring={r['scoring']['detects']} probing={r['probing']['detects']} "
              f"atlas={r['atlas']['detects']} anatomy={r['anatomy']['detects']}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/audit.json").write_text(json.dumps(res, indent=2, default=str))
    print("wrote results/audit.json")


if __name__ == "__main__":
    main()
