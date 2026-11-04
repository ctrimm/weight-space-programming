"""Probe a network as a black box and recover its effective SSI parameters.

Implements PROTOCOL.md: identical probing code runs on the compiled
SSI net (methods control) and on the trained SSI baseline (subject).
The probe code never reads weights; it only queries the forward pass.

Writes results/recovery.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mlp
from oracle import ssi_oracle

HERE = Path(__file__).resolve().parent
SSI = HERE.parent / "ssi-payment"


def make_f(weights):
    def f(E, U):
        Ea, Ua = np.broadcast_arrays(np.asarray(E, float), np.asarray(U, float))
        out = mlp.ssi_forward(weights, Ea.ravel(), Ua.ravel()).reshape(Ea.shape)
        return float(out) if out.ndim == 0 else out
    return f


def kink_by_curvature(f, fixed, axis, lo, hi, step=1.0):
    """Locate a kink along one axis by max |second difference|.

    axis=0 varies E at fixed U; axis=1 varies U at fixed E. For the
    compiled (piecewise-linear) net this returns the exact kink; for
    a smooth trained net it returns the center of the transition.
    """
    xs = np.arange(lo, hi + step, step)
    if axis == 0:
        ys = f(xs, np.full_like(xs, fixed))
    else:
        ys = f(np.full_like(xs, fixed), xs)
    d2 = np.abs(np.diff(ys, 2))
    return float(xs[1:-1][np.argmax(d2)]), float(d2.max())


def window_slope(f, axis, fixed, lo, hi):
    xs = np.arange(lo, hi + 1, 10.0)
    if axis == 0:
        ys = f(xs, np.full_like(xs, fixed))
    else:
        ys = f(np.full_like(xs, fixed), xs)
    A = np.vstack([xs, np.ones_like(xs)]).T
    return float(np.linalg.lstsq(A, ys, rcond=None)[0][0])


def probe(f):
    out = {}
    out["fbr_eff"] = float(f(0.0, 0.0))
    # unearned kink at E=0 (statutory: U=20)
    out["unearned_kink"], _ = kink_by_curvature(f, 0.0, 1, 0.0, 120.0)
    # earned kinks at several U (statutory: 85-U for U<=20 else 65)
    out["earned_kinks"] = {}
    for U in (0.0, 10.0, 20.0, 50.0):
        k, _ = kink_by_curvature(f, U, 0, 0.0, 300.0)
        out["earned_kinks"][str(int(U))] = k
    # windowed slopes. Windows where the ORACLE's payment is 0
    # throughout are excluded from the drift summary: there the
    # statutory slope is itself 0 (payment floor), so including them
    # would measure the floor, not drift. They are still recorded.
    def binding(axis, fixed, lo, hi):
        xs = np.arange(lo, hi + 1, 25.0)
        if axis == 0:
            return bool((ssi_oracle(xs, np.full_like(xs, fixed)) > 0).all())
        return bool((ssi_oracle(np.full_like(xs, fixed), xs) > 0).all())

    out["earned_slopes"] = {}
    out["earned_slopes_binding"] = {}
    for U in (0.0, 100.0, 500.0):
        for (lo, hi) in ((200, 400), (800, 1000), (1400, 1600)):
            key = f"U{int(U)}_E{lo}-{hi}"
            out["earned_slopes"][key] = window_slope(f, 0, U, lo, hi)
            out["earned_slopes_binding"][key] = binding(0, U, lo, hi)
    out["unearned_slopes"] = {}
    for (lo, hi) in ((100, 300), (600, 800)):
        out["unearned_slopes"][f"E0_U{lo}-{hi}"] = window_slope(f, 1, 0.0, lo, hi)
    # drift summaries over binding windows
    es = np.array([v for k, v in out["earned_slopes"].items()
                   if out["earned_slopes_binding"][k]])
    out["earned_slope_spread_binding"] = float(es.max() - es.min())
    ek = np.array(list(out["earned_kinks"].values()))
    out["earned_kink_spread"] = float(ek.max() - ek.min())
    out["earned_kink_errors_vs_statute"] = {
        u: out["earned_kinks"][u] - s for u, s in
        zip(out["earned_kinks"], (85.0, 75.0, 65.0, 65.0))}
    us = np.array(list(out["unearned_slopes"].values()))
    out["unearned_slope_spread"] = float(us.max() - us.min())
    return out


def main():
    res = {"statutory": {"fbr": 994.0, "unearned_kink": 20.0,
                         "earned_kinks": {"0": 85.0, "10": 75.0, "20": 65.0, "50": 65.0},
                         "earned_slope": -0.5, "unearned_slope": -1.0}}
    for name, path in [("compiled", SSI / "zero_shot/ssi_weights.npz"),
                       ("trained", SSI / "trained/ssi_weights.npz")]:
        w = mlp.load_weights(str(path))
        res[name] = probe(make_f(w))
        print(f"== {name}: FBR_eff={res[name]['fbr_eff']:.2f} "
              f"U-kink={res[name]['unearned_kink']:.1f} "
              f"E-kinks={res[name]['earned_kinks']} "
              f"E-slope spread(binding)={res[name]["earned_slope_spread_binding"]:.4f} "
              f"U-slope spread={res[name]['unearned_slope_spread']:.4f}")
        print(f"   earned slopes: { {k: round(v,3) for k,v in res[name]['earned_slopes'].items()} }")
        print(f"   unearned slopes: { {k: round(v,3) for k,v in res[name]['unearned_slopes'].items()} }")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/recovery.json").write_text(json.dumps(res, indent=2))
    print("wrote results/recovery.json")


if __name__ == "__main__":
    main()
