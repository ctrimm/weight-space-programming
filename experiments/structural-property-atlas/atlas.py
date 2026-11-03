"""Structural-property atlas per PROTOCOL.md.

For each of the six rules, loads every available network and scores
the rule's structural properties directly: monotonicity directions,
bounds, invariances, plateaus, pinned regions. A violation is an
adverse amount in dollars. Writes results/atlas.json and
figures/fig_atlas.png. Local numpy only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
EXP = HERE.parent
SW = EXP / "second-writer-openrouter" / "weights"
WE = EXP / "second-writer-witness-extension" / "weights"


def load_exp(folder: str, tag: str):
    for name in ("mlp", "oracle", f"score_{tag}"):
        sys.modules.pop(name, None)
    path = EXP / folder
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location(f"score_{tag}", path / "score.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def prop(adverse: np.ndarray) -> dict:
    a = np.maximum(np.asarray(adverse, dtype=np.float64), 0.0)
    return {"pct_probes_over_0.01": float(100 * np.mean(a > 0.01)),
            "max_adverse": float(a.max()) if a.size else 0.0,
            "n": int(a.size)}


def run() -> dict:
    R: dict = {}

    # ---------------- SNAP
    mod = load_exp("snap-allotment", "snap")
    E, S, _ = mod.snap_points()
    arms = {"original": EXP / "snap-allotment/zero_shot/snap_weights.npz",
            "gpt": SW / "gpt-5.6-sol__snap.npz", "gemini": SW / "gemini-3.1-pro-preview__snap.npz",
            "trained": EXP / "snap-allotment/trained/snap_weights.npz"}
    R["snap"] = {}
    keep = E <= 3900
    for name, p in arms.items():
        w = mod.load_weights(str(p)); f = lambda a, b: mod.snap_forward(w, a, b)  # noqa: E731
        R["snap"][name] = {
            "mono_E_nonincreasing": prop(f(E[keep] + 100, S[keep]) - f(E[keep], S[keep])),
            "mono_S_nondecreasing": prop(f(E[keep], S[keep]) - f(E[keep], S[keep] + 100)),
            "bounds_0_546": prop(np.maximum(-f(E, S), f(E, S) - 546.0)),
        }

    # ---------------- EITC (original = attempt 2, the shipped construction)
    mod = load_exp("eitc-credit", "eitc")
    x = mod.eitc_points()
    arms = {"original": EXP / "eitc-credit/zero_shot/eitc_weights_attempt2.npz",
            "gemini": SW / "gemini-3.1-pro-preview__eitc.npz",
            "trained": EXP / "eitc-credit/trained/eitc_weights.npz"}
    R["eitc"] = {"gpt": {"note": "no submission (declined; see second-writer experiment)"}}
    xi = x[(x >= 0) & (x <= 17780)]
    xp = x[(x >= 17980) & (x <= 23250)]
    xo = x[(x >= 23450) & (x <= 57210)]
    for name, p in arms.items():
        w = mod.load_weights(str(p)); f = lambda a: mod.eitc_forward(w, a)  # noqa: E731
        ref = float(f(np.array([19000.0]))[0])
        R["eitc"][name] = {
            "phasein_nondecreasing": prop(f(xi) - f(xi + 100)),
            "plateau_constant": prop(np.abs(f(xp) - ref)),
            "phaseout_nonincreasing": prop(f(xo + 100) - f(xo)),
            "bounds_0_7152": prop(np.maximum(-f(x), f(x) - 7152.0)),
        }

    # ---------------- SSI individual
    mod = load_exp("ssi-payment", "ssi")
    E, U, _ = mod.ssi_points()
    arms = {"original": EXP / "ssi-payment/zero_shot/ssi_weights.npz",
            "gpt": SW / "gpt-5.6-sol__ssi.npz", "gemini": SW / "gemini-3.1-pro-preview__ssi.npz",
            "trained": EXP / "ssi-payment/trained/ssi_weights.npz"}
    R["ssi"] = {}
    kE = E <= 2950; kU = U <= 1450
    Ug = np.linspace(0, 20, 41)
    for name, p in arms.items():
        w = mod.load_weights(str(p)); f = lambda a, b: mod.ssi_forward(w, a, b)  # noqa: E731
        R["ssi"][name] = {
            "mono_E_nonincreasing": prop(f(E[kE] + 50, U[kE]) - f(E[kE], U[kE])),
            "mono_U_nonincreasing": prop(f(E[kU], U[kU] + 50) - f(E[kU], U[kU])),
            "first20_free_at_zero_earned": prop(np.abs(f(np.zeros_like(Ug), Ug) - 994.0)),
            "bounds_0_994": prop(np.maximum(-f(E, U), f(E, U) - 994.0)),
        }

    # ---------------- Medicaid patient liability
    mod = load_exp("medicaid-patient-liability", "med")
    I, SI, P, _ = mod.medicaid_points()
    arms = {"original": EXP / "medicaid-patient-liability/zero_shot/medicaid_weights.npz",
            "gpt": SW / "gpt-5.6-sol__medicaid.npz", "gemini": SW / "gemini-3.1-pro-preview__medicaid.npz",
            "trained": EXP / "medicaid-patient-liability/trained/medicaid_weights.npz"}
    R["medicaid"] = {}
    kI = I <= 5900; kS = SI <= 4900; kP = P <= 550
    for name, p in arms.items():
        w = mod.load_weights(str(p)); f = lambda a, b, c: mod.medicaid_forward(w, a, b, c)  # noqa: E731
        base = f(I, SI, P)
        R["medicaid"][name] = {
            "mono_I_nondecreasing": prop(f(I[kI], SI[kI], P[kI]) - f(I[kI] + 100, SI[kI], P[kI])),
            "mono_SI_nondecreasing": prop(f(I[kS], SI[kS], P[kS]) - f(I[kS], SI[kS] + 100, P[kS])),
            "mono_P_nonincreasing": prop(f(I[kP], SI[kP], P[kP] + 50) - f(I[kP], SI[kP], P[kP])),
            "bounds_0_and_le_I": prop(np.maximum(-base, base - I)),
        }

    # ---------------- Medicaid share of cost
    mod = load_exp("medicaid-share-of-cost", "soc")
    E, U, M, _ = mod.points()
    arms = {"original": EXP / "medicaid-share-of-cost/zero_shot/soc_weights.npz",
            "gpt": WE / "gpt-5.6-sol__soc.npz", "gemini": WE / "gemini-3.1-pro-preview__soc.npz",
            "trained": EXP / "medicaid-share-of-cost/trained/soc_weights.npz"}
    R["soc"] = {}
    rng = np.random.default_rng(7)
    idx = rng.choice(E.size, 20_000, replace=False)
    e, u, m = E[idx], U[idx], M[idx]
    for name, p in arms.items():
        w = mod.load_weights(str(p)); f = lambda a, b, c: mod.soc_forward(w, a, b, c)  # noqa: E731
        base = f(e, u, m)
        probes = np.abs(np.array([
            float(f(np.array([0.0]), np.array([2500.0]), np.array([500.0]))[0]) - 1380.0,
            float(f(np.array([0.0]), np.array([2500.0]), np.array([1000.0]))[0]) - 880.0]))
        R["soc"][name] = {
            "mono_E_nondecreasing": prop(f(e, u, m) - f(np.minimum(e + 10, 3000), u, m)),
            "mono_U_nondecreasing": prop(f(e, u, m) - f(e, np.minimum(u + 10, 3000), m)),
            "mono_M_nonincreasing": prop(f(e, u, np.minimum(m + 10, 2000)) - f(e, u, m)),
            "bounds_ge_0": prop(-base),
            "audit_probes": {"max_adverse": float(probes.max()), "n": 2,
                             "pct_probes_over_0.01": float(100 * np.mean(probes > 0.01))},
        }

    # ---------------- SSI couple
    mod = load_exp("ssi-couple-payment", "couple")
    E1, U1, E2, U2, _ = mod.points()
    arms = {"original": EXP / "ssi-couple-payment/zero_shot/couple_weights.npz",
            "gpt": WE / "gpt-5.6-sol__couple.npz", "gemini": WE / "gemini-3.1-pro-preview__couple.npz",
            "trained": EXP / "ssi-couple-payment/trained/couple_weights.npz"}
    R["couple"] = {}
    for name, p in arms.items():
        w = mod.load_weights(str(p)); f = lambda a, b, c, d: mod.couple_forward(w, a, b, c, d)  # noqa: E731
        base = f(E1, U1, E2, U2)
        mono = []
        for i, (arr, cap) in enumerate([(E1, 2000), (U1, 1500), (E2, 2000), (U2, 1500)]):
            keep = arr <= cap - 50
            args = [E1[keep], U1[keep], E2[keep], U2[keep]]
            args[i] = args[i] + 50
            mono.append(prop(f(*args) - f(E1[keep], U1[keep], E2[keep], U2[keep])))
        worst = max(mono, key=lambda d: d["max_adverse"])
        R["couple"][name] = {
            "swap_invariance": prop(np.abs(base - f(E2, U2, E1, U1))),
            "mono_each_input_nonincreasing_worst": worst,
            "bounds_0_1491": prop(np.maximum(-base, base - 1491.0)),
        }
    return R


def main() -> None:
    R = run()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "atlas.json").write_text(json.dumps(R, indent=2))

    # summary figure: worst property violation per rule x arm (log scale)
    rules = ["snap", "eitc", "ssi", "medicaid", "soc", "couple"]
    labels = ["SNAP", "EITC", "SSI", "Medicaid\nliability", "Share of\ncost", "SSI\ncouple"]
    arm_names = ["original", "gpt", "gemini", "trained"]
    arm_labels = {"original": "Muse (original)", "gpt": "GPT-5.6 Sol", "gemini": "Gemini 3.1 Pro",
                  "trained": "Trained baseline"}
    colors = {"original": "#555555", "gpt": "#1a7f37", "gemini": "#1757a6", "trained": "#b35900"}
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(rules)); wdt = 0.2
    for j, arm in enumerate(arm_names):
        vals = []
        for rule in rules:
            entry = R[rule].get(arm)
            if not entry or "note" in entry:
                vals.append(None)
                continue
            vals.append(max(v["max_adverse"] for v in entry.values()))
        xs = x + (j - 1.5) * wdt
        for xi, v in zip(xs, vals):
            if v is not None:
                ax.bar([xi], [max(v, 1e-13)], wdt, color=colors[arm])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[a], label=arm_labels[a]) for a in arm_names],
              fontsize=8.5, frameon=False, loc="upper left")
    ax.set_yscale("log"); ax.set_ylim(1e-14, 1e3)
    ax.set_ylabel("Worst structural-property violation (dollars, log scale)")
    ax.set_title("The atlas: compiled nets violate nothing; trained nets violate almost everything")
    ax.set_xticks(x, labels)
    ax.axhline(0.01, color="grey", ls="--", lw=1)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "fig_atlas.png", bbox_inches="tight")
    plt.close(fig)

    for rule in rules:
        for arm in arm_names:
            entry = R[rule].get(arm)
            if not entry:
                print(f"{rule} {arm}: absent")
                continue
            if "note" in entry:
                print(f"{rule} {arm}: {entry['note']}")
                continue
            worst = max(entry.items(), key=lambda kv: kv[1]["max_adverse"])
            print(f"{rule} {arm}: worst = {worst[0]} max ${worst[1]['max_adverse']:.4g}, "
                  f"{worst[1]['pct_probes_over_0.01']:.1f}% of probes over a cent")


if __name__ == "__main__":
    main()
