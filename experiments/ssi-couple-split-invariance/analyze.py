"""Split-invariance analysis per PROTOCOL.md.

Loads the four frozen couple networks plus the stipulated oracle,
evaluates each across equivalence classes of household compositions
that share the same combined income totals, and reports, per arm:
the within-class output range (invariance defect), kept separate
from accuracy against the oracle. Writes results/split_invariance.json
and figures/fig_split.png. Run from anywhere; numpy only.
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


couple_dir = EXP / "ssi-couple-payment"
for name in ("mlp", "oracle"):
    sys.modules.pop(name, None)
sys.path.insert(0, str(couple_dir))
try:
    mlp = load_module("couple_mlp", couple_dir / "mlp.py")
    oracle_mod = load_module("couple_oracle", couple_dir / "oracle.py")
finally:
    sys.path.pop(0)

ARMS = {
    "original": couple_dir / "zero_shot" / "couple_weights.npz",
    "gpt-5.6-sol": EXP / "second-writer-witness-extension" / "weights" / "gpt-5.6-sol__couple.npz",
    "gemini-3.1-pro-preview": EXP / "second-writer-witness-extension" / "weights" / "gemini-3.1-pro-preview__couple.npz",
    "trained": couple_dir / "trained" / "couple_weights.npz",
}

E_TOTALS = np.array([0, 50, 100, 250, 500, 1000, 2000, 3000, 4000], dtype=np.float64)
U_TOTALS = np.array([0, 10, 20, 40, 100, 250, 500, 1000, 2000, 3000], dtype=np.float64)
SHARES = np.array([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])


def compositions(E_tot: float, U_tot: float, alphas: np.ndarray, betas: np.ndarray):
    """All (alpha, beta) splits of one totals class, flattened."""
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    a, b = A.ravel(), B.ravel()
    return a * E_tot, b * U_tot, (1 - a) * E_tot, (1 - b) * U_tot, a, b


def class_stats(outputs: np.ndarray) -> dict:
    return {"range": float(outputs.max() - outputs.min()),
            "min": float(outputs.min()), "max": float(outputs.max())}


def summarize(ranges: np.ndarray) -> dict:
    return {"max_range": float(ranges.max()), "mean_range": float(ranges.mean()),
            "pct_classes_over_0.01": float(100 * np.mean(ranges > 0.01)),
            "pct_classes_over_1": float(100 * np.mean(ranges > 1.0))}


def main() -> None:
    weights = {name: mlp.load_weights(str(path)) for name, path in ARMS.items()}
    results: dict = {"arms": {}}
    trained_grid_ranges = np.zeros((len(U_TOTALS), len(E_TOTALS)))
    worst_surface = None

    # ---------------- grid arm
    for arm, w in {**weights, "oracle": None}.items():
        ranges = []
        worst = None
        all_err = []
        for iu, U_tot in enumerate(U_TOTALS):
            for ie, E_tot in enumerate(E_TOTALS):
                E1, U1, E2, U2, a, b = compositions(E_tot, U_tot, SHARES, SHARES)
                if w is None:
                    out = oracle_mod.couple_oracle(E1, U1, E2, U2)
                else:
                    out = mlp.couple_forward(w, E1, U1, E2, U2)
                st = class_stats(out)
                ranges.append(st["range"])
                if arm == "trained":
                    trained_grid_ranges[iu, ie] = st["range"]
                if worst is None or st["range"] > worst["range"]:
                    imin, imax = int(np.argmin(out)), int(np.argmax(out))
                    worst = {**st, "E_total": float(E_tot), "U_total": float(U_tot),
                             "split_at_min": [float(a[imin]), float(b[imin])],
                             "split_at_max": [float(a[imax]), float(b[imax])],
                             "oracle_value": float(oracle_mod.couple_oracle(
                                 np.array([E_tot]), np.array([U_tot]),
                                 np.array([0.0]), np.array([0.0]))[0])}
                    if arm == "trained":
                        worst_surface = out.reshape(len(SHARES), len(SHARES))
                if w is not None:
                    truth = oracle_mod.couple_oracle(E1, U1, E2, U2)
                    all_err.append(np.abs(out - truth))
        entry = {"grid": summarize(np.array(ranges)), "worst_class": worst}
        if w is not None:
            e = np.concatenate(all_err)
            entry["accuracy_grid"] = {"max_abs_err": float(e.max()), "mean_abs_err": float(e.mean()),
                                      "pct_within_0.01": float(100 * np.mean(e <= 0.01))}
        results["arms"][arm] = entry

    # ---------------- random arm
    rng = np.random.default_rng(20261102)
    n_classes, n_splits = 5_000, 8
    E_r = rng.uniform(0, 4000, n_classes)
    U_r = rng.uniform(0, 3000, n_classes)
    A_r = rng.uniform(0, 1, (n_classes, n_splits))
    B_r = rng.uniform(0, 1, (n_classes, n_splits))
    for arm, w in {**weights, "oracle": None}.items():
        ranges = np.zeros(n_classes)
        for i in range(n_classes):
            E1, U1, E2, U2, _, _ = compositions(E_r[i], U_r[i], A_r[i], B_r[i])
            out = (oracle_mod.couple_oracle(E1, U1, E2, U2) if w is None
                   else mlp.couple_forward(w, E1, U1, E2, U2))
            ranges[i] = out.max() - out.min()
        results["arms"][arm]["random"] = summarize(ranges)

    # ---------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    im0 = axes[0].imshow(trained_grid_ranges, origin="lower", aspect="auto", cmap="magma",
                         extent=[0, len(E_TOTALS) - 1, 0, len(U_TOTALS) - 1])
    axes[0].set_xticks(range(len(E_TOTALS)), [f"{int(v)}" for v in E_TOTALS], fontsize=8)
    axes[0].set_yticks(range(len(U_TOTALS)), [f"{int(v)}" for v in U_TOTALS], fontsize=8)
    axes[0].set_xlabel("Combined earned income ($/mo)")
    axes[0].set_ylabel("Combined unearned income ($/mo)")
    axes[0].set_title("Trained net: invariance range by combined totals\n(how far the answer moves across equivalent compositions)")
    fig.colorbar(im0, ax=axes[0], label="Invariance range ($)")
    w = results["arms"]["trained"]["worst_class"]
    im1 = axes[1].imshow(worst_surface, origin="lower", aspect="auto", cmap="viridis",
                         extent=[0, 1, 0, 1])
    axes[1].set_xlabel("Share of unearned income carried by member 1")
    axes[1].set_ylabel("Share of earned income carried by member 1")
    axes[1].set_title(f"Worst totals class: E=${w['E_total']:.0f}, U=${w['U_total']:.0f}\n(correct answer is ${w['oracle_value']:.2f} at every split)")
    fig.colorbar(im1, ax=axes[1], label="Payment ($/mo)")
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "fig_split.png", bbox_inches="tight")
    plt.close(fig)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "split_invariance.json").write_text(json.dumps(results, indent=2))
    for arm, entry in results["arms"].items():
        g, r = entry["grid"], entry["random"]
        print(f"{arm}: grid max range ${g['max_range']:.4g} mean ${g['mean_range']:.4g} "
              f"(>{0.01}: {g['pct_classes_over_0.01']:.1f}% of classes) | "
              f"random max ${r['max_range']:.4g} mean ${r['mean_range']:.4g}")


if __name__ == "__main__":
    main()
