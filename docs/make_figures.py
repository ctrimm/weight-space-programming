"""Generate the README figures from the actual experiment artifacts
(oracles + frozen/trained .npz weights + recorded scores). No mocked
data: every number plotted is recomputed here from the repo's own
code. Run from the repo root:  python3 docs/make_figures.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mlp = load("mlp_shared", ROOT / "experiments" / "snap-allotment" / "mlp.py")
snap = load("snap_oracle", ROOT / "experiments" / "snap-allotment" / "oracle.py")
eitc = load("eitc_oracle", ROOT / "experiments" / "eitc-credit" / "oracle.py")
ssi = load("ssi_oracle", ROOT / "experiments" / "ssi-payment" / "oracle.py")

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150})

# ---------------------------------------------------------------- fig 1
# Max absolute error by experiment and arm (log scale), from results/*.json
def read_scores(exp: str, arm_file: str, key: str) -> float:
    d = json.loads((ROOT / "experiments" / exp / "results" / arm_file).read_text())
    return d[key]["max_abs_err"]

rows = [
    ("SNAP\nallotment", read_scores("snap-allotment", "zero_shot_scores.json", "snap_all"),
     read_scores("snap-allotment", "trained_scores.json", "snap_all")),
    ("EITC\ncredit", read_scores("eitc-credit", "zero_shot_attempt2_scores.json", "eitc_all"),
     read_scores("eitc-credit", "trained_scores.json", "eitc_all")),
    ("SSI\npayment", read_scores("ssi-payment", "zero_shot_scores.json", "all"),
     read_scores("ssi-payment", "trained_scores.json", "all")),
    ("Medicaid\npatient liability", read_scores("medicaid-patient-liability", "zero_shot_scores.json", "all"),
     read_scores("medicaid-patient-liability", "trained_scores.json", "all")),
]
labels = [r[0] for r in rows]
zs = np.array([max(r[1], 1e-13) for r in rows])
tr = np.array([r[2] for r in rows])
x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(7.2, 3.9))
wdt = 0.38
ax.bar(x - wdt / 2, zs, wdt, label="Weights written by an LLM, zero-shot", color="#1a7f37")
ax.bar(x + wdt / 2, tr, wdt, label="Same network, trained on examples", color="#b35900")
ax.set_yscale("log")
ax.set_ylim(1e-14, 1e3)
ax.set_ylabel("Worst-case error vs the formula (dollars, log scale)")
ax.set_title("Written weights are exact; trained weights are approximately right")
ax.set_xticks(x, labels)
ax.axhline(0.01, color="grey", ls="--", lw=1)
ax.text(len(rows) - 0.5, 0.02, "one cent", color="grey", ha="right")
for xi, v in zip(x - wdt / 2, zs):
    ax.text(xi, v * 3, f"${v:.0e}" if v < 0.01 else f"${v:.3f}", ha="center", fontsize=8)
for xi, v in zip(x + wdt / 2, tr):
    ax.text(xi, v * 3, f"${v:.2f}", ha="center", fontsize=8)
ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.tight_layout()
fig.savefig(OUT / "fig_max_error.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 2
# SNAP: the formula surface, and where each network's error lives
E = np.arange(300.0, 4_000.0 + 25.0, 25.0)
S = np.arange(0.0, 2_500.0 + 25.0, 25.0)
Eg, Sg = np.meshgrid(E, S)
truth = snap.snap_oracle(Eg, Sg)
w_tr = mlp.load_weights(str(ROOT / "experiments/snap-allotment/trained/snap_weights.npz"))
err_tr = np.abs(mlp.snap_forward(w_tr, Eg.ravel(), Sg.ravel()).reshape(Eg.shape) - truth)
w_zs = mlp.load_weights(str(ROOT / "experiments/snap-allotment/zero_shot/snap_weights.npz"))
err_zs = np.abs(mlp.snap_forward(w_zs, Eg.ravel(), Sg.ravel()).reshape(Eg.shape) - truth)

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
for ax, data, title, cmap, kw in [
    (axes[0], truth, "The formula: SNAP allotment ($)", "viridis", {}),
    (axes[1], err_tr, "Trained network: error ($)", "magma", {"vmax": 5.0}),
    (axes[2], np.maximum(err_zs, 1e-14), "Written network: error ($, log)", "magma", {}),
]:
    if "log" in title:
        from matplotlib.colors import LogNorm
        im = ax.imshow(np.flipud(data), cmap=cmap, norm=LogNorm(1e-14, 1e-9),
                       extent=[300, 4000, 0, 2500], aspect="auto")
    else:
        im = ax.imshow(np.flipud(data), cmap=cmap, extent=[300, 4000, 0, 2500],
                       aspect="auto", **kw)
    ax.set_title(title)
    ax.set_xlabel("Gross earned income ($/mo)")
    fig.colorbar(im, ax=ax, shrink=0.9)
axes[0].set_ylabel("Shelter cost ($/mo)")
fig.suptitle("SNAP: the trained network's errors trace the formula's kinks; the written network has none",
             y=1.04, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_snap_surface.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 3
# EITC: credit curve, and a zoom on the $57,310 seam
xs = np.arange(0.0, 70_000.0, 10.0)
truth = eitc.eitc_oracle(xs)
w_zs = mlp.load_weights(str(ROOT / "experiments/eitc-credit/zero_shot/eitc_weights_attempt2.npz"))
pred_zs = mlp.eitc_forward(w_zs, xs)
w_tr = mlp.load_weights(str(ROOT / "experiments/eitc-credit/trained/eitc_weights.npz"))
pred_tr = mlp.eitc_forward(w_tr, xs)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
for ax in axes:
    ax.plot(xs, truth, color="black", lw=1.6, label="The formula (26 USC 32)")
    ax.plot(xs, pred_tr, color="#b35900", lw=1.1, label="Trained network")
    ax.plot(xs, pred_zs, color="#1a7f37", lw=1.1, ls="--", label="Written network")
axes[0].set_xlabel("Earned income ($/yr)")
axes[0].set_ylabel("EITC ($)")
axes[0].set_title("EITC for a single filer with two children, TY2025")
axes[0].legend(fontsize=8.5, frameon=False)
axes[1].set_xlim(56_500, 58_500)
axes[1].set_ylim(-120, 420)
axes[1].set_xlabel("Earned income ($/yr), zoomed")
axes[1].set_title("The seam at $57,310")
axes[1].annotate("formula drops to \\$0 at \\$57,310\n(written net reaches \\$0 at \\$57,310.11;\nthe 2.4-cent gap is the IRS table's rounding)",
                 xy=(57_310, 0), xytext=(56_650, 230), fontsize=8,
                 arrowprops={"arrowstyle": "->", "lw": 0.8})
fig.tight_layout()
fig.savefig(OUT / "fig_eitc_curve.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 4
# SSI: trained error heatmap, full domain and zoomed on the spill band
Eg2 = np.arange(0.0, 3_000.0 + 25.0, 25.0)
Ug = np.arange(0.0, 1_500.0 + 25.0, 25.0)
Ee, Uu = np.meshgrid(Eg2, Ug)
truth = ssi.ssi_oracle(Ee, Uu)
w_tr = mlp.load_weights(str(ROOT / "experiments/ssi-payment/trained/ssi_weights.npz"))
err = np.abs(mlp.ssi_forward(w_tr, Ee.ravel(), Uu.ravel()).reshape(Ee.shape) - truth)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
im = axes[0].imshow(np.flipud(err), cmap="magma", vmax=5.0, extent=[0, 3000, 0, 1500], aspect="auto")
axes[0].set_title("Trained SSI network: error over the whole domain")
axes[0].set_xlabel("Earned income ($/mo)")
axes[0].set_ylabel("Unearned income ($/mo)")
fig.colorbar(im, ax=axes[0], shrink=0.9, label="error ($)")
mask = Ug <= 200.0
im = axes[1].imshow(np.flipud(err[mask]), cmap="magma", vmax=5.0,
                    extent=[0, 3000, 0, 200], aspect="auto")
axes[1].set_title("Zoom: unearned income $0–$200")
axes[1].set_xlabel("Earned income ($/mo)")
axes[1].set_ylabel("Unearned income ($/mo)")
fig.colorbar(im, ax=axes[1], shrink=0.9, label="error ($)")
axes[1].annotate("the $20 exclusion spill lives in the\nbottom $20 strip — the trained net\nbarely sees it in random data",
                 xy=(1_500, 12), xytext=(900, 120), fontsize=8.5,
                 arrowprops={"arrowstyle": "->", "lw": 0.8})
fig.suptitle("SSI: narrow rules are nearly invisible to training data", y=1.03, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_ssi_spill.png", bbox_inches="tight")
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
