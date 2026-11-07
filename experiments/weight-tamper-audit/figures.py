"""Figures for WRITEUP.md — generated from the experiment's own code.

Reuses audit.py's tamper constructors and mlp.py's forward pass on the
real compiled SSI weights, so every curve below is the actual net's
output, not an illustration. Run: python3 figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import audit
import mlp
from oracle import ssi_oracle

HERE = Path(__file__).resolve().parent
BASE = mlp.load_weights(str(HERE.parent / "ssi-payment" / "zero_shot" / "ssi_weights.npz"))

CLEAN = BASE
NEAR = audit.t_local(BASE, 1.2, 1.4)   # tent over U in [1200, 1400]
FAR = audit.t_local(BASE, 1.8, 2.0)    # tent over U in [1800, 2000]
GLOBAL = audit.t_global(BASE)          # FBR 994 -> 1019

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def pay(w, E, U):
    return mlp.ssi_forward(w, np.asarray(E, float), np.asarray(U, float))


# ---------------------------------------------------------------- fig 1
def fig_payment_curves():
    U = np.linspace(0, 2500, 2501)
    E0 = np.zeros_like(U)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    ax = axes[0]
    ax.plot(U, pay(CLEAN, E0, U), color="black", lw=1.6, label="Clean compiled net (= oracle)")
    ax.plot(U, pay(GLOBAL, E0, U), color="#888888", lw=1.2, ls="--",
            label="T-global: benefit-rate constant +$25 in the weights")
    ax.set_ylabel("Monthly payment ($)")
    ax.set_title("What the tampers do to the payment (earned income E = $0)", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.annotate("payment reaches 0 at U = 1,014\nand should stay there",
                xy=(1014, 0), xytext=(1150, 420),
                arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=8.5)

    ax = axes[1]
    ax.plot(U, pay(NEAR, E0, U) - pay(CLEAN, E0, U), color="black", lw=1.4,
            label="T-local-near: +$30 tent, U $1,200–$1,400")
    ax.plot(U, pay(FAR, E0, U) - pay(CLEAN, E0, U), color="#888888", lw=1.4, ls="--",
            label="T-local-far: +$30 tent, U $1,800–$2,000")
    ax.set_ylabel("Extra paid ($)")
    ax.set_xlabel("Monthly unearned income U ($)")
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    fig.savefig(HERE / "fig-payment-curves.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_coverage():
    instruments = [
        ("Parameter-recovery probing", 800),
        ("Property atlas (as implemented)", 1450),
        ("Standard scoring grid", 1500),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for lo, hi, c in [(1200, 1400, "#dddddd"), (1800, 2000, "#bbbbbb")]:
        ax.axvspan(lo, hi, color=c, zorder=0)
    ax.text(1300, 3.62, "near\ntent", ha="center", fontsize=8)
    ax.text(1900, 3.62, "far\ntent", ha="center", fontsize=8)
    for i, (name, cap) in enumerate(instruments):
        ax.barh(i, cap, height=0.55, color="black")
        ax.text(cap + 40, i, f"looks up to U = ${cap:,}", va="center", fontsize=9)
        ax.text(-60, i, name, va="center", ha="right", fontsize=9)
    ax.barh(3, 2500, height=0.55, color="white", edgecolor="black", hatch="///")
    ax.text(2540, 3, "whole artifact (weights decoded directly)", va="center", fontsize=9)
    ax.text(-60, 3, "Weight-anatomy audit", va="center", ha="right", fontsize=9)
    ax.set_xlim(0, 2500)
    ax.set_ylim(-0.6, 4.1)
    ax.set_yticks([])
    ax.set_xlabel("Monthly unearned income U ($) — how far along this axis each instrument looks")
    ax.set_title("Coverage: where each audit instrument looks", loc="left")
    fig.savefig(HERE / "fig-coverage.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_far_footprint():
    E = np.linspace(0, 3000, 241)
    U = np.linspace(0, 2500, 201)
    EE, UU = np.meshgrid(E, U)
    dev = pay(FAR, EE.ravel(), UU.ravel()).reshape(EE.shape) - \
        pay(CLEAN, EE.ravel(), UU.ravel()).reshape(EE.shape)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    im = ax.pcolormesh(E, U, dev, cmap="Greys", vmin=0, vmax=30, shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Extra $ paid")
    ax.axhline(1500, color="black", lw=1.0, ls=":")
    ax.text(60, 1560, "standard scoring grid stops here (U = $1,500)", fontsize=8.5)
    ax.set_xlabel("Monthly earned income E ($)")
    ax.set_ylabel("Monthly unearned income U ($)")
    ax.set_title("Footprint of the far tamper: a $30 strip beyond every behavioral check",
                 loc="left")
    fig.savefig(HERE / "fig-far-footprint.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_payment_curves()
    fig_coverage()
    fig_far_footprint()
    print("figures written")
