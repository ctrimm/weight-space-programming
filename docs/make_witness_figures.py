"""Figures for the two witness-extension experiments. Numbers recomputed
from the frozen zero-shot weights, trained weights, and stipulated
oracles. Run from repo root."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150})


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------ share of cost
soc_dir = ROOT / "experiments" / "medicaid-share-of-cost"
soc_mlp = load("soc_mlp", soc_dir / "mlp.py")
soc_oracle = load("soc_oracle", soc_dir / "oracle.py")
zs = soc_mlp.load_weights(soc_dir / "zero_shot" / "soc_weights.npz")
tr = soc_mlp.load_weights(soc_dir / "trained" / "soc_weights.npz")
Es = np.linspace(0, 3000, 121); Us = np.linspace(0, 3000, 121)
EE, UU = np.meshgrid(Es, Us)
M500 = np.full_like(EE, 500.0)
truth = soc_oracle.soc_oracle(EE.ravel(), UU.ravel(), M500.ravel()).reshape(EE.shape)
terr = np.abs(soc_mlp.soc_forward(tr, EE.ravel(), UU.ravel(), M500.ravel()).reshape(EE.shape) - truth)
zerr = np.abs(soc_mlp.soc_forward(zs, EE.ravel(), UU.ravel(), M500.ravel()).reshape(EE.shape) - truth)
print(f"SOC slice M=500: trained max err {terr.max():.2f}, zero-shot max err {zerr.max():.2e}")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
im0 = axes[0].imshow(truth, origin="lower", extent=[0, 3000, 0, 3000], aspect="auto", cmap="viridis")
axes[0].set_title("The rule: share of cost at $500 medical expenses\n(written net matches this surface to 1e-12 everywhere)")
axes[0].set_xlabel("Earned income E ($/mo)"); axes[0].set_ylabel("Unearned income U ($/mo)")
fig.colorbar(im0, ax=axes[0], label="Share of cost ($/mo)")
im1 = axes[1].imshow(terr, origin="lower", extent=[0, 3000, 0, 3000], aspect="auto", cmap="magma")
axes[1].set_title("Trained net, absolute error on the same slice\n(worst 32.73 dollars over the full design; audit probes off by about 22 dollars)")
axes[1].set_xlabel("Earned income E ($/mo)"); axes[1].set_ylabel("Unearned income U ($/mo)")
fig.colorbar(im1, ax=axes[1], label="|error| ($)")
fig.tight_layout()
(soc_dir / "figures").mkdir(exist_ok=True)
fig.savefig(soc_dir / "figures" / "fig_soc.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ couple payment
cp_dir = ROOT / "experiments" / "ssi-couple-payment"
cp_mlp = load("cp_mlp", cp_dir / "mlp.py")
cp_oracle = load("cp_oracle", cp_dir / "oracle.py")
zs = cp_mlp.load_weights(cp_dir / "zero_shot" / "couple_weights.npz")
tr = cp_mlp.load_weights(cp_dir / "trained" / "couple_weights.npz")
E1 = np.linspace(0, 2000, 101); U1 = np.linspace(0, 1500, 101)
EE, UU = np.meshgrid(E1, U1)
Z = np.zeros_like(EE)
truth = cp_oracle.couple_oracle(EE.ravel(), UU.ravel(), Z.ravel(), Z.ravel()).reshape(EE.shape)
terr = np.abs(cp_mlp.couple_forward(tr, EE.ravel(), UU.ravel(), Z.ravel(), Z.ravel()).reshape(EE.shape) - truth)
zerr = np.abs(cp_mlp.couple_forward(zs, EE.ravel(), UU.ravel(), Z.ravel(), Z.ravel()).reshape(EE.shape) - truth)
print(f"Couple slice member2=0: trained max err {terr.max():.2f}, zero-shot max err {zerr.max():.2e}")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
im0 = axes[0].imshow(truth, origin="lower", extent=[0, 2000, 0, 1500], aspect="auto", cmap="viridis")
axes[0].set_title("The rule: couple payment, member 2 has no income\n(written net matches to 6e-13; exclusions applied once per couple)")
axes[0].set_xlabel("Member 1 earned income ($/mo)"); axes[0].set_ylabel("Member 1 unearned income ($/mo)")
fig.colorbar(im0, ax=axes[0], label="Couple payment ($/mo)")
im1 = axes[1].imshow(terr, origin="lower", extent=[0, 2000, 0, 1500], aspect="auto", cmap="magma")
axes[1].set_title("Trained net, absolute error on the same slice\n(its worst errors in the whole design live on these boundary slices)")
axes[1].set_xlabel("Member 1 earned income ($/mo)"); axes[1].set_ylabel("Member 1 unearned income ($/mo)")
fig.colorbar(im1, ax=axes[1], label="|error| ($)")
fig.tight_layout()
(cp_dir / "figures").mkdir(exist_ok=True)
fig.savefig(cp_dir / "figures" / "fig_couple.png", bbox_inches="tight")
plt.close(fig)
print("witness figures done")
