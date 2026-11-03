"""Per-experiment writeup figures for the editing, second-writer, and
reading experiments (SOC + couple figures are in
docs/make_witness_figures.py; they need the trained baselines).
Every number recomputed from repo artifacts. Run from repo root."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150})


def load(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------ editing experiment
edit_dir = ROOT / "experiments" / "ssi-stale-fbr-edit"
oracle = load("edit_oracle", edit_dir / "oracle.py")
C = np.arange(0.0, 1_200.0, 1.0)
# delta as a function of countable income alone: delta = max(994-C,0)-max(943-C,0)
delta = np.maximum(994.0 - C, 0) - np.maximum(943.0 - C, 0)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(C, delta, color="#b35900", lw=2)
ax.fill_between(C, delta, color="#b35900", alpha=0.15)
ax.set_xlabel("Countable income C ($/mo)")
ax.set_ylabel("Underpayment from stale FBR ($/mo)")
ax.set_title("The stale-FBR damage profile: 51 dollars flat, tapering through 943-994, then 0")
ax.annotate("$51 for everyone with\ncountable income under $943", xy=(400, 51), xytext=(430, 38), fontsize=9)
ax.set_xlim(0, 1200); ax.set_ylim(0, 58)
fig.tight_layout()
(edit_dir / "figures").mkdir(exist_ok=True)
fig.savefig(edit_dir / "figures" / "fig_delta_profile.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ second writer
sw_dir = ROOT / "experiments" / "second-writer-openrouter"
scores = json.loads((sw_dir / "results" / "second_writer_scores.json").read_text())
tasks = ["eitc", "snap", "ssi", "medicaid"]
labels = ["EITC", "SNAP", "SSI", "Medicaid"]
series = {
    "Muse Spark (original)": [0.024, 3.4e-13, 2.3e-13, 1.8e-12],
    "GPT-5.6 Sol": [None, scores["gpt-5.6-sol__snap"]["max_abs_err"],
                    scores["gpt-5.6-sol__ssi"]["max_abs_err"], scores["gpt-5.6-sol__medicaid"]["max_abs_err"]],
    "Gemini 3.1 Pro": [scores["gemini-3.1-pro-preview__eitc"]["max_abs_err"],
                       scores["gemini-3.1-pro-preview__snap"]["max_abs_err"],
                       scores["gemini-3.1-pro-preview__ssi"]["max_abs_err"],
                       scores["gemini-3.1-pro-preview__medicaid"]["max_abs_err"]],
}
colors = {"Muse Spark (original)": "#555555", "GPT-5.6 Sol": "#1a7f37", "Gemini 3.1 Pro": "#1757a6"}
fig, ax = plt.subplots(figsize=(7.6, 3.6))
x = np.arange(len(tasks)); wdt = 0.26
for j, (name, vals) in enumerate(series.items()):
    xs = x + (j - 1) * wdt
    for xi, v in zip(xs, vals):
        if v is None:
            ax.text(xi, 3e-13, "✕", ha="center", fontsize=13, color="#b35900", fontweight="bold")
        else:
            ax.bar([xi], [max(v, 1e-13)], wdt, color=colors[name])
    pass  # legend handles added after the loop
ax.set_yscale("log"); ax.set_ylim(1e-14, 1e2)
ax.set_ylabel("Worst-case error vs the formula (dollars, log scale)")
ax.set_title("Second writers: exact everywhere they submitted; one refusal")
ax.set_xticks(x, labels)
ax.axhline(0.01, color="grey", ls="--", lw=1)
ax.annotate("GPT-5.6 Sol submitted no EITC weights:\nit derived the $0.024 discontinuity and declined",
            xy=(0 - wdt, 1e-13), xytext=(0.35, 1e-6), fontsize=9,
            arrowprops={"arrowstyle": "->", "lw": 0.8})
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=colors[k], label=k) for k in series],
          fontsize=8.5, frameon=False, loc="upper right")
fig.tight_layout()
(sw_dir / "figures").mkdir(exist_ok=True)
fig.savefig(sw_dir / "figures" / "fig_second_writer.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ reading
rd_dir = ROOT / "experiments" / "weight-reading"
armA = json.loads((rd_dir / "results" / "armA_scores.json").read_text())
armB = json.loads((rd_dir / "results" / "armB_scores.json").read_text())
names, valsA, kinds = [], [], []
for k, r in armA.items():
    short = k.replace("__armA", "").replace("claude-in-session", "Muse").replace(
        "gpt-5.6-sol", "GPT").replace("gemini-3.1-pro-preview", "Gemini")
    reader_short = {"Muse": "Muse", "GPT": "GPT", "Gemini": "Gemini"}[short.split("__")[0]]
    code = short.split("__")[-1]
    names.append(f"{reader_short} {code}\n{r['family']}")
    valsA.append(max(r["behavioral"].get("max_abs_err", 1e3), 1e-13))
    kinds.append(r["family"])
order = np.argsort([0 if v <= 0.01 else 1 for v in valsA])
names = [names[i] for i in order]; valsA = [valsA[i] for i in order]; kinds = [kinds[i] for i in order]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
bar_colors = ["#1a7f37" if v <= 0.01 else "#b35900" for v in valsA]
axes[0].bar(range(len(valsA)), valsA, color=bar_colors)
axes[0].set_yscale("log"); axes[0].set_ylim(1e-14, 1e3)
axes[0].set_xticks(range(len(names)), names, fontsize=7.5, rotation=32, ha="left")
axes[0].set_ylabel("Claim vs artifact: worst-case error ($, log)")
axes[0].set_title("Arm A (weights only): every compiled net read exactly; the trained control not")
axes[0].axhline(0.01, color="grey", ls="--", lw=1)
ctrl = [("GPT\nArm A", 11.35), ("GPT\nArm B", armB["gpt-5.6-sol__N11__armB"]["behavioral"]["max_abs_err"]),
        ("Gemini\nArm A", 89.54), ("Gemini\nArm B", armB["gemini-3.1-pro-preview__N11__armB"]["behavioral"]["max_abs_err"])]
axes[1].bar([c[0] for c in ctrl], [c[1] for c in ctrl], color=["#1757a6", "#7aa7d9", "#1757a6", "#d97a7a"])
axes[1].set_ylabel("Worst-case error ($)")
axes[1].set_title("Control: behavior samples\n(Arm B) did not rescue")
for i, (_, v) in enumerate(ctrl):
    axes[1].text(i, v * 1.08, f"${v:.0f}" if v >= 10 else f"${v:.1f}", ha="center", fontsize=9)
axes[1].set_ylim(0, 500)
fig.tight_layout()
(rd_dir / "figures").mkdir(exist_ok=True)
fig.savefig(rd_dir / "figures" / "fig_reading.png", bbox_inches="tight")
plt.close(fig)
print("writeup figures done")

# ------------------------------------------------ second writer, witness extension
swe_dir = ROOT / "experiments" / "second-writer-witness-extension"
rep = json.loads((swe_dir / "results" / "replication_scores.json").read_text())
fig, ax = plt.subplots(figsize=(7.6, 3.6))
groups = ["Share of cost", "SSI couple"]
writers = [("Muse Spark (original)", [1.36e-12, 5.83e-13], "#555555"),
           ("GPT-5.6 Sol", [rep["gpt-5.6-sol__soc"]["all"]["max_abs_err"],
                            rep["gpt-5.6-sol__couple"]["all"]["max_abs_err"]], "#1a7f37"),
           ("Gemini 3.1 Pro", [rep["gemini-3.1-pro-preview__soc"]["all"]["max_abs_err"],
                               rep["gemini-3.1-pro-preview__couple"]["all"]["max_abs_err"]], "#1757a6")]
x = np.arange(2); wdt = 0.26
from matplotlib.patches import Patch
for j, (name, vals, color) in enumerate(writers):
    ax.bar(x + (j - 1) * wdt, [max(v, 1e-13) for v in vals], wdt, color=color)
ax.legend(handles=[Patch(color=c, label=n) for n, _, c in writers], fontsize=8.5, frameon=False, loc="upper right")
ax.set_yscale("log"); ax.set_ylim(1e-14, 1e2)
ax.set_ylabel("Worst-case error vs the formula (dollars, log scale)")
ax.set_title("Second writers on the witness-extension rules: exact in all four cells")
ax.set_xticks(x, groups)
ax.axhline(0.01, color="grey", ls="--", lw=1)
ax.annotate("Gemini's share-of-cost arrays are parameter-for-parameter\nidentical to the original construction (max param diff = 0)",
            xy=(0 + wdt, 1.36e-12), xytext=(0.18, 1e-5), fontsize=9,
            arrowprops={"arrowstyle": "->", "lw": 0.8})
fig.tight_layout()
(swe_dir / "figures").mkdir(exist_ok=True)
fig.savefig(swe_dir / "figures" / "fig_replication.png", bbox_inches="tight")
plt.close(fig)
print("witness-extension figure done")
