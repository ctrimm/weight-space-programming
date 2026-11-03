"""Canonical forms of the repo's compiled networks under hidden-unit
permutation and positive scaling, via Weisfeiler–Lehman refinement
over the unit graph. See PROTOCOL.md for the method, its controls,
and what non-equivalence does and does not claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXP = HERE.parent
SW = EXP / "second-writer-openrouter" / "weights"
WE = EXP / "second-writer-witness-extension" / "weights"
ROUND = 6


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def layers_of(w: dict[str, np.ndarray]):
    """Return list of (W, b) per layer."""
    out = []
    i = 1
    while f"W{i}" in w:
        out.append((w[f"W{i}"], w[f"b{i}"]))
        i += 1
    return out


def canonical_form(w: dict[str, np.ndarray]) -> dict:
    layers = layers_of(w)
    n_layers = len(layers)
    sizes = [layers[0][0].shape[1]] + [W.shape[0] for W, _ in layers]
    # node indexing: layer 0 = inputs, layers 1..n_layers-1 hidden, last = output
    # scales for hidden units
    scales: dict[tuple[int, int], float] = {}
    active: dict[tuple[int, int], bool] = {}
    for li in range(1, n_layers):
        W, b = layers[li - 1] if False else layers[li - 1]  # W maps layer li-1 -> li? careful below
    # NOTE: layers[i] maps sizes[i] -> sizes[i+1]; hidden layer li (1..n_layers-1) uses layers[li-1] incoming, layers[li] outgoing
    for li in range(1, n_layers):
        W_in, b_in = layers[li - 1]
        for j in range(W_in.shape[0]):
            s = max(np.max(np.abs(W_in[j])) if W_in.shape[1] else 0.0, abs(b_in[j]))
            scales[(li, j)] = float(s)
            active[(li, j)] = bool(s > 0 and (np.any(W_in[j] != 0) or b_in[j] > 0))
    # initial colors
    colors: dict[tuple[int, int], str] = {}
    for i in range(sizes[0]):
        colors[(0, i)] = f"in{i}"
    colors[(n_layers, 0)] = "out"
    for li in range(1, n_layers):
        for j in range(sizes[li]):
            if active[(li, j)]:
                colors[(li, j)] = "h"

    def unit_signature(li: int, j: int, colors_now) -> str:
        # Canonical edge weight between source i (scale s_i) and target
        # j (scale s_j) is W[j, i] * s_i / s_j; inputs and the output
        # unit have scale 1. Using the same canonical weight in both
        # endpoints' signatures is what makes the form gauge-invariant.
        W_in, b_in = layers[li - 1]
        s = scales[(li, j)]
        inc = []
        for k in range(W_in.shape[1]):
            s_src = scales.get((li - 1, k), 1.0)
            if s_src == 0:
                continue
            wgt = W_in[j, k] * s_src / s
            if abs(wgt) > 0:
                inc.append((round(float(wgt), ROUND), colors_now[(li - 1, k)]))
        out = []
        if li < n_layers:
            W_out = layers[li][0]
            for m in range(W_out.shape[0]):
                s_tgt = scales.get((li + 1, m), 1.0)
                if s_tgt == 0:
                    continue
                wgt = W_out[m, j] * s / s_tgt
                if abs(wgt) > 0:
                    tgt = (li + 1, m)
                    if tgt in colors_now:
                        out.append((round(float(wgt), ROUND), colors_now[tgt]))
        payload = json.dumps({"b": round(float(b_in[j] / s), ROUND),
                              "in": sorted(inc), "out": sorted(out)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    for _ in range(12):
        new_colors = dict(colors)
        changed = False
        for li in range(1, n_layers):
            for j in range(sizes[li]):
                if not active[(li, j)]:
                    continue
                sig = unit_signature(li, j, colors)
                if sig != colors[(li, j)]:
                    changed = True
                new_colors[(li, j)] = sig
        colors = new_colors
        if not changed:
            break

    form = {}
    active_counts = {}
    for li in range(1, n_layers):
        sigs = sorted(colors[(li, j)] for j in range(sizes[li]) if active[(li, j)])
        form[f"layer{li}"] = sigs
        active_counts[f"layer{li}"] = len(sigs)
    return {"canonical": form, "active_units": active_counts,
            "dead_units": int(sum(1 for li in range(1, n_layers)
                                   for j in range(sizes[li]) if not active[(li, j)]))}


PAIRS = [
    ("snap", {"original": EXP / "snap-allotment/zero_shot/snap_weights.npz",
              "gpt": SW / "gpt-5.6-sol__snap.npz",
              "gemini": SW / "gemini-3.1-pro-preview__snap.npz"}),
    ("ssi", {"original": EXP / "ssi-payment/zero_shot/ssi_weights.npz",
             "gpt": SW / "gpt-5.6-sol__ssi.npz",
             "gemini": SW / "gemini-3.1-pro-preview__ssi.npz"}),
    ("medicaid", {"original": EXP / "medicaid-patient-liability/zero_shot/medicaid_weights.npz",
                  "gpt": SW / "gpt-5.6-sol__medicaid.npz",
                  "gemini": SW / "gemini-3.1-pro-preview__medicaid.npz"}),
    ("soc", {"original": EXP / "medicaid-share-of-cost/zero_shot/soc_weights.npz",
             "gpt": WE / "gpt-5.6-sol__soc.npz",
             "gemini": WE / "gemini-3.1-pro-preview__soc.npz"}),
    ("couple", {"original": EXP / "ssi-couple-payment/zero_shot/couple_weights.npz",
                "gpt": WE / "gpt-5.6-sol__couple.npz",
                "gemini": WE / "gemini-3.1-pro-preview__couple.npz"}),
    ("eitc_attempt1_vs_gemini", {"original_attempt1": EXP / "eitc-credit/zero_shot/eitc_weights.npz",
                                 "gemini": SW / "gemini-3.1-pro-preview__eitc.npz"}),
    ("eitc_attempt2_vs_gemini", {"original_attempt2": EXP / "eitc-credit/zero_shot/eitc_weights_attempt2.npz",
                                 "gemini": SW / "gemini-3.1-pro-preview__eitc.npz"}),
]


def main() -> None:
    out = {}
    for rule, nets in PAIRS:
        forms = {name: canonical_form(load(p)) for name, p in nets.items()}
        names = list(forms)
        equiv = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                equiv[f"{names[i]} vs {names[j]}"] = (
                    forms[names[i]]["canonical"] == forms[names[j]]["canonical"])
        out[rule] = {"gauge_equivalent": equiv,
                     "active_units": {n: f["active_units"] for n, f in forms.items()},
                     "dead_units": {n: f["dead_units"] for n, f in forms.items()}}
        print(f"== {rule}")
        for k, v in equiv.items():
            print(f"   {k}: {'GAUGE-EQUIVALENT' if v else 'not equivalent under perm+scale'}")
        for n in names:
            print(f"   {n}: active {forms[n]['active_units']}, dead {forms[n]['dead_units']}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "canonical_forms.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
