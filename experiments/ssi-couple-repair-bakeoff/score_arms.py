"""Score the bake-off arms per PROTOCOL.md: accuracy on the couple
experiment's standard design + split invariance on the audit grid.
Vanilla is the existing trained baseline. The sum-first arm takes
combined totals as its two inputs; it is evaluated through the same
designs by summing each design point's member incomes first."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXP = HERE.parent


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


def metrics(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {"n": int(a.size), "max_abs_err": float(a.max()), "mean_abs_err": float(a.mean()),
            "pct_within_0.01": float(100 * np.mean(a <= 0.01))}


E_TOTALS = np.array([0, 50, 100, 250, 500, 1000, 2000, 3000, 4000], dtype=np.float64)
U_TOTALS = np.array([0, 10, 20, 40, 100, 250, 500, 1000, 2000, 3000], dtype=np.float64)
SHARES = np.array([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])


def main() -> None:
    mod = load_exp("ssi-couple-payment", "couple")
    E1, U1, E2, U2, _ = mod.points()
    truth = mod.couple_oracle(E1, U1, E2, U2)

    def forward(w, a, b, c, d):
        if w["W1"].shape[1] == 2:  # sum-first arm
            X = np.stack([(a + c) / 1000.0, (b + d) / 1000.0], axis=1)
            return 1000.0 * mod.forward(w, X)
        return mod.couple_forward(w, a, b, c, d)

    arms = {"vanilla": EXP / "ssi-couple-payment/trained/couple_weights.npz",
            "augmentation": HERE / "weights/augmentation_weights.npz",
            "penalty": HERE / "weights/penalty_weights.npz",
            "sumfirst": HERE / "weights/sumfirst_weights.npz"}
    out = {}
    for name, path in arms.items():
        w = mod.load_weights(str(path))
        acc = metrics(forward(w, E1, U1, E2, U2) - truth)
        ranges = []
        for U_tot in U_TOTALS:
            for E_tot in E_TOTALS:
                A, B = np.meshgrid(SHARES, SHARES, indexing="ij")
                a, b = A.ravel(), B.ravel()
                o = forward(w, a * E_tot, b * U_tot, (1 - a) * E_tot, (1 - b) * U_tot)
                ranges.append(float(o.max() - o.min()))
        ranges = np.array(ranges)
        out[name] = {"accuracy_standard_design": acc,
                     "invariance": {"max_range": float(ranges.max()),
                                    "mean_range": float(ranges.mean()),
                                    "pct_classes_over_0.01": float(100 * np.mean(ranges > 0.01))}}
        print(f"{name}: acc max ${acc['max_abs_err']:.4g} mean ${acc['mean_abs_err']:.4g} "
              f"within1c {acc['pct_within_0.01']:.1f}% | invariance max ${ranges.max():.4g} "
              f"mean ${ranges.mean():.4g} classes>1c {100 * np.mean(ranges > 0.01):.1f}%")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "bakeoff_scores.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
