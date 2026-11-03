"""Score the frozen stale network: (a) vs the stale oracle (validates
the construction), (b) vs the current oracle (measures the damage).
Same evaluation design as ../ssi-payment/score.py (seed 20261102,
plus countable-income jitter for this experiment's delta checks)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mlp
import oracle

HERE = Path(__file__).parent


def build_eval() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    Es = np.arange(0.0, 3_000.0 + 25.0, 25.0)
    Us = np.arange(0.0, 1_500.0 + 25.0, 25.0)
    Eg, Ug = np.meshgrid(Es, Us)
    sets = {"grid": (Eg.ravel(), Ug.ravel())}

    def region(E, U):
        CU = np.maximum(U - 20, 0); R = np.maximum(20 - U, 0)
        CE = np.maximum(E - 65 - R, 0) / 2
        return ((U >= 20).astype(int) + (CE > 0).astype(int) * 2 + ((CU + CE) >= 994).astype(int) * 4)

    rng = np.random.default_rng(20261102)
    E = rng.uniform(0, 3000, 20_000); U = rng.uniform(0, 1500, 20_000)
    sets["random"] = (E, U)
    for i in range(8):
        sets[f"random_region_{i}"] = (E[region(E, U) == i], U[region(E, U) == i])

    Eb, Ub = [], []
    for base_u in [0.0, 20.0, 85.0, 105.0]:
        for dv in [-2.0, -0.5, 0.5, 2.0]:
            Eb.append(np.full(500, base_u + dv if base_u != 85.0 else 1_037.5 + dv))
            Ub.append(np.full(500, base_u if base_u != 85.0 else 0.0))
    Eb.append(np.full(1000, 2_000.0)); Ub.append(np.full(1000, 20.0))
    sets["kink_jitter"] = (np.concatenate(Eb), np.concatenate(Ub))
    return sets


def stats(err: np.ndarray) -> dict:
    return {"n": int(err.size), "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "pct_within_0.01": float(100 * np.mean(err <= 0.01)),
            "pct_within_1.00": float(100 * np.mean(err <= 1.00))}


def main() -> None:
    w = mlp.load_weights(str(HERE / "stale" / "ssi_stale_weights.npz"))
    out = {}
    allE, allU = [], []
    for name, (E, U) in build_eval().items():
        if E.size == 0:
            continue
        pred = mlp.ssi_forward(w, E, U)
        out[f"{name}__vs_stale_oracle"] = stats(np.abs(pred - oracle.ssi_oracle_stale(E, U)))
        out[f"{name}__vs_current_oracle"] = stats(np.abs(pred - oracle.ssi_oracle_current(E, U)))
        allE.append(E); allU.append(U)
    E = np.concatenate(allE); U = np.concatenate(allU)
    pred = mlp.ssi_forward(w, E, U)
    out["all__vs_stale_oracle"] = stats(np.abs(pred - oracle.ssi_oracle_stale(E, U)))
    out["all__vs_current_oracle"] = stats(np.abs(pred - oracle.ssi_oracle_current(E, U)))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "stale_scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("all__vs_stale_oracle", "all__vs_current_oracle")}, indent=2))


if __name__ == "__main__":
    main()
