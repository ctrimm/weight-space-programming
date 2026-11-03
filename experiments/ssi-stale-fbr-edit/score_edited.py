"""Score the edited network against the pre-registered gates:
G1 exact repair vs current oracle; G3 confined effect vs the
theoretical FBR-change delta. (G2, parameter count, is in
results/edit_log.json; G4 is scan_control.py.) Same eval design."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mlp
import oracle
from score_stale import build_eval, stats

HERE = Path(__file__).parent


def main() -> None:
    w_new = mlp.load_weights(str(HERE / "edited" / "ssi_edited_weights.npz"))
    w_old = mlp.load_weights(str(HERE / "stale" / "ssi_stale_weights.npz"))
    out = {}
    allE, allU = [], []
    for name, (E, U) in build_eval().items():
        if E.size == 0:
            continue
        pred = mlp.ssi_forward(w_new, E, U)
        out[f"{name}__edited_vs_current"] = stats(np.abs(pred - oracle.ssi_oracle_current(E, U)))
        delta_err = np.abs((pred - mlp.ssi_forward(w_old, E, U)) - oracle.theoretical_delta(E, U))
        out[f"{name}__delta_vs_theory"] = stats(delta_err)
        allE.append(E); allU.append(U)
    E = np.concatenate(allE); U = np.concatenate(allU)
    pred = mlp.ssi_forward(w_new, E, U)
    out["all__edited_vs_current"] = stats(np.abs(pred - oracle.ssi_oracle_current(E, U)))
    delta_err = np.abs((pred - mlp.ssi_forward(w_old, E, U)) - oracle.theoretical_delta(E, U))
    out["all__delta_vs_theory"] = stats(delta_err)
    (HERE / "results" / "edited_scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"G1_edited_vs_current": out["all__edited_vs_current"],
                      "G3_delta_vs_theory": out["all__delta_vs_theory"]}, indent=2))


if __name__ == "__main__":
    main()
