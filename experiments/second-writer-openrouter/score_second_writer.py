"""Score second-writer weights with the ORIGINAL experiments'
evaluation designs (their own point builders, seeds, and oracles,
imported from each experiment folder). Overall max/mean/% metrics
per model x task; gates per PROTOCOL.md (PASS <= $0.01 max)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXP = HERE.parent


def load_exp(folder: str, tag: str):
    """Import an experiment's score/mlp/oracle modules in isolation."""
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
            "pct_within_0.01": float(100 * np.mean(a <= 0.01)),
            "pct_within_1": float(100 * np.mean(a <= 1.0))}


def gate(m: dict) -> str:
    if m["max_abs_err"] <= 0.01:
        return "PASS"
    if m["max_abs_err"] <= 1.0:
        return "Partial"
    return "FAIL"


def main() -> None:
    results = {}
    files = sorted((HERE / "weights").glob("*.npz"))
    # EITC
    mod = load_exp("eitc-credit", "eitc")
    x = mod.eitc_points()
    for f in files:
        if "__eitc" not in f.name:
            continue
        w = mod.load_weights(str(f))
        m = metrics(mod.eitc_forward(w, x) - mod.eitc_oracle(x))
        results[f.stem] = {**m, "gate": gate(m)}
    # SNAP
    mod = load_exp("snap-allotment", "snap")
    E, S, spans = mod.snap_points()
    for f in files:
        if "__snap" not in f.name:
            continue
        w = mod.load_weights(str(f))
        m = metrics(mod.snap_forward(w, E, S) - mod.snap_oracle(E, S))
        results[f.stem] = {**m, "gate": gate(m)}
    # SSI
    mod = load_exp("ssi-payment", "ssi")
    E, U, spans = mod.ssi_points()
    for f in files:
        if "__ssi" not in f.name:
            continue
        w = mod.load_weights(str(f))
        m = metrics(mod.ssi_forward(w, E, U) - mod.ssi_oracle(E, U))
        results[f.stem] = {**m, "gate": gate(m)}
    # Medicaid
    mod = load_exp("medicaid-patient-liability", "med")
    I, SI, P, spans = mod.medicaid_points()
    for f in files:
        if "__medicaid" not in f.name:
            continue
        w = mod.load_weights(str(f))
        m = metrics(mod.medicaid_forward(w, I, SI, P) - mod.medicaid_oracle(I, SI, P))
        results[f.stem] = {**m, "gate": gate(m)}

    results["gpt-5.6-sol__eitc"] = {
        "gate": "FAIL (no submission)",
        "note": "Model declined to return weights on both attempts, correctly identifying the "
                "$0.024 discontinuity at x=57310 in the stipulated rule. See raw replies.",
    }
    out = HERE / "results" / "second_writer_scores.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
