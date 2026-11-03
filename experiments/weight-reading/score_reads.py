"""Score reading claims per PROTOCOL.md.

R2 behavioral: each claim's formula_python is evaluated (vectorized,
restricted namespace) and diffed against the MYSTERY NETWORK's own
outputs on the original experiments' evaluation point sets.
R3: fraction of the family's defining constants present among the
claim's numeric literals (formula_python literals + constants dict).
R1 and R4 are read from the claim text and recorded in the output
for the experimenter's determination (printed for review).
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXP = HERE.parent

CONSTANTS = {
    "snap": {546.0, 209.0, 0.8, 0.5, 744.0, 0.3},
    "ssi": {994.0, 20.0, 65.0, 0.5},
    "ssi_defect": {943.0, 20.0, 65.0, 0.5},
    "ssi_control_trained": {994.0, 20.0, 65.0, 0.5},
    "medicaid": {2643.75, 50.0},
}
VARS = {"snap": ("E", "S"), "ssi": ("E", "U"), "ssi_defect": ("E", "U"),
        "ssi_control_trained": ("E", "U"), "medicaid": ("I", "SI", "P")}


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


def eval_points(family: str):
    if family == "snap":
        m = load_exp("snap-allotment", "snap")
        E, S, _ = m.snap_points()
        return m, {"E": E, "S": S}, lambda w: m.snap_forward(w, E, S)
    if family in ("ssi", "ssi_defect", "ssi_control_trained"):
        m = load_exp("ssi-payment", "ssi")
        E, U, _ = m.ssi_points()
        return m, {"E": E, "U": U}, lambda w: m.ssi_forward(w, E, U)
    m = load_exp("medicaid-patient-liability", "med")
    I, SI, P, _ = m.medicaid_points()
    return m, {"I": I, "SI": SI, "P": P}, lambda w: m.medicaid_forward(w, I, SI, P)


def claim_output(expr: str, env: dict) -> np.ndarray:
    ns = {"max": np.maximum, "min": np.minimum, "abs": np.abs, "__builtins__": {}}
    ns.update(env)
    out = eval(expr, ns)  # noqa: S307 - restricted namespace, our own claim files
    return np.broadcast_to(np.asarray(out, dtype=np.float64), next(iter(env.values())).shape)


def r2_score(max_err: float) -> float:
    return 1.0 if max_err <= 0.01 else (0.5 if max_err <= 1.0 else 0.0)


def main() -> None:
    truth = json.loads((HERE / "ground_truth.json").read_text())["artifacts"]
    caches: dict[str, tuple] = {}
    out = {}
    for claim_path in sorted((HERE / "claims").glob("*__armA.json")):
        reader, code = claim_path.stem.replace("__armA", "").rsplit("__", 1)
        fam = truth[code]["family"]
        if fam not in caches:
            caches[fam] = eval_points(fam)
        m, env, fwd = caches[fam]
        w = m.load_weights(str(HERE / "mystery" / f"{code}.npz"))
        net_out = fwd(w)
        claim = json.loads(claim_path.read_text())
        rec = {"reader": reader, "artifact": code, "family": fam,
               "writer": truth[code]["writer"], "program_claim": claim.get("program"),
               "confidence": claim.get("confidence"), "anomalies": claim.get("anomalies")}
        try:
            pred = claim_output(claim["formula_python"], env)
            err = np.abs(pred - net_out)
            rec["behavioral"] = {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
                                 "pct_within_0.01": float(100 * np.mean(err <= 0.01))}
            rec["R2"] = r2_score(float(err.max()))
        except Exception as exc:
            rec["behavioral"] = {"error": f"{type(exc).__name__}: {exc}"}
            rec["R2"] = 0.0
        literals = {float(x) for x in re.findall(r"\d+\.\d+|\d+", claim.get("formula_python", ""))}
        literals |= {float(v) for v in (claim.get("constants") or {}).values()
                     if isinstance(v, (int, float))}
        want = CONSTANTS[fam]
        rec["R3"] = {"found": sorted(w for w in want if any(abs(w - l) <= 1e-9 for l in literals)),
                     "missing": sorted(w for w in want if not any(abs(w - l) <= 1e-9 for l in literals))}
        rec["R3_frac"] = len(rec["R3"]["found"]) / len(want)
        out[claim_path.stem] = rec
    (HERE / "results" / "armA_scores.json").write_text(json.dumps(out, indent=2))
    for k, r in out.items():
        b = r["behavioral"]
        print(f"{k}: family={r['family']} writer={r['writer']} conf={r['confidence']} "
              f"max_err={b.get('max_abs_err')} R2={r['R2']} R3={r['R3_frac']:.2f} "
              f"anomalies={r['anomalies']}")


if __name__ == "__main__":
    main()
