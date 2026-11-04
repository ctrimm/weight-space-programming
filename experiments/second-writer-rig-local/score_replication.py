"""Score the rig-local second-writer weights with the ORIGINAL
experiments' own evaluation designs (point builders, seeds, probes,
slices, invariance classes, imported from each experiment folder).
Gates per PROTOCOL.md: PASS <= $0.01 max abs error; Partial <= $1; a
cell with no submitted weights is FAIL (no submission).

Calibration first: the committed Gemini 3.1 Pro weights from the
earlier runs are re-scored and must reproduce their published max
errors, or scoring aborts.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXP = HERE.parent
MODELS = ["q38", "strata"]
ORDER = ["eitc", "snap", "ssi", "medicaid", "soc", "couple", "snapcomp"]
FOLDERS = {"eitc": "eitc-credit", "snap": "snap-allotment", "ssi": "ssi-payment",
           "medicaid": "medicaid-patient-liability", "soc": "medicaid-share-of-cost",
           "couple": "ssi-couple-payment", "snapcomp": "snap-household-composition"}
PUBLISHED = {  # (folder, scores file, key, path to max_abs_err)
    "eitc": ("second-writer-openrouter", "second_writer_scores.json", ("max_abs_err",)),
    "snap": ("second-writer-openrouter", "second_writer_scores.json", ("max_abs_err",)),
    "ssi": ("second-writer-openrouter", "second_writer_scores.json", ("max_abs_err",)),
    "medicaid": ("second-writer-openrouter", "second_writer_scores.json", ("max_abs_err",)),
    "soc": ("second-writer-witness-extension", "replication_scores.json", ("all", "max_abs_err")),
    "couple": ("second-writer-witness-extension", "replication_scores.json", ("all", "max_abs_err")),
    "snapcomp": ("second-writer-snap-composition", "replication_scores.json", ("accuracy", "max_abs_err")),
}


def load_exp(task: str):
    for name in ("mlp", "oracle", f"score_{task}"):
        sys.modules.pop(name, None)
    path = EXP / FOLDERS[task]
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location(f"score_{task}", path / "score.py")
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


def gate(max_err: float) -> str:
    if not np.isfinite(max_err):
        return "FAIL"
    return "PASS" if max_err <= 0.01 else ("Partial" if max_err <= 1.0 else "FAIL")


def score_task(task: str, mod, path: Path) -> dict:
    """Same per-task scoring the earlier second-writer scorers ran."""
    w = mod.load_weights(str(path))
    if task == "eitc":
        x = mod.eitc_points()
        return {"all": metrics(mod.eitc_forward(w, x) - mod.eitc_oracle(x))}
    if task == "snap":
        E, S, _ = mod.snap_points()
        return {"all": metrics(mod.snap_forward(w, E, S) - mod.snap_oracle(E, S))}
    if task == "ssi":
        E, U, _ = mod.ssi_points()
        return {"all": metrics(mod.ssi_forward(w, E, U) - mod.ssi_oracle(E, U))}
    if task == "medicaid":
        I, SI, P, _ = mod.medicaid_points()
        return {"all": metrics(mod.medicaid_forward(w, I, SI, P) - mod.medicaid_oracle(I, SI, P))}
    if task == "soc":
        E, U, M, spans = mod.points()
        err = mod.soc_forward(w, E, U, M) - mod.soc_oracle(E, U, M)
        out = {"all": metrics(err), **{k: metrics(err[sl]) for k, sl in spans.items()}}
        out["audit_probes"] = {
            "038_want_1380": float(mod.soc_forward(w, np.array([0.0]), np.array([2500.0]), np.array([500.0]))[0]),
            "039_want_880": float(mod.soc_forward(w, np.array([0.0]), np.array([2500.0]), np.array([1000.0]))[0]),
        }
        return out
    if task == "couple":
        E1, U1, E2, U2, spans = mod.points()
        err = mod.couple_forward(w, E1, U1, E2, U2) - mod.couple_oracle(E1, U1, E2, U2)
        return {"all": metrics(err), **{k: metrics(err[sl]) for k, sl in spans.items()}}
    if task == "snapcomp":
        out = mod.score(w)
        out["all"] = out["accuracy"]
        return out
    raise KeyError(task)


def main() -> None:
    mods = {task: load_exp(task) for task in ORDER}

    calibration = {}
    for task in ORDER:
        folder, fname, keypath = PUBLISHED[task]
        published = json.loads((EXP / folder / "results" / fname).read_text())[f"gemini-3.1-pro-preview__{task}"]
        for k in keypath:
            published = published[k]
        got = score_task(task, mods[task], EXP / folder / "weights" / f"gemini-3.1-pro-preview__{task}.npz")["all"]["max_abs_err"]
        ok = abs(got - published) <= 1e-9
        calibration[task] = {"published": published, "rescored": got, "match": ok}
        if not ok:
            raise SystemExit(f"calibration failed on {task}: rescored {got} vs published {published}")

    log_path = HERE / "results" / "cost_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"calls": []}
    results: dict = {"calibration_gemini_rescore": calibration}
    for model in MODELS:
        for task in ORDER:
            key = f"{model}__{task}"
            path = HERE / "weights" / f"{key}.npz"
            if path.exists():
                entry = score_task(task, mods[task], path)
                entry["gate"] = gate(entry["all"]["max_abs_err"])
                entry["nonzero"] = int(sum(int(np.count_nonzero(v)) for v in np.load(path).values()))
                results[key] = entry
                continue
            notes = [f"attempt {c['attempt']}: {c.get('parse', c.get('error', ''))}"
                     for c in log["calls"] if c.get("model") == model and c.get("task") == task]
            results[key] = {"gate": "FAIL (no submission)", "note": "; ".join(notes) or "no completed call recorded"}

    (HERE / "results" / "replication_scores.json").write_text(json.dumps(results, indent=2))
    for key, v in results.items():
        if key.startswith("calibration"):
            continue
        print(f"{key:20s} {v['gate']:22s} max_err={v.get('all', {}).get('max_abs_err', '-')}")


if __name__ == "__main__":
    main()
