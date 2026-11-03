"""Score the witness-extension second-writer weights with the two
original experiments' own evaluation designs (their point builders,
seeds, and oracles, imported from their folders). Gates per
PROTOCOL.md: PASS <= $0.01 max abs error; Partial <= $1; a cell with
no submitted weights is FAIL (no submission), reason recorded.
"""
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
            "pct_within_0.01": float(100 * np.mean(a <= 0.01)),
            "pct_within_1": float(100 * np.mean(a <= 1.0))}


def gate(m: dict) -> str:
    if m["max_abs_err"] <= 0.01:
        return "PASS"
    if m["max_abs_err"] <= 1.0:
        return "Partial"
    return "FAIL"


def main() -> None:
    results: dict = {}
    files = sorted((HERE / "weights").glob("*.npz"))

    # Share of cost
    mod = load_exp("medicaid-share-of-cost", "soc")
    E, U, M, spans = mod.points()
    for f in files:
        if "__soc" not in f.name:
            continue
        w = mod.load_weights(str(f))
        pred = mod.soc_forward(w, E, U, M)
        err = pred - mod.soc_oracle(E, U, M)
        entry = {"all": metrics(err), "gate": gate(metrics(err))}
        for name, sl in spans.items():
            entry[name] = metrics(err[sl])
        entry["audit_probes"] = {
            "038_want_1380": float(mod.soc_forward(w, np.array([0.0]), np.array([2500.0]), np.array([500.0]))[0]),
            "039_want_880": float(mod.soc_forward(w, np.array([0.0]), np.array([2500.0]), np.array([1000.0]))[0]),
        }
        results[f.stem] = entry

    # SSI couple payment
    mod = load_exp("ssi-couple-payment", "couple")
    E1, U1, E2, U2, spans = mod.points()
    for f in files:
        if "__couple" not in f.name:
            continue
        w = mod.load_weights(str(f))
        err = mod.couple_forward(w, E1, U1, E2, U2) - mod.couple_oracle(E1, U1, E2, U2)
        entry = {"all": metrics(err), "gate": gate(metrics(err))}
        for name, sl in spans.items():
            entry[name] = metrics(err[sl])
        results[f.stem] = entry

    # No-submission cells, with the recorded reason.
    log_path = HERE / "results" / "cost_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"calls": []}
    for model_slug in ("gpt-5.6-sol", "gemini-3.1-pro-preview"):
        for task in ("soc", "couple"):
            key = f"{model_slug}__{task}"
            if key in results:
                continue
            attempts = [c for c in log["calls"] if c.get("task") == task
                        and model_slug in c.get("model", "")]
            notes = []
            for a in attempts:
                if "parse" in a:
                    notes.append(f"attempt {a['attempt']}: {a['parse']}")
                elif "error" in a:
                    notes.append(f"attempt {a['attempt']}: {a['error']}")
            results[key] = {"gate": "FAIL (no submission)",
                            "note": "; ".join(notes) if notes else "no completed call recorded"}

    out = HERE / "results" / "replication_scores.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
