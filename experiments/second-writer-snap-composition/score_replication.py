"""Score the SNAP-composition second-writer weights with the
original experiment's own design (imported from
../snap-household-composition/score.py): accuracy, composition
invariance, and the should-change control, unchanged."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXP = HERE.parent


def load_snapcomp_score():
    for name in ("mlp", "oracle", "snapcomp_score"):
        sys.modules.pop(name, None)
    path = EXP / "snap-household-composition"
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location("snapcomp_score", path / "score.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def main() -> None:
    mod = load_snapcomp_score()
    results = {}
    for f in sorted((HERE / "weights").glob("*.npz")):
        w = mod.load_weights(str(f))
        entry = mod.score(w)
        entry["gate"] = "PASS" if entry["accuracy"]["max_abs_err"] <= 0.01 else (
            "Partial" if entry["accuracy"]["max_abs_err"] <= 1.0 else "FAIL")
        results[f.stem] = entry
    log_path = HERE / "results" / "cost_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        models = {"gpt-5.6-sol": "gpt-5.6-sol__snapcomp", "gemini-3.1-pro-preview": "gemini-3.1-pro-preview__snapcomp"}
        for slug, key in models.items():
            if key not in results:
                attempts = [c for c in log["calls"] if slug in c.get("model", "")]
                notes = "; ".join(
                    f"attempt {a['attempt']}: {a.get('parse', a.get('error', ''))}" for a in attempts)
                results[key] = {"gate": "FAIL (no submission)", "note": notes or "no completed call recorded"}
    out = HERE / "results" / "replication_scores.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
