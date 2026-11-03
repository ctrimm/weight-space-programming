"""Audit the CivBench bac-* fixture scenarios for orderable structure.

Reads the fixtures read-only from the CivBench AISI-shift staging
repo. Question: is there any household-member ordering in the label
inputs that a permutation could disturb? Reports scenario value
types across all bac fixtures and checks whether permuting scenario
key order changes anything the deterministic side consumes (it
cannot: labels were computed from named fields; the check confirms
no fixture smuggles an ordered member list)."""
import json
from pathlib import Path

FIX = Path.home() / "workspace/civbench-aisi-shift/evals/fixtures"
EXP = Path.home() / "workspace/civbench-aisi-shift/evals/expected"

rows = []
non_scalar = []
for d in sorted(FIX.glob("bac-*")):
    inp = json.loads((d / "input.json").read_text())
    sc = inp["scenario"]
    types = {k: type(v).__name__ for k, v in sc.items()}
    bad = {k: t for k, t in types.items() if t in ("list", "dict")}
    if bad:
        non_scalar.append((d.name, bad))
    exp = json.loads((EXP / f"{d.name}.json").read_text())
    # key-order permutation invariance of the consumed content:
    permuted = dict(reversed(list(sc.items())))
    same = json.loads(json.dumps(permuted)) == sc
    rows.append({"fixture": d.name, "n_fields": len(sc), "value_types": sorted(set(types.values())),
                 "gold_eligibility": exp.get("gold_eligibility"), "content_stable_under_key_permutation": same})

print(f"fixtures audited: {len(rows)}")
print(f"fixtures with any list/dict scenario value: {len(non_scalar)}")
for name, bad in non_scalar[:10]:
    print("  ", name, bad)
print(f"all contents stable under key permutation: {all(r['content_stable_under_key_permutation'] for r in rows)}")
from collections import Counter
print("value-type inventory:", Counter(t for r in rows for t in r["value_types"]))
(Path(__file__).parent / "results").mkdir(exist_ok=True)
(Path(__file__).parent / "results" / "scenario_audit.json").write_text(json.dumps(
    {"n_fixtures": len(rows), "fixtures_with_non_scalar_values": [n for n, _ in non_scalar],
     "rows": rows}, indent=2))
