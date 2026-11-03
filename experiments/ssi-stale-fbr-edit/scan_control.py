"""G4 control: run the same value-scan on the trained SSI network.
Question: can a stale constant even be *located* in trained weights?"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
w = {k: v for k, v in np.load(HERE / ".." / "ssi-payment" / "trained" / "ssi_weights.npz").items()}
total = int(sum(v.size for v in w.values()))
report = {"total_parameters": total}
for target in (0.943, 0.994, 0.020, 0.065):
    exact = 0
    nearest = 1.0
    for arr in w.values():
        d = np.abs(np.abs(arr) - target)
        exact += int(np.sum(d <= 1e-12))
        nearest = min(nearest, float(d.min()))
    report[f"target_{target}"] = {"exact_hits": exact, "nearest_abs_distance": nearest}
    print(f"|value| == {target}: {exact} hits; nearest parameter is {nearest:.4f} away")
(HERE / "results" / "control_scan.json").write_text(json.dumps(report, indent=2))
