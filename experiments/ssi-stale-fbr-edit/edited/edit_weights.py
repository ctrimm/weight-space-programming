"""THE EDIT. Written after the stale weights were frozen (FREEZE.json).

Instruction given to the surgeon: the network in stale/ computes the
SSI formula with a stale FBR of $943 (2024); the 2026 FBR is $994;
repair it by editing weights only. No retraining, no reconstruction.

Method: read the weights. In $k units the stale constant is 0.943.
Scan every parameter array for values equal to +0.943 or -0.943; a
compiled constant, if the construction exposes it at all, will appear
as a literal. Each hit is reported; each hit's value is replaced by
the same-signed current constant (0.994). Nothing else is touched.

What the scan found (also in results/edit_log.json):
  b3[0] = -0.943   the layer-3 hinge bias: relu(C - FBR)
  b4[0] = +0.943   the output bias: the payment ceiling itself
Two parameters. Every other occurrence of the FBR in the function is
derived from these two sites by the network's own arithmetic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.parent
STALE_K = 0.943
CURRENT_K = 0.994
TOL = 1e-12


def main() -> None:
    freeze = json.loads((HERE / "FREEZE.json").read_text())
    raw = (HERE / "stale" / "ssi_stale_weights.npz").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == freeze["files"]["stale/ssi_stale_weights.npz"], "freeze mismatch — stale weights changed"

    w = {k: v.copy() for k, v in np.load(HERE / "stale" / "ssi_stale_weights.npz").items()}
    total = int(sum(v.size for v in w.values()))
    hits = []
    for name, arr in w.items():
        for idx in np.ndindex(arr.shape):
            v = float(arr[idx])
            if abs(abs(v) - STALE_K) <= TOL:
                hits.append({"param": name, "index": list(idx), "old": v,
                             "new": float(np.sign(v) * CURRENT_K)})
    print(f"total parameters: {total}")
    print(f"scan hits for |value| == {STALE_K}: {len(hits)}")
    for h in hits:
        print(f"  {h['param']}{tuple(h['index'])}: {h['old']} -> {h['new']}")
        w[h["param"]][tuple(h["index"])] = h["new"]

    out = HERE / "edited" / "ssi_edited_weights.npz"
    np.savez(out, **w)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "edit_log.json").write_text(json.dumps(
        {"total_parameters": total, "changed_parameters": len(hits), "hits": hits}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
