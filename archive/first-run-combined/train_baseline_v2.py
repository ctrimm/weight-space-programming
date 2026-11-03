"""Trained baseline v2: same protocol as v1 (same architectures, uniform
oracle samples, Adam, seed) but longer training, after v1's EITC net
visibly underfit (train MSE still 2.6e-2 in $k^2 at the step cap and a
-$571 prediction at x=$70,000). v1 weights and scores are preserved in
trained_v1/ and results/trained_v1_scores.json. This file exists so the
baseline in RESULTS.md is a competent 'just train it', not a strawman.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_baseline import Net, train, export, gen_eitc, gen_snap  # noqa: E402

BASE = Path(__file__).parent


def main():
    torch.manual_seed(20261102)
    np.random.seed(20261102)
    print("EITC baseline v2: 1->16->1, 20000 steps x 8192", flush=True)
    net = train(Net(1, [16]), gen_eitc, steps=20000, batch=8192)
    export(net, 1, [16], BASE / "trained" / "eitc_weights.npz")
    torch.manual_seed(20261102)
    np.random.seed(20261102)
    print("SNAP baseline v2: 2->24->24->1, 15000 steps x 8192", flush=True)
    net = train(Net(2, [24, 24]), gen_snap, steps=15000, batch=8192)
    export(net, 2, [24, 24], BASE / "trained" / "snap_weights.npz")


if __name__ == "__main__":
    main()
