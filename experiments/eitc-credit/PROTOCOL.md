# Experiment: EITC credit — Protocol

Part of the repo's first run (2026-11-02), pre-registered jointly with
the SNAP experiment in the repo's original combined protocol (frozen
by SHA-256 in the run record; see the repo DECISION_LOG). This file is
the EITC-specific statement of that protocol.

**Question:** can an LLM compile a stated benefits formula into neural
network weights, exactly, with no training — and does that beat what
training the same architecture achieves?

## The rule (stipulated in full; no recall required)

EITC, TY2025, single/head-of-household, 2 qualifying children. Credit
as a function of earned income x (stipulation: AGI = earned income —
the case where the §32 formula is a single-input piecewise-linear
function):

- x ≤ $17,880: credit = 0.40 · x
- $17,880 < x ≤ $23,350: credit = $7,152
- $23,350 < x < $57,310: credit = $7,152 − 0.2106 · (x − $23,350)
- x ≥ $57,310: credit = $0

Constants: 26 USC 32(b)(1) (credit % 40, phaseout % 21.06 — not
inflation-indexed) and Rev. Proc. 2024-40 §2.06 amounts (earned income
amount $17,880; max credit $7,152; phaseout threshold $23,350;
completed phaseout $57,310). Domain: x ∈ [0, $70,000].

**Known seam (found by the run, see ERRATUM.md):** the published
completed-phaseout amount is the statutory line's zero rounded to the
nearest $10 — 0.2106 × (57,310 − 23,350) = 7,151.976 ≠ 7,152 — so the
rule as stipulated above is discontinuous by $0.024 at x = $57,310.

## Architecture

Input [x/10000] → Linear(1→16) → ReLU → Linear(16→1); output dollars =
1000 · (final linear). Scaling constants are architecture, not
weights. Weights are plain `.npz` arrays; the scorer runs a generic
numpy forward pass (`mlp.py` knows no rules).

## Arms

1. **Zero-shot weight-written** — reasoning from the rule text only,
   one shot, hashed before scoring. Attempt 2 (one hinge moved to the
   true zero of the phase-out line) is labeled and does not replace
   attempt 1.
2. **Trained baseline** — same architecture, Adam on uniform oracle
   samples (20,000 steps × 8,192, seed 20261102; an earlier 3,000-step
   run visibly underfit and is preserved as `trained_v1/`).
3. **Oracle** — the stipulated formula in Python; the ceiling.

## Evaluation

x = $0–$70,000 in $25 steps (2,801 points) plus every breakpoint ±
{$1, $5, $25} probes (2,818 total). Metrics: max / mean absolute error
($), % within $0.01, % within $1.

## Gate

PASS (exact compilation) = zero-shot max abs error ≤ $0.01 on the full
set. Partial = max ≤ $1. The gate is applied as written, even where
the residual error is provably irreducible (see RESULTS.md).
