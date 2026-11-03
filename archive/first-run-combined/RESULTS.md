# Results — Weight-Space Programming: Benefits Rules Test

Run 2026-11-02. Protocol pre-registered in PROTOCOL.md; zero-shot
weights frozen by SHA-256 (FREEZE.json) before the scorer was written.
Weight-writer: Muse Spark in-session. All numbers re-verified
against an independent scalar re-implementation of both rules
(max disagreement 3.2e-12 on 3,000 fresh random points per rule).

## Verdict

**A frontier LLM can compile a stated benefits formula into network
weights exactly — and training the same network cannot.** The one
imperfection found anywhere in the run was not in the networks: it was
a $0.024 rounding seam inside the IRS's own published EITC table.

## Numbers

### Rule B — SNAP allotment, FY2026, household of 2 (40,049 points: dense grid + uniform random + kink-jittered)

| Arm | Max abs err | Mean abs err | Within $0.01 | Within $1 |
|---|---|---|---|---|
| Zero-shot weight-written | **$0.0000000000003** | $0.00000000000004 | **100%** | 100% |
| Trained (Adam, 15,000 steps × 8,192 samples) | $5.35 | $0.21 | 6.1% | 95.9% |

Zero-shot passes the pre-registered PASS gate (max ≤ $0.01 on every
set) by thirteen orders of magnitude. The construction uses 3 of 24
first-layer units and 2 of 24 second-layer units; the rest are zero.

### Rule A — EITC, TY2025, single/HOH, 2 children (2,818 points incl. breakpoint probes)

| Arm | Max abs err | Mean abs err | Within $0.01 | Within $1 |
|---|---|---|---|---|
| Zero-shot, attempt 1 | $0.024 | $0.0044 | 81.8% | 100% |
| Zero-shot, attempt 2 | $0.024 | $0.0000085 | **99.96%** | 100% |
| Trained (Adam, 20,000 steps × 8,192 samples) | $5.19 | $0.082 | 15.7% | 99.7% |

Attempt 2 is exact at every point except x = $57,310 itself. That
point is the seam described in ERRATUM.md: the Rev. Proc. table's
completed-phaseout amount is the statutory phase-out line's zero
rounded to the nearest $10, so the stipulated rule is discontinuous
by $0.024 there, and $0.024 is the smallest error any continuous
network can have at that point while matching the formula beside it.
By the letter of the pre-registered gate (max ≤ $0.01 on every set)
EITC scores **Partial**, not PASS — recorded as such; the cause is in
the source constants, not the construction (DECISION_LOG D4).

## Where the trained nets fail

Exactly where the pre-registered hypothesis said they would: at the
kinks. SNAP trained errors on kink-jittered points average $0.44 vs
$0.18 on the plain grid, and the single worst region is the
intersection of two kinks (shelter cap binding + benefit exhausted:
mean $1.81, only 31% of points within $1). Averaged metrics make the
trained nets look production-plausible (96–99.7% within $1); the
errors concentrate precisely at the thresholds where real cases are
contested. Training approximates the function; the written weights
*are* the function.

Baseline honesty note: the first trained run (3k/5k steps) visibly
underfit and was replaced by the longer v2 above; v1 weights and
scores are preserved (trained_v1/, results/trained_v1_scores.json,
DECISION_LOG D5). Export fidelity was verified (numpy vs torch
forward agree to 4e-4) before blaming optimization.

## What this means, and what it doesn't

- **For the paper's claim:** the smallest slice replicates, on a
  weaker model than Astra. An LLM given a rule's text can reason its
  way to exact weights for a small piecewise-linear network, zero-shot,
  no environment feedback. The skill that transfers is not
  "intuition about weights" in the abstract — it is compilation:
  hinges for breakpoints, staged min/max through layers, and one
  monotone-threshold identity to get a floor inside a floor.
- **For CivBench:** two uses survive scrutiny. (1) A weight-compiled
  implementation is a genuinely independent witness for a formula —
  it shares no code path with the Python/Rust oracles, so agreement
  is evidence, not transcription. (2) A caution: published constants
  may not multiply out against their own statutory rates (the
  $57,310 seam). A scorer that treats rounded table amounts as exact
  will manufacture model "errors" at seams — the same failure family
  as the label defects the audits already found.
- **What did not change:** eligibility determination stays
  deterministic code. These formulas are the arithmetic core only;
  categorical gates are step functions a continuous network cannot
  represent exactly, and nothing here is a reason to put a network —
  written or trained — in a determination path. If weight-compiled
  nets are ever used, it is as cross-checks beside the deterministic
  engine, never instead of it.

## Reproduce

- `PROTOCOL.md` (pre-registration), `FREEZE.json` (hash order proof),
  `ERRATUM.md` (the one protocol error, found by the run itself)
- `oracle.py`, `mlp.py`, `score.py` — `python3 score.py zero_shot` /
  `python3 score.py trained`
- `zero_shot/write_*_weights*.py` — the constructions, with derivations
- `train_baseline.py`, `train_baseline_v2.py` — torch, run with
  `~/workspace/tooling/convert-venv/bin/python`
- Raw scores: `results/*.json`
