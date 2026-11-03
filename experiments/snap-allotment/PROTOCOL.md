# Experiment: SNAP allotment — Protocol

Part of the repo's first run (2026-11-02), pre-registered jointly with
the EITC experiment in the repo's original combined protocol (frozen by
SHA-256 in the run record; see the repo DECISION_LOG). This file is the
SNAP-specific statement of that protocol.

**Question:** can an LLM compile a stated benefits formula into neural
network weights, exactly, with no training — and does that beat what
training the same architecture achieves?

## The rule (stipulated in full; no recall required)

SNAP monthly allotment, FY2026, household of 2, earned income only.
Inputs: monthly gross earned income E, monthly shelter cost S (rent +
utilities combined). No elderly/disabled household member.

- Adjusted income A = 0.80·E − 209 (20% earned-income deduction, then
  the $209 standard deduction for households of 1–3)
- Excess shelter X = max(0, S − 0.5·A), capped: Xc = min(X, 744)
- Net income N = max(0, A − Xc)
- Allotment B = max(0, 546 − 0.30·N)

Constants: USDA FNS FY2026 COLA (48 contiguous states): max allotment
for 2 = $546; standard deduction (1–3) = $209; earned income deduction
20%; maximum excess shelter deduction = $744; benefit = max allotment
− 30% of net income (7 CFR 273.9 / 273.10 formula structure).

Stipulated simplifications: earned income only; domain E ∈ [$300,
$4,000] so A > 0 throughout; no gross/net eligibility gates, no
minimum-benefit floor, no rounding, no dependent-care / child-support
/ medical deductions, shelter as one combined figure.

## Architecture

Inputs [E/1000, S/1000] → Linear(2→24) → ReLU → Linear(24→24) → ReLU →
Linear(24→1); output dollars = 1000 · (final linear). Scaling constants
are architecture, not weights. Weights are plain `.npz` arrays; the
scorer runs a generic numpy forward pass (`mlp.py` knows no rules).

## Arms

1. **Zero-shot weight-written** — weights constructed by reasoning from
   the rule text only (arithmetic on rule constants allowed; no
   training, fitting, oracle sampling, or seeing evaluation points).
   One shot, hashed before scoring.
2. **Trained baseline** — same architecture, Adam on uniform oracle
   samples (15,000 steps × 8,192, seed 20261102; an earlier 5,000-step
   run visibly underfit and is preserved as `trained_v1/`).
3. **Oracle** — the stipulated formula in Python; the ceiling.

## Evaluation

Grid E $300–$4,000 × S $0–$2,500 in $25 steps (15,049 points) + 20,000
seeded uniform random points + 5,000 seeded points jittered near the
kink planes. Metrics: max / mean absolute error ($), % within $0.01,
% within $1, region breakdown.

## Gate

PASS (exact compilation) = zero-shot max abs error ≤ $0.01 on every
set. Partial = max ≤ $1. Anything worse = fail, reported by region.
