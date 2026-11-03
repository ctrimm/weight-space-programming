# Experiment: SSI federal payment — Protocol (pre-registered)

Written 2026-11-02, before any weights were constructed or scored.
Same question as the repo's other experiments: can an LLM compile a
stated benefits formula into network weights, exactly, with no
training — and does that beat training the same architecture?

## The rule (stipulated in full; no recall required)

SSI federal monthly payment, individual, 2026. Inputs: monthly earned
income E, monthly unearned income U.

- $20 general exclusion applies to unearned income first:
  countable unearned CU = max(0, U − 20); unused part spills to earned:
  R = max(0, 20 − U)
- Countable earned CE = max(0, E − 65 − R) / 2 ($65 earned exclusion
  plus the spill, then one-half disregarded)
- Countable income C = CU + CE
- Payment P = max(0, 994 − C)

Constants: 2026 Federal Benefit Rate for an individual = $994 (SSA);
exclusions per 42 USC 1382a(b) / 20 CFR 416.1112 (unearned) and
416.1124 (earned). Note the spill (R) is real law — and is one of the
details the benefits-as-code oracle behind CivBench gets wrong (its
$20 exclusion never spills; see the CivBench SSI audit, 2026-11-01).
This stipulation follows the CFR, not that oracle.

Stipulated simplifications: individual only (no couple rate $1,491);
federal payment only (no state supplementary payment); no student
earned-income exclusion, no in-kind support / presumed maximum value,
no PASS/IRWE exclusions; no rounding (SSA rounds at several steps in
production); categorical eligibility and the $2,000 resource limit are
out of scope — this is the payment arithmetic only. Domain: E ∈ [0,
$3,000], U ∈ [0, $1,500] (both comfortably past the break-even points:
earnings alone break even at $2,073, unearned alone at $1,014).

Why this rule is a harder test than SNAP: the spill makes the earned
exclusion's kink plane depend on U (the boundary E = 65 + max(0, 20−U)
is itself piecewise), and the payment floor applies to a sum of two
differently-scaled pieces — staging depth, not just hinges.

## Architecture

Inputs [E/1000, U/1000] → Linear(2→24) → ReLU → Linear(24→24) → ReLU →
Linear(24→24) → ReLU → Linear(24→1); output dollars = 1000 · (final
linear). Three hidden layers: the construction needs one layer for the
exclusions, one for the earned excess (which depends on the spill),
one for the payment floor applied to the assembled countable income.

## Arms

1. **Zero-shot weight-written** — reasoning from the rule text only;
   one shot; `.npz` hashed (FREEZE.json) before the scorer is run.
2. **Trained baseline** — same architecture, Adam on uniform oracle
   samples (15,000 steps × 8,192, seed 20261102).
3. **Oracle** — the stipulated formula; the ceiling.

## Evaluation

Grid E $0–$3,000 × U $0–$1,500 in $25 steps (121 × 61 = 7,381) +
20,000 seeded uniform random points + 5,000 seeded kink-jittered
points (near U = 20, near the earned-excess plane, near the
payment-zero plane). Metrics: max / mean abs error, % within $0.01,
% within $1, region breakdown.

## Gate

PASS = zero-shot max abs error ≤ $0.01 on every set. Partial = max
≤ $1. Worse = fail, reported by region.
