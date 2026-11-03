# Experiment: SSI federal payment, couple — Protocol (pre-registered)

Written 2026-11-02 before construction. Witness-pattern extension,
companion to `../medicaid-share-of-cost/`.

## Why this rule

The CivBench SSI audit found the couple side of SSI is where the
upstream oracle's constants went stale ($1,415 = the 2024 couple
rate, shipped as 2026) and where its state-supplement figure was
wrong. Couple cases are also where a subtle counting rule lives:
an eligible couple's income is combined and the exclusions apply
**once, to the couple** — not once per person. An implementation
that applies the $20/$65 exclusions per spouse overstates the
disregard by up to $85/month of countable income. This witness
encodes the combined rule.

## The rule (stipulated in full)

SSI federal monthly payment, eligible couple, 2026. Inputs: each
member's monthly earned and unearned income (E1, U1, E2, U2).

- Combine: E = E1 + E2; U = U1 + U2.
- Countable unearned CU = max(0, U − 20); spill R = max(0, 20 − U)
  (exclusions apply once, to the combined amounts).
- Countable earned CE = max(0, E − 65 − R) / 2.
- Payment P = max(0, 1491 − (CU + CE)); $1,491 is the 2026 federal
  benefit rate for a couple (SSA).

Stipulated simplifications: both members eligible for the full
month (no individual-rate months, no deeming from an ineligible
spouse — that is a different formula); federal payment only; no
rounding. Domain: E1, E2 in [0, $2,000]; U1, U2 in [0, $1,500].

What this tests: input-combination staging (the network's first
job is to add four inputs into two combined streams) and
once-per-couple semantics. Mechanically it reuses the individual
SSI staging; the honest expectation is that it is the easier of
this pair — its value is coverage of the exact slice where the
audited labels went stale, not new difficulty.

## Architecture

Inputs [(E1)/1000, (U1)/1000, (E2)/1000, (U2)/1000] →
Linear(4→24) → ReLU → Linear(24→24) → ReLU → Linear(24→24) →
ReLU → Linear(24→1); output dollars = 1000 · (final linear).

## Evaluation

25,000 seeded uniform random points over the 4-D domain + 5,000
seeded kink-jittered points (near the combined U = 20 plane, the
combined earned-excess plane, and the payment-zero plane) + two
structured slices: member 2 at zero (grid over E1, U1, $25 steps)
and member 1 at zero (same grid) — the slices check the combination
logic where one stream vanishes. Gate: PASS = zero-shot max abs
error ≤ $0.01 everywhere. Arms: zero-shot; trained baseline
(Adam 15,000 × 8,192, seed 20261102); oracle.

## Cost

$0 API spend (in-session construction, local training).
