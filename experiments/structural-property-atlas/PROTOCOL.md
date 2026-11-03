# Protocol — structural-property atlas (pre-registered)

Written 2026-11-03 before the harness was run. Companion to the
split-invariance audit: that experiment measured one structural
property (composition invariance) deeply on one rule. This one
measures the structural properties each stipulated rule implies,
across all six rules and every network in the repo.

## Principle

A benefit formula implies more than point values. It implies
*properties*: directions of monotonicity, bounds, invariances,
plateaus, and regions where the answer is pinned by a single
constant. A network can match the formula's values on a random
test set while violating its properties — the trained couple net
did exactly that. This atlas scores properties directly.

A property violation is reported as an **adverse amount** in
dollars: how far the network's behavior goes against the property
at a probe. Per property and arm: the share of probes with adverse
amount > $0.01, and the maximum adverse amount.

## Arms (per rule, as available)

Original in-session construction; GPT-5.6 Sol; Gemini 3.1 Pro;
trained baseline. (EITC has no GPT arm — it declined to submit;
that cell is recorded, not filled.)

## Properties by rule

- **SNAP:** allotment nonincreasing in earned income (+$100
  probes); nondecreasing in shelter cost (+$100); output within
  [0, 546].
- **EITC:** nondecreasing across the phase-in region; constant on
  the plateau (deviation from the value at $19,000); nonincreasing
  across the phase-out region; output within [0, 7152].
- **SSI (individual):** payment nonincreasing in earned and in
  unearned income (+$50 probes); the first-$20 property — at zero
  earned income, payment equals the full $994 for all unearned
  income in [0, 20]; output within [0, 994].
- **Medicaid patient liability:** liability nondecreasing in
  resident income (+$100) and in community-spouse income (+$100,
  because the spousal allowance shrinks); nonincreasing in
  premiums (+$50); output >= 0 and <= resident income.
- **Medicaid share of cost:** nondecreasing in earned and unearned
  income (+$10); nonincreasing in medical expenses (+$10); output
  >= 0; the two CivBench audit probes reproduced ($1,380 / $880).
- **SSI couple:** member-swap invariance on the standard design
  (the full split-invariance grid lives in its own experiment);
  payment nonincreasing in each of the four member incomes (+$50,
  worst input reported); output within [0, 1491].

Probes are drawn from each experiment's own standard evaluation
design (same point builders and seeds as its scorer), plus the
named property-specific probes above.

## Pre-registered expectations

- Every compiled arm: no property violation above float noise
  anywhere (adverse amounts < 1e-9, 0% of probes over $0.01).
- Every trained arm: violations in most property families, worst
  in invariance and narrow-band properties; bounds violations
  (negative outputs, outputs above the statutory maximum) are
  expected for at least some trained arms and are the most
  damning class, because no correct implementation can produce
  them at all.

## Cost

$0. Local numpy over previously frozen weights.
