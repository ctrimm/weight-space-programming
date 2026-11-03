# Protocol — boundary case: SSA rounding (pre-registered)

Written 2026-11-03, before any construction, training, or scoring
for this experiment. No API calls; $0 spend.

## Why this experiment exists

Every rule in this repo so far is continuous and piecewise linear,
which is exactly the function class a ReLU network represents
exactly. The obvious boundary question: what happens with a rule
that is **not continuous**? US benefit law is full of them —
rounding rules and lookup tables. This experiment takes the one
that sits in the same SSA text as the held-out PIA test
(`../ssa-pia-heldout/SOURCE_TEXT.md`):

> "We round this amount to the next lower multiple of $.10 if it is
> not already a multiple of $.10."

(the PIA dime-rounding rule), with the whole-dollar rounding of
the monthly benefit ("before rounding down to next lower whole
dollar", same source) treated as a corollary.

## Prediction, registered in advance

A ReLU network computes a continuous function. The dime-rounding
function jumps by $0.10 at every multiple of $0.10. Claim: **no
continuous function can approximate it with max error below
$0.05** (half the jump) on any interval containing a jump in its
interior. Proof sketch: at a jump point m the true value is m
(right-continuous); approaching from the left the true value is
m - 0.10. A continuous candidate f takes a single value v at m;
its error at m is |v - m| and its supremum error just left of m
approaches |v - (m - 0.10)|. The better of the two errors is at
least $0.05 for every v, so the max error is at least $0.05.
Corollary: the whole-dollar rounding rule has a floor of $0.50.

Therefore the repo's PASS gate (max abs error <= $0.01) is
**provably unattainable** for this rule, for any network, any
writer, any training procedure. The experiment succeeds if the
measured arms respect the floor and match the predicted error
pattern (error concentrated at the jumps, near zero elsewhere).

## Arms

1. **Best construction (real network, small range).** A ReLU net
   implementing centered ramps at every dime jump on PIA range
   $0–$20 (200 jumps, one hinge pair per jump): equals the true
   rounded value everywhere except within $0.005 of a jump, where
   it ramps through the midpoint. Predicted max error: exactly
   $0.05. Predicted share of points within $0.01 on a dense grid:
   about 92% (the ramps cover 10% of the axis at half-width
   $0.005, and error exceeds $0.01 only in the inner part of each
   ramp).
2. **Best construction (closed form, full range).** The same
   centered-ramp function evaluated in closed form on PIA
   $0–$5,000, dense grid. Predicted max error $0.05, mean error
   near $0.0005 (ramps occupy 0.1% of each dime cell at
   delta = $0.0005 half-width; the writeup uses delta = $0.005
   for the network arm and $0.0005 here, and says so).
3. **Trained baseline.** A 1-64-64-1 ReLU MLP trained on the
   rounding rule over $0–$5,000 (repo baseline recipe, shortened
   for a 1-D problem; steps and seed recorded in the run script).
   Predicted: max error at or above $0.05, expected near $0.10;
   mean error on the order of $0.025.
4. **Composition.** The exact PIA network from the held-out
   experiment (control construction) scored against the **rounded**
   PIA oracle: isolates what the rounding step alone does to an
   otherwise exact pipeline. Predicted max error just under
   $0.10, mean about $0.05 — the rounding gap itself — before any
   network approximation of the rounding is even attempted.

## Scoring

- Rounding arms: grid $0–$5,000 in $0.001 steps over a $0–$50
  window for jump-resolved measurement, plus 200,000 uniform
  random points over $0–$5,000 (seed 20261103) for range-wide
  statistics; the small-range network arm is scored on $0–$20 in
  $0.0005 steps (40,001 points).
- Report for every arm: max abs error, mean abs error, share of
  points within $0.01, and the measured error at jump points vs
  away from jumps.
- Gate language: this experiment does not PASS/FAIL the rule; it
  verifies or refutes the predicted floor ($0.05) and pattern.
