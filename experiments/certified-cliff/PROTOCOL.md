# Protocol — certified error bands at an eligibility cliff (pre-registered)

Written 2026-11-03, before any construction, training, or scoring
for this experiment. No API calls; $0 spend. All computation local
CPU.

## Why this experiment exists

The rounding boundary (`../ssa-rounding-boundary/`) proved the
one-dimensional version of a general fact: a continuous network
cannot implement a jump of size J with max error below J/2, and a
constructed ramp attains exactly J/2. Rounding jumps are small
($0.10). The jumps that dominate benefit law are not: an eligibility
cliff — pass the gross income limit by $1 and the entire benefit
disappears — has J equal to the whole benefit. This experiment takes
the rounding result to the cliff and asks the question the rounding
experiment did not: **what does a narrower error band cost?**

The trade is intuitive — a steeper ramp confines the disagreement
with the true rule to a narrower income band, but a steeper ramp
needs larger weights. Intuition is not a curve. This experiment
measures the curve and certifies the band: for the constructed nets,
the disagreement region is not estimated from samples, it is known
from the construction (the net equals the rule outside the ramp by
construction), and the sampling is a check on the certificate, not
the source of it.

## Setup

Rule (stipulated, SNAP-shaped): household of 2, benefit = $546 if
gross monthly income G ≤ $3,000, else $0. One input, one cliff,
J = $546. (The numbers are the FY2026 size-2 allotment and a round
limit; the experiment is about the shape, not the program.)

- **Constructions**: ReLU ramp nets of width w ∈ {400, 200, 100, 50,
  10, 2} dollars centered on the limit: output = 546 · clip((L +
  w/2 − G)/w, 0, 1), written directly as weights in this repo's mlp
  format with the standard /1000 input scaling. For each: certified
  band (by construction), measured disagreement region on a dense
  grid (|net − rule| > $0.01), max error, and max |weight|.
- **Trained control**: the repo's standard small MLP recipe trained
  on the same rule (sampler: G uniform on [0, 6000] with extra mass
  near the cliff; the honest-tuning standard applies — it gets the
  samples it needs). Measured: max error, effective disagreement
  region on the same dense grid. Its band is whatever training left;
  nothing about it is certified.

## Predictions, registered in advance

1. Every construction's max error is exactly J/2 = $273, at every
   width — the floor from the rounding proof, attained, width-free.
2. Each construction's measured disagreement region equals its
   certified band w (to grid resolution): the certificate is exact,
   not conservative.
3. Max |weight| grows like 1/w: halving the band doubles the largest
   weight (slope 546/w in dollar units, × the /1000 input scaling).
   The band–magnitude curve is a hyperbola, and at w = $2 the
   largest weight is ~250× the w = $400 value.
4. The trained control's max error will be at or above $273 × 0.9
   (it cannot beat the floor either — continuity binds it too), but
   its disagreement region will be wider than the w = $50
   construction's and asymmetric around the cliff: it will pay the
   error without the certificate.

## What this does and does not claim

A certified band is not a usable eligibility system: inside the band
the net is wrong by up to $273, and determinations belong in
deterministic code (this repo's standing position). The claim is
narrower and, for verification purposes, the operative one: where a
continuous approximation to a discontinuous rule is used at all —
as a witness, a triage score, a cross-check — the size of the region
in which it can disagree with the rule is a *choice with a price*,
and the price is readable from the weights.
