# Protocol — canonical circuits under gauge transformations (pre-registered)

Written 2026-11-03 before the analysis was run.

## Background

ReLU networks have gauge freedoms — transformations of the weights
that leave the computed function unchanged:

- **Permutation:** reordering the hidden units of a layer (with
  their incoming rows, biases, and outgoing columns carried along).
- **Positive scaling:** multiplying a hidden unit's incoming
  weights and bias by s > 0 while dividing its outgoing weights by
  s (ReLU is positively homogeneous).
- **Sign pairing:** representing a signed quantity as the
  difference of two ReLU units instead of one — a genuinely
  different encoding choice, *not* a relabeling.

Across this repo, independent writers produced functionally
identical networks with visibly different parameter values (and, in
one case, literally identical arrays). This experiment asks how
much of that difference is mere gauge.

## Method

Canonicalization by iterative refinement (Weisfeiler–Lehman style)
over the layered unit graph:

- Each hidden unit is normalized by its own positive scale
  (s = max(|incoming weights|, |bias|); incoming and bias divided
  by s, outgoing weights multiplied by s), removing the scaling
  gauge.
- Unit colors are refined iteratively: a unit's color hashes its
  normalized bias plus the sorted multisets of (normalized incoming
  weight, source-unit color) and (outgoing weight, target-unit
  color), with input units labeled by their input identity and the
  output unit labeled. Refinement runs to stability.
- A net's canonical form is the per-layer sorted multiset of final
  colors, over *active* units only (units that can output something
  nonzero; dead units are counted separately).
- Two nets are reported **gauge-equivalent** if their canonical
  forms are identical at 6-decimal rounding. Non-equivalence is
  reported as "not equivalent under permutation + positive
  scaling" — sign-pair encodings and WL-indistinguishable
  structures can still differ, and the functional scores (already
  known) are the arbiter of sameness of function.

## Controls

- **Positive control:** Gemini's share-of-cost net is
  array-identical to the original — it must come out equivalent.
- **Negative control:** the original EITC attempt-2 net and
  Gemini's EITC net compute *different* functions (they differ at
  the $57,310 seam) — they must come out non-equivalent.
- **Matched-behavior pair:** the original EITC attempt-1 net and
  Gemini's net are functionally identical; whether they are also
  gauge-equivalent is an open result, not a control.

## Pairs

Every writer pair, per rule: SNAP, SSI, Medicaid liability, share
of cost, SSI couple (3 pairs each), EITC (attempt-1 vs Gemini;
attempt-2 vs Gemini). Also reported: active-unit counts per layer
per net (a parsimony measure that is gauge-invariant, unlike raw
nonzero-parameter counts).

## Cost

$0. Local analysis of frozen weights.
