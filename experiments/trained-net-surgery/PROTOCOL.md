# Protocol — surgical weight edits on a trained network (pre-registered)

Written 2026-11-03, before any editing or scoring for this
experiment. No API calls; $0 spend. All computation local CPU.

## Why this experiment exists

This repo's one weight edit so far (the stale-FBR repair) operated
on a **compiled** net, where the constant's location is known by
construction. The open question behind Cory's framing — direct
weight editing as an atomic operation — is whether editing survives
contact with a network nobody compiled: a trained net whose
representation is entangled and whose defect is behavioral, not a
stale constant.

The patient is the trained SSI-couple baseline
(`../ssi-couple-payment/trained/couple_weights.npz`). Its known
defect (structural property atlas; `../ssi-couple-split-invariance/`):
the couple rule depends only on the spouses' combined incomes
(exclusions apply once per couple), but the trained net's payment
moves when a fixed combined income is re-split between spouses —
split-dependence up to $25.15, with 98.9% of composition classes
affected. The repair bake-off (`../ssi-couple-repair-bakeoff/`)
attacked this from the training side (augmentation, penalty,
architecture). Nobody has attacked it from the weight side, after
training, without retraining.

## Method

Two edit arms, both applied to the frozen trained weights; no arm
retrains the network.

- **Arm 1 — structural symmetrization (exact, zero optimization).**
  The net's inputs are ordered [E1, U1, E2, U2]. Replace the first
  layer's spouse-1 column block and spouse-2 column block with their
  average. Layer-1 pre-activations then depend on the inputs only
  through (E1+E2, U1+U2), so the edited net is split-invariant *by
  construction*, whatever the deeper layers compute. One edit, four
  lines, no data.
- **Arm 2 — sparse gradient edit (generic editor control).** Freeze
  all but the k = 32 parameters with the largest gradient of a
  split-dependence penalty (sampled composition pairs, same totals,
  different splits) and descend the penalty plus an anchor term
  holding outputs near the original net's on a fixed case grid.
  This is the "editor who doesn't know the structure" baseline:
  generic minimal editing, no anatomical insight.

Measurements (identical for original, Arm 1, Arm 2): split-dependence
(max and mean |Δpayment| over composition pairs at fixed totals, the
split-invariance experiment's probe design), accuracy vs the couple
oracle (max / mean error on the standard scoring grid from
`../ssi-couple-payment/score.py`), and edit size (number of parameters
changed, max |Δw|).

## Predictions, registered in advance

1. Arm 1: split-dependence becomes exactly 0.0 (to float precision).
   This is arithmetic, not a hope.
2. Arm 1: accuracy cost is modest — max error vs the oracle stays
   within 2× the original trained net's max error. Direction not
   registered: averaging destroys the split pathway the net used,
   which was fitting noise around a sum-only rule, so an improvement
   would not surprise; a degradation within 2× is the bound.
3. Arm 2: violation falls by ≥ 50% but does not reach 0 (a generic
   sparse editor suppresses the symptom on sampled pairs; it cannot
   install the invariant), and its accuracy cost is worse than
   Arm 1's per unit of violation removed.
4. Edit size: Arm 1 changes exactly half of W1's entries (48 of 96);
   Arm 2 changes 32 parameters anywhere. The structural edit is
   larger by count and better by outcome — i.e., for behavioral
   defects, *knowing the anatomy* beats *editing few weights*.

## What would change my mind

If Arm 2 reaches exact invariance at lower accuracy cost than Arm 1,
generic sparse editing is stronger than this protocol expects and
the anatomy-first story weakens. If Arm 1's accuracy collapses
(> 2× error), the trained net was leaning on split information in a
way that matters for the sum function too, and "the defect was pure
noise-fitting" is wrong.
