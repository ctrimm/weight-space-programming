# Protocol — SSI couple repair bake-off (pre-registered)

Written 2026-11-03 before training. Follows the split-invariance
audit (`../ssi-couple-split-invariance/`), which found the trained
couple baseline's output moves across legally equivalent household
compositions in 98.9% of totals classes (worst $25.15), because it
learned something like a per-member exclusion instead of the
once-per-couple rule.

## Question

Can ordinary training be *repaired* so a learned net respects the
invariance, or does the invariance have to be built into the
architecture? Four arms, same stipulated couple rule, same training
budget (Adam, 15,000 steps x 8,192 batch, lr 3e-3 cosine, seed
20261102) unless the arm's definition says otherwise:

1. **Vanilla** — the existing trained baseline, reused as-is
   (identical recipe and seed; retraining would only add noise).
2. **Augmentation** — same 4-input architecture; training batches
   are drawn as combined totals plus uniform random splits
   (totals E ~ U[0,4000], U ~ U[0,3000]; shares alpha, beta ~
   U[0,1]), so equivalent compositions appear constantly.
3. **Penalty** — same architecture and member-uniform sampling as
   vanilla, plus a symmetry loss: for each batch, a second forward
   pass on a random re-split of the same households' totals, with
   penalty weight 1.0 on the squared output difference (scaled
   units), added to the usual MSE.
4. **Sum-first architecture** — a 2-input network that receives
   (E1+E2, U1+U2) directly; same hidden widths (24/24/24) and
   recipe. Invariance holds by construction; the open question is
   its *accuracy*.

## Evaluation (identical for all arms)

- **Accuracy** on the couple experiment's standard design
  (`../ssi-couple-payment/score.py` point builder).
- **Split invariance** on the split-invariance experiment's grid
  (90 totals classes x 121 compositions): max / mean within-class
  range, share of classes over $0.01.
- Both reported for every arm; no arm is declared a repair on
  accuracy alone or invariance alone.

## Pre-registered expectations

- Augmentation shrinks the invariance defect substantially but does
  not reach float noise.
- Penalty shrinks it further than augmentation at some accuracy
  cost; it does not reach zero either.
- Sum-first has exactly zero invariance defect (float noise) and
  accuracy comparable to or better than vanilla, because its
  hypothesis space no longer contains the per-member-exclusion
  failure mode.

## Cost

$0. Local CPU training only.
