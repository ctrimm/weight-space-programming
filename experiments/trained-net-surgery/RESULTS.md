# Results — surgical weight edits on a trained network

**Verdict: direct weight editing survives contact with a trained
network — if the edit is structural. A four-line closed-form edit to
the first layer removes the trained SSI-couple net's split-dependence
exactly (max $30.56 → $0.00) and improves its accuracy (max error
$9.24 → $4.53), with no training at all. The generic alternative —
a sparse gradient edit of the 32 most-implicated weights — barely
denting the violation (−6%) while damaging typical-case accuracy
17×, is not a close second. For behavioral defects, knowing the
anatomy beats editing few weights.**

## Measurements (identical probes for all arms)

Split-dependence: totals grid (E_tot ≤ 2400, U_tot ≤ 1600) × 25
compositions per total; dependence = max − min payment at fixed
totals. (The split-invariance experiment's own probe design recorded
$25.15 for this net; probe designs differ, maxima move a little —
$30.56 here.) Accuracy: 20,000 uniform cases vs the couple oracle.

| Arm | Split-dep max | Split-dep mean | Acc max err | Acc mean err | Params changed |
|---|---|---|---|---|---|
| Original trained net | $30.56 | $1.30 | $9.24 | $0.037 | — |
| **Arm 1: symmetrize W1** | **$0.00** (2.3e-13) | 1.8e-14 | **$4.53** | $0.029 | 96 (all of W1) |
| Arm 2: sparse gradient edit (32 params) | $28.65 | $1.69 | $5.75 | $0.648 | 32 |

## Predictions scored

1. *Arm 1 invariance becomes exactly 0.* **Holds** (2.3e-13, float
   noise). Arithmetic, as registered.
2. *Arm 1 accuracy stays within 2× of original.* **Holds, in the
   good direction**: max error halves ($9.24 → $4.53). The split
   pathway was noise-fitting, as suspected; removing it helped even
   the sum function.
3. *Arm 2 cuts the violation ≥ 50% but not to 0, at worse accuracy
   cost than Arm 1.* **Fails on magnitude, holds on ordering.** The
   sparse editor managed only −6% on max violation and made mean
   split-dependence *worse* ($1.30 → $1.69), while mean accuracy
   degraded 17×. The pre-registered recipe (300 Adam steps, k = 32,
   anchor 0.01 — see surgery.py) is what was tested; a more patient
   generic editor might do better, and that caveat stands. But the
   gap is not a tuning gap: the structural edit is exact by
   construction and free.
4. *Edit size: Arm 1 changes 48 of 96 W1 entries.* **Wrong as
   drafted, mine.** W1 is (24, 4) and every column belongs to a
   spouse block, so the edit changes all 96 entries. The registered
   point survives in spirit — the structural edit is larger by count
   than the sparse edit and better by outcome — but the count in the
   protocol was an arithmetic error, recorded here rather than
   quietly fixed.

## Context and what this establishes

The repair bake-off's best training-side arm (augmentation) reached
split-invariance max $5.30 after full retraining. Post-hoc surgery
reaches exactly $0.00 with one closed-form edit and improves
accuracy at the same time. Combined with the stale-FBR repair, the
pattern across this repo is now: **compiled nets can be edited
because their anatomy is known, and trained nets can be edited
exactly where their defect has a structural description** — here,
"the function must depend on spouse inputs only through their
sums," which is a statement about W1's block structure. Where a
defect has no such description, the generic editor is what remains,
and Arm 2 is the measured price of that.

Artifacts: `surgery.py`, `results/surgery.json`. $0 spend.
