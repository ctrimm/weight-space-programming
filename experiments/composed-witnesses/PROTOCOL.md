# Protocol — composing compiled witnesses, and the weight diff as a legislative diff (pre-registered)

Written 2026-11-03, before any construction, training, or scoring
for this experiment. No API calls; $0 spend. All computation local
CPU.

## Why this experiment exists

Every witness in this repo is one rule per network, built in one
piece. Two composition questions were never tested:

1. **Do compiled nets compose like the code does?** Real programs
   share sub-rules: the medically-needy share of cost
   (`../medicaid-share-of-cost/`) counts income with the SSI rules
   ($20 / $65+spill / one-half) and then subtracts the maintenance
   need level. The direct compiled SOC net was written in one piece.
   The composed alternative — a compiled *countable income* net
   whose output feeds a compiled *share-of-cost head* — should be
   the same function, assembled the way the law is assembled. If
   composition is exact, witnesses can be built (and audited)
   module by module.
2. **Is a year-over-year weight diff the legislative change?** When
   a compiled rule's parameters are updated (COLA), the stale-FBR
   experiment showed the edit is tiny and locatable. The diff
   version of the same claim: compile the same rule under two
   fiscal years' parameters with the same compiler; the weight diff
   should be exactly the parameter change — sparse, and sufficient
   to reconstruct the new rule's constants without seeing them. For
   trained nets, the same diff should be meaningless: retrain on the
   new year's data and every weight moves.

## Method

**Part 1 — composition.** Build (a) a countable-income net
C(E, U) = CU + CE under SSI counting, compiled with the same DAG
construction style as the other witnesses; (b) a head net
H(C, M) = max(C − 600 − M, 0). Compose them into a single weight
file in the repo's mlp format (block composition: C's output unit
feeds H's input, intermediate layers concatenated block-diagonally).
Score the composed net against the SOC oracle and against the
existing direct compiled SOC net on that experiment's full scoring
grid. Also record the composed net's parameter count vs the direct
net's.

**Part 2 — legislative diff.** The SNAP size-2 allotment core:
benefit(N) = max(546 − 0.30·N, 0) clipped at the minimum benefit,
compiled twice with the same writer — FY2024 parameters (max
allotment 535, from the snap-qc-per generator's params) and FY2026
parameters (max allotment 546, this repo's `../snap-allotment/`
oracle). The compiled function is the allotment-and-rate core only
(the minimum-benefit floor is applied in code, per the boundary
doctrine, in both years). Diff the two weight files: count changed
parameters, and test whether FY2026's allotment constant is
recoverable from (FY2024 net + diff) alone. Control: two MLPs
trained with the repo's snap-allotment recipe, same seed and
initialization, one on each year's target; diff their weights the
same way.

## Predictions, registered in advance

1. Composed net == SOC oracle to < 1e-9 max error on the full grid,
   and == the direct compiled SOC net to the same tolerance: same
   function, different assembly. (If the two compilations differ
   beyond float noise, composition or the direct build has a bug
   and the experiment reports which.)
2. The composed net uses more parameters than the direct net
   (modularity is not free) — registered so the cost is on record,
   expected overhead < 2×.
3. Compiled diff: a small number of parameters change (registered
   bound: ≤ 8), every changed parameter encodes the allotment
   constant, and the new constant is recovered exactly from the
   diff: 546 = 535 + Δ read off the changed weights.
4. Trained-pair diff: ≥ 95% of parameters change despite the target
   function moving by a single $11 constant; the constant is not
   recoverable from the diff by any of the simple decodings tried
   (output-at-zero shift is recoverable behaviorally, but that is
   probing, not reading the diff).

## What would change my mind

If the trained pair's diff turns out sparse or decodable (e.g.
because same-init training keeps the solution in the same basin and
only the output bias absorbs the constant), prediction 4 fails in an
interesting direction: it would mean year-over-year diffs of
*retrained* models can also carry meaning under controlled
conditions, and the compiled/trained contrast needs that qualifier.
