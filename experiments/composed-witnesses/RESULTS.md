# Results — composing compiled witnesses, and the weight diff as a legislative diff

**Verdict: compiled witnesses compose exactly — bit-identical to
the one-piece build, at zero parameter overhead — and a
year-over-year weight diff of a compiled rule is the legislative
change itself: 5 entries carrying all three updated constants,
each recoverable from the diff alone. The trained control's diff
is neither: half its parameters move, in a pattern that decodes to
nothing.**

## Part 1 — composition

A countable-income net (SSI counting: $20 / $65+spill / one-half)
compiled on its own, then block-composed with a share-of-cost head
(max(C − 600 − M, 0)) into a single weight file:

| Check | Result |
|---|---|
| Countable net vs counting function | max err 4.5e-13 |
| Composed net vs SOC oracle (grid × M ∈ {0,100,300,600,1000}) | max err 4.5e-13 |
| Composed net vs the direct compiled SOC net | **max err 0.0** |
| Parameters: composed vs direct | total 1321 vs 1321; **nonzero 16 vs 16** |

The composed and one-piece constructions are not merely equal to
tolerance — on the full scoring grid their outputs are identical,
because both stage the same gadgets in the same order. Modularity
cost nothing here (prediction 2, which registered a small overhead,
fails in the pleasant direction): the direct writer had already,
in effect, composed the same modules inside one file. Witnesses
can be assembled — and therefore audited — module by module.

## Part 2 — the legislative diff

The repo's SNAP (E, S) rule compiled by the same writer under
FY2024 parameters (std $198, cap $672, allotment $535) and FY2026
parameters ($209 / $744 / $546); both nets exact vs their oracles
(≤ 3.4e-13). The weight diff, in full:

| Parameter | FY2024 | FY2026 | Encodes |
|---|---|---|---|
| b1[0] | −0.198 | −0.209 | standard deduction |
| b1[1] | 0.099 | 0.1045 | standard deduction (spill site) |
| b1[2] | −0.573 | −0.6395 | std/2 − shelter cap |
| b2[1] | −0.535 | −0.546 | max allotment |
| b3[0] | 0.535 | 0.546 | max allotment |

**Five entries. Nothing else in the 1,297 parameters moves.**
Decoding the diff against the FY2024 net (sites per the writer's
anatomy) recovers all three FY2026 constants exactly: allotment
$546, standard deduction $209, shelter cap $744. A COLA update to
a compiled witness is a five-line patch whose meaning is fully
readable before it is applied.

*Scope note:* the protocol framed Part 2 on the allotment constant
alone; the repo's actual SNAP rule carries three fiscal-year
constants, so the experiment ran on the real rule — the same
predictions against a stronger test.

**Trained control.** Two MLPs, same seed and initialization, one
trained per year (5,000 steps; max errors $34.07 / $21.28, the
usual trained-baseline class). Their diff: **648 of 1,297
parameters (50.0%) change by more than 1e-6, with deltas up to
1.92** — and no subset of the changed entries decodes to the
constants. Prediction 4's ≥ 95% figure **fails**: same-init
training leaves half the net untouched (dead pathways whose
gradients never fired — the protocol's own "what would change my
mind" section named this basin effect in advance). But the
qualitative claim survives intact: where the compiled diff is a
complete, readable account of the legislative change, the trained
diff is 648 numbers that tell you the function moved without
telling you what moved it. The honest statement of the contrast
is sparsity-plus-decodability (5 entries, exact recovery) vs
half-the-network opacity — not 5 vs 1,297.

## What this establishes

Composition and diffing are the two operations a maintenance
lifecycle needs: build witnesses from audited modules, and update
them year over year with patches a reviewer can read. Both work
on compiled nets for the same reason the stale-FBR edit worked —
the anatomy is known, so operations on weights are operations on
the rule. The five experiments in this batch close the loop the
project opened: weights can be **written** (compilation),
**read** (blind reading, recovery), **edited** (stale-FBR,
surgery), **audited** (tamper audit), and now **composed and
diffed** — atomically, in the weight space itself.

Artifacts: `compose.py`, `results/composed.json`. $0 spend.
