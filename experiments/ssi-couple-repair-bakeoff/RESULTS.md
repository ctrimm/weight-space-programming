# Results — SSI couple repair bake-off

**Verdict: of the training-side repairs, augmentation is the
stronger one; neither comes close to eliminating the defect. The
architectural repair eliminates it exactly — the sum-first net's
invariance range is 0.0, not approximately zero — but pays for it
in accuracy at this training budget. The compiled constructions
remain the only entries that are both exact and invariant.**

| Arm | Accuracy max | Accuracy mean | Within $0.01 | Invariance max range | Invariance mean range | Classes moving > $0.01 |
|---|---|---|---|---|---|---|
| Vanilla (existing baseline) | $48.11 | $0.319 | 21.2% | $25.15 | $3.64 | 98.9% |
| Augmentation | $8.98 | $0.063 | 43.4% | $5.30 | $0.88 | 95.6% |
| Penalty (symmetry loss) | $48.07 | $0.183 | 30.9% | $19.28 | $1.65 | 93.3% |
| Sum-first architecture | $25.14 | $0.220 | 26.8% | **0.0 exactly** | **0.0** | **0.0%** |
| Compiled constructions (reference) | ~5e-13 | ~1e-14 | 100% | ~5e-13 | ~1e-13 | 0.0% |

![Left: worst accuracy error per arm. Right: worst invariance range per arm, corrected figures. Augmentation dominates the penalty arm on both axes; sum-first's invariance bar is at the floor because its range is exactly zero](figures/fig_bakeoff.png)

## Arm by arm, against the pre-registered expectations

- **Augmentation** (train on random splits of random totals):
  defect shrinks ~5x at the max and ~4x at the mean, accuracy
  improves on every measure — but 95.6% of classes still move.
  Showing the net millions of equivalent compositions teaches it
  to keep them *closer together*, not *equal*. Expectation
  confirmed.
- **Penalty** (vanilla sampling + a symmetry loss against a random
  re-split, weight 1.0): the pre-registered expectation was that
  this would beat augmentation. **It did not.** At this budget and
  penalty weight it is worse than augmentation on invariance max
  ($19.28 vs $5.30), invariance mean ($1.65 vs $0.88), and both
  accuracy measures; its only edge is the share of classes moving
  at all (93.3% vs 95.6%). A penalty prices a violation softly,
  and at λ = 1.0 the price was evidently too low to reshape the
  fit — recorded as a failed expectation, not tuned post hoc.
- **Sum-first** (the net receives only the combined totals):
  invariance is exact — max and mean range 0.0 across all 90
  classes, because on this grid the combined inputs are
  bit-identical across compositions and in general because the
  per-member pathway does not exist. Accuracy, however, is
  mediocre (max $25.14, mean $0.220): trained on member-uniform
  samples, its totals distribution is triangular and thin exactly
  at the extreme totals where the kinks bite. The expectation
  "accuracy comparable or better" was **wrong**; architecture buys
  the structural property exactly, and at a fixed budget it can
  cost fit quality elsewhere.

## What this establishes

The remediation ladder, as measured (not as hypothesized):

1. **Data augmentation** — the best training-side repair on both
   axes here; the defect still survives in 95.6% of classes.
2. **Symmetry penalties** — at λ = 1.0, weaker than augmentation
   on nearly every measure; the defect survives in 93.3%.
3. **Architecture that cannot express the violation** — defect
   exactly zero, by construction rather than by fit; accuracy is
   a separate, ordinary approximation problem that the constraint
   does not solve.
4. **Compilation from the rule** — defect zero and values exact.

So the standing conclusion survives, in a sharper form: if a
property matters, build it in — training toward it, by either
method tried here, leaves legally identical households treated
differently in more than 9 cases out of 10 — but do not expect the
architectural constraint to improve the fit for free.

## Erratum (2026-11-03, same day)

The first version of this file reported penalty and sum-first
figures (penalty invariance max $2.60; sum-first accuracy max
$3.33, "most accurate learned arm") that came from a corrupted
background-process result delivery, not from scoring the weights.
The scorer also had a latent bug on the sum-first path (it called
a function the couple score module does not export), which the
corrupted delivery masked. Both are fixed; every number above was
produced by re-scoring the shipped weight files in a foreground
run, and `results/bakeoff_scores.json` is that run's output. The
augmentation and vanilla figures were identical across both
deliveries and are unchanged.

## Cost

**$0.** Local CPU training (four 15,000-step runs). Process note:
the first training pass's output files were lost to a sandbox
filesystem restart before scoring; the arms were retrained, and
the weights and scores in this folder are the retrained run's.
