# Results — SSI couple split invariance

**Verdict: the compiled networks are exactly invariant; the trained
baseline is pervasively not.** Under the stipulated couple rule, the
payment depends only on the combined income totals, so every
reallocation of the same totals between the two spouses is the same
household and must get the same payment. Across 90 totals classes x
121 compositions (plus 5,000 random classes x 8 splits), the three
compiled nets never move by more than a trillionth of a dollar. The
trained net's answer depends on who carries the income in
essentially every class.

## Invariance range (max − min output within a totals class)

| Arm | Grid max | Grid mean | Classes > $0.01 | Classes > $1 | Random max | Random mean |
|---|---|---|---|---|---|---|
| Original (compiled) | 6.7e-13 | 3.9e-14 | 0% | 0% | 8.9e-13 | 2.3e-13 |
| GPT-5.6 Sol (compiled) | 2.2e-13 | 9.9e-15 | 0% | 0% | 4.5e-13 | 2.5e-14 |
| Gemini 3.1 Pro (compiled) | 2.2e-13 | 9.9e-15 | 0% | 0% | 4.5e-13 | 2.5e-14 |
| **Trained baseline** | **$25.15** | **$3.64** | **98.9%** | **72.2%** | **$24.92** | **$1.23** |
| Oracle (control) | 0 | 0 | 0% | 0% | 4.5e-13 | 8.6e-15 |

On the random arm the trained net exceeds a $0.01 range in 99.9% of
classes and $1 in 41.0%. GPT's and Gemini's range statistics are
identical to each other at every reported digit — two different
parameterizations that are indistinguishable even in their float
noise on this design. The oracle control validates the harness
(exactly 0 on the grid; float summation noise only on the random
arm).

![Left: the trained net's invariance range across combined totals — concentrated at low earned income with unearned income in the exclusion/spill band. Right: its output across splits at the worst totals class, moving about $25 with the unearned split while the correct answer is flat](figures/fig_split.png)

## Anatomy of the worst class

Combined earned $0, combined unearned $250. Correct payment:
**$1,261.00 at every split.** The trained net returns $1,260.26
when member 2 carries all the unearned income — nearly right — and
$1,285.41 when member 1 carries 75% of it. The entire range is
driven by the unearned split, which is the fingerprint of the
specific mistake: the trained net learned something close to
applying the $20 exclusion *per member* rather than once to the
combined amount. That is the same once-per-couple semantics the
couple experiment was built to test, failing in the learned
approximation even though its average error on the standard design
looked modest ($0.32 mean).

## Accuracy, kept separate

Invariance is not accuracy, and this grid is deliberately
boundary-heavy (extreme totals classes are weighted equally with
ordinary ones), so trained-arm accuracy here is worse than on the
standard evaluation design: max $50.29, mean $7.75, 5.6% within a
cent across grid compositions, versus max $48.11 / mean $0.32 /
21.2% on the standard design. The compiled arms remain exact
against the oracle on every composition, as their range statistics
imply. The point of the experiment is the first table: a model can
post a tolerable mean error and still treat legally identical
households differently almost everywhere.

## What this establishes

- **Member-order swapping was the tip of the measurement.** The
  general object is the equivalence class of compositions sharing
  combined totals, and the general metric is the within-class
  range. It costs nothing to compute and it catches a failure mode
  that random-point accuracy dilutes.
- **The trained net's defect is structural, not noisy.** Its worst
  region sits exactly on the exclusion/spill band, and its worst
  class decomposes into a recognizable wrong rule (per-member
  exclusion). Learned approximations do not merely blur the right
  rule; they can quietly implement a different one.
- **Compiled witnesses carry the invariance for free**, because the
  construction's first operation is the combination the rule
  specifies. All three writers' nets — including two written from
  bare prompts by other vendors' models — are invariant to the last
  float digit.

For the CivBench label program the implication is a reporting
standard: any learned scorer or oracle should be published with its
invariance defects alongside its accuracy, on the transformations
the law itself declares irrelevant.

## Cost

**$0.** Local numpy over previously frozen weights; no API calls,
no training. (The repo's cumulative API total is in the top-level
README's cost section.)
