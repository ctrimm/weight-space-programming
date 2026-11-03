# Weight-Space Programming — Benefits Rules Test: Protocol (pre-registered)

Written 2026-11-02, before any weights were constructed or scored.
Motivation: Rikiya et al., "Sparks of Weight-Space Programming"
(https://rikiyat.github.io/blog/weight-space-programming/, 2026-11-02) report a
frontier LLM writing small game-playing networks from scratch, zero-shot, that
beat PPO trained for 100M steps, and fixing a 0.8B language model by editing
0.016% of its weights. This test asks the same question in our domain: **can an
LLM compile a stated benefits rule into neural network weights, exactly,
with no training — and does that beat what training the same architecture
achieves?**

The weight-writer in this run is Muse Spark, in-session (the same
provenance pattern as the 2026-10-08 / 2026-11-01 CivBench runs). It is not
the paper's model; a weaker result is expected and is itself the finding.

## The two rules (both stipulated in full; no recall required)

### Rule A — EITC, TY2025, single/head-of-household, 2 qualifying children
Credit as a function of earned income x (stipulation: AGI = earned income,
the case where the §32 formula is a single-input piecewise-linear function):

- x ≤ $17,880: credit = 0.40 · x
- $17,880 < x ≤ $23,350: credit = $7,152
- $23,350 < x < $57,310: credit = $7,152 − 0.2106 · (x − $23,350)
- x ≥ $57,310: credit = $0

Constants: 26 USC 32(b)(1) (credit % 40, phaseout % 21.06 — not
inflation-indexed) and Rev. Proc. 2024-40 §2.06 via NY Tax Dept EITC
parameters page, TY2025 (earned income amount $17,880; max credit $7,152;
phaseout threshold $23,350; completed phaseout $57,310). Internal
consistency check: 0.40 × 17,880 = 7,152 exactly; 7,152 / 0.2106 = 33,960.0,
and 23,350 + 33,960 = 57,310 exactly. Domain: x ∈ [0, $70,000].

### Rule B — SNAP monthly allotment, FY2026, household of 2, earned income only
Inputs: monthly gross earned income E, monthly shelter cost S (rent +
utilities combined). Household has no elderly/disabled member.

- Adjusted income A = 0.80·E − 209  (20% earned-income deduction, then the
  $209 standard deduction for households of 1–3)
- Excess shelter X = max(0, S − 0.5·A), capped: Xc = min(X, 744)
- Net income N = max(0, A − Xc)
- Allotment B = max(0, 546 − 0.30·N)

Constants: USDA FNS FY2026 COLA (48 contiguous states): max allotment for 2
= $546; standard deduction (1–3 persons) = $209; earned income deduction
20%; maximum excess shelter deduction = $744; benefit = max allotment −
30% of net income (7 CFR 273.9 / 273.10 formula structure).

Stipulated simplifications (logged so the target is unambiguous; each one
keeps the function continuous piecewise-linear):
- Earned income only (no unearned income line).
- Domain restricted to E ∈ [$300, $4,000], so A > 0 throughout and the
  shelter-deduction base is never negative.
- No gross/net income eligibility gates, no minimum-benefit floor ($24),
  no rounding to whole dollars, no dependent-care / child-support /
  medical deductions, shelter cost given as a single combined figure
  (no separate standard utility allowance).

## Architectures (fixed, standard MLPs; forward pass is generic)

- Rule A net: inputs [x/10000] → Linear(1→16) → ReLU → Linear(16→1);
  output dollars = 1000 · (final linear). 
- Rule B net: inputs [E/1000, S/1000] → Linear(2→24) → ReLU →
  Linear(24→24) → ReLU → Linear(24→1); output dollars = 1000 · (final linear).

The scaling constants are architecture, not weights. Weights are plain
arrays in `.npz` files; the scorer loads them and runs a generic numpy
forward pass. Nothing rule-specific may live in forward code.

## Arms

1. **Zero-shot weight-written (the test).** Weights constructed by reasoning
   from the rule text above only. The constructor may use arithmetic on the
   rule constants to compute weight values (the paper's Exp. 6 pattern: the
   model writes a Python program that sets the weights). It may not train,
   fit, sample the oracle, or see any evaluation points or scores. One shot:
   the `.npz` files are hashed (SHA-256, `FREEZE.json`) before the scorer
   is run. Any later revision is a labeled attempt 2 and does not replace
   attempt 1 in the report.
2. **Trained baseline (the comparator).** Same architectures, weights
   trained with Adam on samples drawn from the oracle (Rule A: 50k uniform
   samples; Rule B: 100k uniform samples; MSE on dollar output; fixed seed;
   trained in torch, exported to the same `.npz` format). This is what
   "just train it" buys.
3. **Oracle (the ceiling).** The stipulated formulas implemented directly
   in Python. Scores 0 error by construction; included so every error
   number is plainly a property of the network, not the target.

Out of scope as comparators: the CivBench bac-eitc / bac-snap fixtures.
Their gold is eligibility + citations from the full benefits-as-code
oracle (categorical gates, BBCE, unearned income, …), not the allotment /
credit amounts isolated here. This test isolates the arithmetic core that
both oracles share.

## Evaluation (fixed before scoring)

- Rule A grid: x = $0…$70,000 in $25 steps (2,801 points), plus every
  breakpoint ± {$1, $5, $25} probes.
- Rule B grid: E = $300…$4,000 in $25 steps × S = $0…$2,500 in $25 steps
  (149 × 101 = 15,049 points), plus 20,000 seeded uniform random points,
  plus 5,000 seeded points jittered near the rule's kink planes
  (S − 0.5A ∈ {0, 744}, N = 0, 546 − 0.3N = 0).
- Metrics per arm: max absolute error ($), mean absolute error ($),
  % of points within $0.01, % within $1. Errors broken out by region
  (which linear piece) for Rule B.

## Gates (pre-registered)

- **PASS — exact compilation:** zero-shot max abs error ≤ $0.01 on every
  evaluation set. (Float tolerance; the constructions, if right, are exact
  up to floating point.)
- **Partial:** max error ≤ $1 but > $0.01 — the shape was understood, the
  arithmetic was not exact.
- **Fail:** anything worse. Report which region fails; do not tune and
  re-report silently.
- **Baseline hypothesis:** the trained net will show small mean error but
  max error ≫ $0.01 near kinks — i.e., training approximates this function,
  weight-writing (if it works) *is* this function. If the trained net also
  hits ≤ $0.01 max error, the honest conclusion is that training was
  already sufficient and weight-writing adds nothing on rules this small.

## What this does not show

Nothing about eligibility determination (which stays deterministic code,
per standing principle), nothing about rules with discontinuities a
continuous network cannot represent exactly (eligibility step gates),
and nothing about the paper's larger claims (reading weights, editing
0.8B models). This is the smallest falsifiable slice: stated rule in,
exact weights out.
