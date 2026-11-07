# Weight-Space Programming for the Social Safety Net

**Can a large language model write a neural network that *is* a benefits
formula — exactly, weight by weight, with no training?**

This repo tests the claim of *weight-space programming* in the domain
where being almost right is not right at all: the arithmetic of the
American social safety net.

## In plain language: what this is

A neural network is, mechanically, a big list of numbers (its
"weights") plus a fixed recipe for combining them. Normally those
numbers come from training: show the network millions of examples,
nudge the numbers a little each time, and hope the result computes
what you wanted. Nobody chooses the numbers directly, because nobody
can hold that much arithmetic in their head.

These experiments test whether a language model can choose the
numbers directly. We took four real formulas from U.S. benefit
programs — how much SNAP a household gets, how much EITC a worker
gets, how an SSI payment shrinks as income rises, what a nursing-home
resident on Medicaid owes each month — and asked an AI to write down,
from the formula alone, the exact weights of a small neural network
that computes it. No training data. No trial and error. Then we
checked the network's answer against the formula on tens of thousands
of inputs.

In three of the four cases the written network is not approximately
right. It is the formula, to within a trillionth of a dollar. The
fourth is exact everywhere except one point where the IRS's own
published table contradicts the IRS's own rate by 2.4 cents (details
in the EITC experiment). For comparison, networks *trained* on
examples of the same formulas, the normal way, get close but are
wrong by dollars — and their biggest errors land on the thresholds
and caps where real cases are fought over.

## In plain language: what this is not

- **It is not a benefits calculator you should use.** Each formula
  here is a stripped-down slice of the real rules: one household
  type, a short list of simplifications, no rounding, none of the
  state variations. Every experiment's protocol lists what was left
  out. If you need an actual benefit amount, use your state's
  official calculator or a caseworker.
- **It is not a way to decide who gets benefits.** Real eligibility
  turns on categorical questions — age, disability, household
  composition — that are yes/no gates, not smooth arithmetic. These
  networks only reproduce the money math that runs *after* those
  gates. Nothing here suggests a neural network should decide
  anyone's eligibility, and the author would oppose using it that
  way.
- **It is not a replacement for the normal software that runs these
  programs.** A formula written as ordinary code can be read,
  audited, and litigated line by line. A bag of weights cannot. The
  plausible use for a network like this is narrow: a second,
  independently built copy of a formula, used to *check* the first
  copy. When the two disagree, a human goes and finds out why.
- **It is not proof that AI "understands" anything.** The networks
  here are tiny and the formulas are the kind of arithmetic
  (straight lines joined at breakpoints) that a network can, in
  principle, represent exactly. The finding is that a language model
  can work out how. Whether that extends to bigger networks or
  messier rules is an open question this repo does not answer.
- **The paper this builds on is not settled science either.** It is
  one group's blog post about a model the public cannot use yet.
  This repo is a small, checkable attempt to see if the core trick
  works on a problem where the right answers are known.

## The paper

Rikiya Tomita and collaborators, *Sparks of Weight-Space Programming:
LLMs can natively read & write neural network weights* (Nov 2026):
<https://rikiyat.github.io/blog/weight-space-programming/>

They report that a frontier LLM, given only the rules of a game, writes
small game-playing networks from scratch that outplay networks trained
with PPO for 100 million steps — and that, given a 0.8B language model
with a reasoning-loop defect, it finds the responsible direction in the
weights and removes it, editing 0.016% of the parameters and lifting
MATH-500 from 7.9% to 48.1% with no gradient descent anywhere.

Those are demonstrations on games and model surgery. This repo asks
whether the underlying skill — *reasoning about what a network's
weights must be for the network to compute a stated function* —
transfers to rules that decide what people eat, where they live, and
what care they can afford.

## Why benefits rules

Three reasons this domain is the right stress test, and not just a
novelty:

1. **The functions are exactly checkable.** A SNAP allotment formula is
   continuous piecewise-linear arithmetic with published constants.
   There is no vibes-based grading: a network either reproduces the
   formula to the cent on every input or it doesn't, and where it
   doesn't is itself diagnostic. Few domains offer a ground truth this
   hard.
2. **Approximation fails at exactly the points that matter.** A
   trained network that is right to within a dollar on 96% of inputs
   sounds production-ready — until you look at where the other 4%
   live: at the thresholds, caps, and phase-out boundaries where real
   cases are contested, appealed, and audited. Benefits administration
   already runs on deterministic engines whose whole value is being
   exact at the seams. Any neural artifact in this space has to be
   judged by that standard.
3. **Independent implementations are a live problem here.** Benefit
   rules are re-implemented by every state, vendor, and screening
   tool, and the implementations drift — audits of one widely used
   open-source oracle (see the CivBench label audits) found roughly
   10% of its computed labels wrong, concentrated at boundary cases.
   A weight-compiled network shares *no code path* with a Python or
   Rust oracle. If it agrees with them, that is independent evidence.
   If it disagrees, the disagreement points at a seam worth a human's
   attention. That — a witness, not a replacement — is the realistic
   role for this technique in the safety net.

## Method

Every experiment follows the same pre-registered discipline:

- The rule is **stipulated in full** in the experiment's `PROTOCOL.md`
  — constants verified against primary sources first — so the test
  measures compilation, not recall.
- The LLM **writes the weights by reasoning** (arithmetic on the rule
  constants is allowed; training, fitting, and sampling the oracle are
  not). One shot. The weights are **SHA-256 frozen before the scorer
  is written**, so the order of operations is auditable.
- The same architecture is **trained** with Adam on oracle samples as
  the comparator. (When a first training run visibly underfits, it is
  preserved and a competent run replaces it — strawmen prove nothing.)
- Scoring is a **generic forward pass**: the harness knows no rules.
  Evaluation sets include dense grids, uniform random points, and
  points jittered onto the rule's kink planes, because that is where
  errors hide.

## Experiments and findings

| # | Rule | Zero-shot written weights | Trained baseline (same arch.) | Gate |
|---|---|---|---|---|
| 1 | [SNAP allotment](experiments/snap-allotment/) — FY2026, household of 2 | max error $3×10⁻¹³ over 40,049 pts; 100% within $0.01 | max $5.35; 6.1% within $0.01 | **PASS (exact)** |
| 2 | [EITC credit](experiments/eitc-credit/) — TY2025, 2 children | exact at 99.96% of 2,818 pts; $0.024 at one seam point | max $5.19; 15.7% within $0.01 | Partial — see below |
| 3 | [SSI federal payment](experiments/ssi-payment/) — 2026, individual | max error $2×10⁻¹³ over 32,381 pts; 100% within $0.01 | max $9.54; 26.6% within $0.01 | **PASS (exact)** |
| 4 | [Medicaid patient liability](experiments/medicaid-patient-liability/) — post-eligibility, community spouse | max error $2×10⁻¹² over 86,105 pts; 100% within $0.01 | max $31.67; 3.3% within $0.01 | **PASS (exact)** |
| 5 | [SSI stale-FBR edit](experiments/ssi-stale-fbr-edit/) — experiment 3 recompiled with the stale 2024 FBR ($943), then repaired | stale net exact for 2024 but up to $51.00/mo wrong vs 2026; after a **2-parameter weight edit**: max error $2×10⁻¹³ over 56,381 pts | control, not a baseline: the trained SSI net contains no parameter equal to either year's FBR — a stale constant in trained weights is not even locatable | **PASS (surgical repair, 2 of 1,297 params)** |
| 6 | [Medicaid share of cost](experiments/medicaid-share-of-cost/) — CA medically-needy, SSI counting as a subroutine | max error $1×10⁻¹² over 43,607 pts; both CivBench audit probes reproduced exactly ($1,380 / $880) | max $32.73; 3.3% within $0.01; probes under by ~$22 | **PASS (exact)** |
| 7 | [SSI couple payment](experiments/ssi-couple-payment/) — 2026 couple rate $1,491, exclusions once per couple | max error $6×10⁻¹³ over 44,762 pts; 100% within $0.01 incl. both zero-member slices | max $48.11; 21.2% within $0.01; worst on the boundary slices | **PASS (exact)** |

![Worst-case error by experiment: written weights at 1e-13 to 2.4 cents, trained weights at 5 to 32 dollars](docs/figures/fig_max_error.png)

The headline comparison. Three of the four written networks sit at
floating-point noise (below $10⁻¹²); the fourth is the EITC seam
explained below. Every trained network, on the same architecture, is
wrong by dollars somewhere.

![SNAP allotment surface, trained network error tracing the kinks, written network error at float noise](docs/figures/fig_snap_surface.png)

SNAP, as a surface over income and shelter cost. The trained
network's errors (middle) draw a map of the formula's kinks — the
shelter-deduction boundary, the cap, the exhaustion corner. The
written network's error (right, log scale) is floating-point noise,
largest around $10⁻¹⁰.

![EITC curve with a zoom on the $57,310 seam](docs/figures/fig_eitc_curve.png)

EITC, full curve and seam zoom. At the scale of the full curve all
three lines coincide; the differences live in the zoom and in
Figure 1.

![SSI trained error heatmap, full domain and zoomed on the $20 spill band](docs/figures/fig_ssi_spill.png)

SSI. The trained network's errors hug the bottom edge, where the
$20 exclusion spill operates. In the zoomed panel the spill band
lights up across nearly the whole earnings range — the written
network encodes the spill exactly.

Regenerate all four with `python3 docs/make_figures.py` (numpy +
matplotlib only; every number is recomputed from the experiment
artifacts, nothing is drawn from the summary tables).

The EITC "Partial" is the most interesting result in the repo. The
written network is exact everywhere except a single point — $57,310,
the completed phase-out amount — where it is off by $0.024. That is
not a construction error: the IRS table rounds the phase-out's zero
to the nearest $10, so the rule *as published* is discontinuous by
$0.024 at that point, and $0.024 is the smallest error any continuous
network can have there. The experiment caught an inconsistency in its
own protocol's arithmetic (documented in that experiment's
`ERRATUM.md`) — published benefit constants may not multiply out
against their own statutory rates, and any scorer that treats rounded
table amounts as exact will manufacture phantom model errors at the
seams.

The structural pattern across all four: the written constructions are
small (2–4 active units per layer; the rest zeros) and staged —
hinges for breakpoints, min/max assembled layer by layer, floors
applied last. Training, by contrast, spreads the function across every
unit and pays for it at every kink: its worst errors sit exactly on
the kink intersections (e.g., SNAP's shelter-cap + benefit-exhausted
corner, mean error $1.81). Narrow rules fare worst of all under
sampling: SSI's $20 exclusion spill operates in a band 20 dollars
wide, uniform training data almost never lands there, and the trained
net's error in the spill region runs $4.88–$7.22 against $0.15
overall — the written weights encode the spill exactly, for free.

The fifth experiment tests the paper's other direction: *editing*.
The SSI formula was compiled with the stale 2024 benefit rate
($943 instead of $994) — the same defect the CivBench audit found in
published benefits-as-code labels. The stale network is a perfect
implementation of the wrong year: underpaying by up to $51.00 a
month. Scanning its weights for the literal value 0.943 finds the
rate in exactly two places (a hinge bias and the output bias), and
changing just those **2 of 1,297 parameters** to the 2026 rate
restores exactness — the edited file is array-for-array identical
to the network built fresh with the correct rate. The trained SSI
network, by contrast, contains no parameter equal to either year's
rate: in trained weights a constant is smeared everywhere and
addressable nowhere, so it could not be found or fixed without
retraining. Full record in `experiments/ssi-stale-fbr-edit/`.

A second-writer replication (2026-11-02,
`experiments/second-writer-openrouter/`) gave GPT-5.6 Sol and
Gemini 3.1 Pro the same four stipulated rules through OpenRouter,
one stateless prompt each. Both compiled SNAP, SSI, and Medicaid
exactly. On EITC they split: Gemini reproduced the original
writer's first attempt metric for metric (Partial, $0.024 seam),
while GPT-5.6 Sol returned no weights at all — it derived the
$0.024 discontinuity from the constants and declined to submit an
inexact network. Total measured cost of the replication: $0.65.

The reading direction is also done (2026-11-02,
`experiments/weight-reading/`): eleven anonymized networks were
handed to fresh readers — GPT-5.6 Sol, Gemini 3.1 Pro, and the
original author cross-reading the others' work — with only the
weights, the architecture, and the meaning of the inputs. Every
compiled network was read exactly: claimed formulas behaviorally
identical to the artifacts, every constant recovered. Given the
stale-FBR network plus a current-law sheet, both blind readers
flagged the stale $943 rate, one quantifying the harm at exactly
$51 a month. The trained SSI network broke the pattern in the
instructive way: both readers named the program but invented
different benefit rates ($1,000; $914) for a constant that training
never wrote down anywhere, and 25 input/output samples did not
rescue them (one reader's error quadrupled to $414). A compiled
network can be audited by a stranger; a trained one can only be
recognized.

**Two more witnesses** (`experiments/medicaid-share-of-cost/`,
`experiments/ssi-couple-payment/`). The share-of-cost construction
reuses the whole SSI income-counting machine as a subroutine inside
a second program — different threshold, an extra input subtracted
after the count — and reproduces the CivBench Medicaid audit's two
corrected values ($1,380 and $880) exactly, where the trained
baseline underbills both households by about $22. The couple
construction combines two people's income *before* counting, so the
$20 and $65 exclusions apply once per couple as the rule requires;
it is exact even on the boundary slices where one member has no
income at all — the slices where the trained baseline makes its
worst errors in the whole design (up to $48). Both constructions were then
replicated by the second writers (`experiments/second-writer-witness-extension/`):
GPT-5.6 Sol and Gemini 3.1 Pro compiled both rules exactly from
bare prompts, audit probes included — and Gemini's share-of-cost
arrays turned out parameter-for-parameter identical to the original
construction.

**Split invariance** (`experiments/ssi-couple-split-invariance/`).
The couple rule depends only on combined income totals, so every
reallocation of the same totals between the spouses is the same
household under the law. Measured across 90 totals classes x 121
compositions (plus 5,000 random classes), all three compiled couple
nets are invariant to within a trillionth of a dollar — while the
trained baseline's answer moves in 98.9% of classes (mean range
$3.64, worst $25.15), with the worst class decomposing into a
recognizable wrong rule: an exclusion applied per member instead
of once per couple. Swapping the spouses was one point in this
equivalence class; the within-class range is the general metric,
and it catches structural failure that average accuracy dilutes.

**The structural-property atlas** (`experiments/structural-property-atlas/`).
The split-invariance result generalized: across all six rules and
23 property families (monotonicity directions, output bounds,
plateaus, pinned bands, invariances), no compiled net from any
writer violates anything by more than 5e-13 — while every trained
baseline violates most of its rule's properties, including outputs
the program cannot produce at all (SNAP allotments outside
[0, $546] on 21.5% of probes; SSI's first-$20 property failed on
100% of probes).

**The repair bake-off** (`experiments/ssi-couple-repair-bakeoff/`).
Can training be fixed? Augmentation shrinks the couple net's
invariance defect ~5x and improves its accuracy on every measure;
a symmetry penalty at the weight tried is weaker on nearly every
measure — and the defect survives in ~94% of classes either way. A
sum-first architecture that cannot express the violation reaches
exactly zero invariance error, at an accuracy cost. If a property
matters enough to test for, it matters enough to build in.

**SNAP at member level** (`experiments/snap-household-composition/`,
replicated in `experiments/second-writer-snap-composition/`). The
allotment rule rebuilt with per-member inputs: the compiled
construction is exact and invariant, the trained baseline is
composition-dependent in 96.7% of classes (worst $18.32), and both
outside writers replicated the construction exactly — combining
the members' incomes in layer 1 unprompted, and matching the
oracle on the should-change control (earned vs unearned) as well
as the invariance.

**Canonical circuits** (`experiments/canonical-circuits/`). Under
hidden-unit permutation and positive scaling, independent writers
collapse to *one* canonical circuit in Medicaid liability and
share of cost; in SSI and the couple rule the two API writers share
a circuit leaner than the original's; Gemini's SNAP is a genuinely
different circuit for the same function. Most cross-writer
parameter difference is relabeling, and the method says so only
where it is true (the EITC negative control, whose functions
differ, does not collapse).

**CivBench narrative order, label side**
(`experiments/civbench-narrative-order/`). Verified mechanically
that all 156 scored CivBench fixtures carry household composition
only as aggregate scalar fields — there is no member order for a
deterministic label to depend on. The order risk lives in the
rendered prose prompts, i.e. in models, which is CivBench Build 1
work and stays gated there.

**Held-out test: a rule from regulatory text only**
(`experiments/ssa-pia-heldout/`). Every rule above was restated as
a cleaned formula before any writer saw it. This one was not: the
writers received the Social Security Administration's published
text for the 2027 primary insurance amount, verbatim, and nothing
else. GPT-5.6 Sol, Gemini 3.1 Pro, and the in-session control all
returned exact networks on the first attempt (max error 3.6e-12
over 120,003 points) — and all three returned the *same eight
nonzero parameters*: given prose percentages and bend points,
three models from three vendors converged on the identical
circuit. Full-suite verification of everything above: see
VERIFICATION.md.

**Formula recovery: the calculator as a black box** (exp116–exp119,
mirrored from [snap-qc-per-experiments](https://github.com/ctrimm/snap-qc-per-experiments),
where this line is canonical). Everything above *writes* weights from a
known formula. This line runs the other direction: given only query
access to a SNAP benefit calculator (the QC harness's calculator from
snap-qc-per-experiments, facts in and payment out, no code access),
adaptive probing recovered its exact parameters — allotments,
deduction tables, the $672 shelter cap, the floored 20% earned
staircase — and a planted stale-parameter pair was priced exactly by
diffing the recovered formula against the statute (−$34 / −$15). The
recovered formula compiled back into weights reproduces the
calculator at 100% (max error $0) once the staircase is computed in
code, the same boundary doctrine as the rounding result below.
Run against real FY2024 QC data (exp117/exp118), the statutory formula
reconstructs the QC minimodel at 99.9%, issued benefits match it far
less often (61.6% without earnings, 37.0% with), and the effective
parameters fit from issued benefits are the statutory ones ($198,
0.30, 20%): the error is case-level variance, not formula
configuration. exp119 recovers the Alaska/Hawaii/Guam/Virgin Islands
allotment tables from their own slices (Hawaii and Guam reconstruct
at 100.0%; Alaska's three schedules surface as clusters). Files:
`experiments/exp116_formula_recovery.*`, `exp116b_qc_audit.*`,
`exp117_real_slice_recovery.*`, `exp118_earnings_recovery.*`,
`exp119_territory_tables.*`; the generator under test is mirrored at
`src/synth/`. Full record in the chronicles and DECISION_LOG D17–D18.
Three follow-ups (exp120–exp122, DECISION_LOG D19; files
`experiments/exp120_cluster268_classification.*`,
`exp121_cluster54_identification.*`, `exp122_alaska_schedules.*`)
closed the line's open threads: the +$268 cluster is ~5/6 client-side unreported
earnings and ~1/6 agency failure to act, by the cases' own QC
variance codes; the +$54 minimodel cluster is the $180 homeless
shelter deduction, carried in a field (`HOMELESS_DED`) the
`FS*DED` set leaves at zero — with it, mainland minimodel
reconstruction is 99.98%; and Alaska's schedules are assignable
after all, via `BENMAX`, the case's own maximum-allotment
parameter (reconstruction 82.3% → 99.0% on the same frame), which
also confirms the blind-recovered Hawaii/Guam/Virgin Islands tables
at 100%.

**Five follow-ups on weights as the atomic unit** (2026-11-03,
each pre-registered in its PROTOCOL.md). *Trained-net recovery*
(`experiments/trained-net-recovery/`): the exp116 probing
instrument turned on a trained network itself — the trained SSI
baseline's slopes recover essentially exactly (−0.500, −1.000) and
its FBR within $8, but its earned-exclusion kink sits ~$30 early
with the statutory spill interaction flattened (kink movement $4
across unearned-income slices vs the statute's $20): it learned a
coherent *different* rule, and probing names the differing feature.
*Trained-net surgery* (`experiments/trained-net-surgery/`): the
trained couple net's split-dependence ($30.56 max) removed
*exactly* (→ $0.00) by a closed-form symmetrization of the first
layer — which also halved its max error ($9.24 → $4.53) — while a
generic sparse gradient edit of 32 weights managed −6% on the
violation and damaged mean accuracy 17×. *Certified cliff*
(`experiments/certified-cliff/`): at an eligibility cliff (jump
$546), constructed ramp nets pay exactly the $273 floor at every
band width, disagree with the rule only inside the certified
band, and pay for narrower bands in largest-weight like 1/w
(8 → 1,500 as the band goes $400 → $2); the trained control's
error is diffuse — off by > $1 across 87% of the domain, no band
to certify. *Weight-tamper audit*
(`experiments/weight-tamper-audit/`): of four audit instruments,
only the weight-anatomy audit catches a +$30 tent planted beyond
every behavioral probe's range (U > 1500); scoring, recovery
probing, and the property atlas each miss exactly the tampers
outside their coverage, and none false-positives on the clean
net. Long-form write-up with figures:
`experiments/weight-tamper-audit/WRITEUP.md`. *Composed witnesses* (`experiments/composed-witnesses/`): a
compiled countable-income net block-composed with a share-of-cost
head is **bit-identical** to the one-piece compiled net at zero
parameter overhead, and the FY2024 → FY2026 weight diff of the
compiled SNAP rule is exactly 5 entries carrying all three COLA
constants ($535→$546, $198→$209, $672→$744), each recoverable
from the diff alone — where two same-init trained nets' diff is
648 parameters that decode to nothing. Full record in each
folder's RESULTS.md and DECISION_LOG D20.

**The boundary: rounding** (`experiments/ssa-rounding-boundary/`).
The SSA text's last step — round the PIA down to the next lower
$0.10 — is a step function, and no continuous network can
implement it exactly: at a $0.10 jump, any candidate is at least
$0.05 wrong somewhere (proof in the experiment's protocol). The
best construction attains exactly that floor ($0.0500 max error,
99.2% of points within a cent); a trained network lands on the
same midpoint strategy in mid-range (mean error $0.0250) plus a
low-end offset; and the exact PIA network scored against the
rounded-by-law amount is off by up to $0.0999 — the rounding gap
itself. Continuous rules are exactly representable; discontinuous
ones only to half their jump. For a whole-dollar round the floor
is $0.50; for an eligibility gate, the jump is the whole decision.

## What all of this cost

**$2.48 in total API spend** ($0.653 for the second-writer
replication, $1.3469 for the weight-reading experiment, $0.2095
for the witness-extension replication, $0.1949 for the
member-level SNAP replication, $0.0831 for the held-out PIA test, all measured from the providers'
own usage accounting, reconciled against the account aggregate,
and itemized in those experiments' RESULTS.md). Everything else — the original
constructions, the trained baselines, the editing experiment — ran
in-session or on local CPU at no API cost. Each experiment's
RESULTS.md carries its own cost line.

## Guardrails

The plain-language section at the top is the short version. The
technical version: these experiments cover the *arithmetic cores* of
programs only. Categorical gates are step functions a continuous
network cannot represent exactly, and determinations belong in
deterministic code regardless. The defensible uses for a
weight-compiled network are independent witness implementations for
verification, and seam-hunting. Each protocol lists its
simplifications explicitly; the stipulated formulas are research
artifacts, not legal advice.

## Layout

```
experiments/
  snap-allotment/               FY2026 SNAP allotment, hh of 2
  eitc-credit/                  TY2025 EITC, 2 children (incl. ERRATUM.md)
  ssi-payment/                  2026 SSI federal payment, individual
  medicaid-patient-liability/   Medicaid post-eligibility liability
  ssi-stale-fbr-edit/           Repairing a stale constant by weight edit
  second-writer-openrouter/     GPT-5.6 Sol + Gemini 3.1 Pro replication
  weight-reading/               Fresh readers identify rules from weights alone
  medicaid-share-of-cost/       CA medically-needy share of cost (SSI counting reused)
  ssi-couple-payment/           2026 SSI couple payment, exclusions once per couple
  second-writer-witness-extension/  GPT + Gemini replicate the two witness rules
  ssi-couple-split-invariance/  Equivalent compositions must pay identically: they don't, for the trained net
  structural-property-atlas/    23 structural properties x every net: compiled violate none, trained violate most
  ssi-couple-repair-bakeoff/    Augmentation vs penalty vs architecture: only architecture reaches zero
  snap-household-composition/   SNAP with per-member inputs: invariance + should-change controls
  second-writer-snap-composition/  GPT + Gemini replicate the member-level SNAP rule
  canonical-circuits/           Writers' circuits under permutation/scaling gauge: mostly one circuit
  civbench-narrative-order/     Label-side audit: CivBench scenarios have no member order to permute
  trained-net-recovery/         Probing a trained net: its effective formula, kink ~$30 early, spill flattened
  trained-net-surgery/          Closed-form W1 symmetrization removes a trained net's split-dependence exactly
  certified-cliff/              Ramp-compiled eligibility cliff: $273 floor, certified band, price ~1/w in max weight
  weight-tamper-audit/          Which audit instruments catch weight-space tampers (only anatomy catches all)
  composed-witnesses/           Block composition is bit-identical; FY24->FY26 weight diff = 5 entries, fully decodable
```

Cross-experiment figures live in `docs/figures/` (generator: `docs/make_figures.py`); each experiment's RESULTS.md embeds its own figures from its `figures/` folder (generators: `docs/make_writeup_figures.py`, `docs/make_witness_figures.py`).

The formula-recovery line (exp116–exp119, see "Experiments and findings") lives as flat files in `experiments/` — code, chronicle, and results JSON per experiment — mirrored from ctrimm/snap-qc-per-experiments, its canonical home; the generator those experiments run against is mirrored under `src/synth/`.

Each experiment is self-contained: `PROTOCOL.md` (pre-registration),
`oracle.py` (the stipulated rule), `mlp.py` (generic forward pass),
`zero_shot/` (the weight constructions, with derivations, plus
`FREEZE.json` hash records where applicable), `train_baseline.py`,
`score.py`, `results/` (raw scores), `RESULTS.md` (verdict).

The first run (SNAP + EITC) was originally executed as one combined
experiment; its unmodified record — combined protocol, freeze hashes,
scorer, and results — is preserved in `archive/first-run-combined/`.
The per-experiment folders are the canonical layout going forward,
and their scorers reproduce the archived numbers exactly.

## Reproduce

```bash
cd experiments/snap-allotment
python3 score.py zero_shot     # needs numpy only
python3 score.py trained
# retrain a baseline (needs torch):
python3 train_baseline.py
```

## What's next

- The witness pattern now covers the arithmetic
  cores named for the CivBench label program — six rules, each with
  a trained baseline, and the two newest independently replicated
  by two other frontier models. Extending coverage to further rule
  families is future work.
- The formula-recovery line's three open threads are closed
  (exp120–exp122): the +$268 cluster is classified by variance
  code, the +$54 cluster is the homeless shelter deduction, and
  Alaska's schedules are assigned via BENMAX. One wobble remains,
  recorded in exp122: 8 large rural Alaska households sit +$3–4
  above the assigned reconstruction, consistent with a slightly
  higher rural standard deduction — pinning it needs a larger
  Alaska sample (another fiscal year's file), not more method.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
Copyright (c) 2026 Cory Trimm.

Use it, share it, adapt it, cross-reference it, cite it — for any
purpose — provided you give credit to **Cory Trimm**. Suggested
citation: Trimm, Cory, *Weight-Space Programming for the Social
Safety Net* (2026), <https://github.com/ctrimm/weight-space-programming>.
Full terms in [LICENSE](LICENSE).
