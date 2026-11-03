# Decision log — weight-space programming for the social safety net

## D6 (2026-11-02): Repo restructure; SSI and Medicaid experiments added
Decision: the work becomes a standalone repo with one self-contained
folder per experiment under experiments/. The original combined
SNAP+EITC run record moves unmodified to archive/first-run-combined/;
per-experiment scorers were verified to reproduce its numbers exactly.
SSI target: the federal payment formula (2026 FBR $994, individual),
stipulated WITH the $20 general-exclusion spill to earned income —
the CFR-correct version, and pointedly not the benefits-as-code
oracle's version (its exclusion never spills; CivBench SSI audit,
2026-11-01). The spill makes the earned-exclusion kink plane depend
on unearned income, which is the structural step up from SNAP.
Medicaid target: post-eligibility patient liability (42 CFR 435.725)
with the community-spouse MMMNA allowance — chosen because MAGI
Medicaid is mostly step thresholds (not compilable exactly, would
measure the wrong thing) and the medically-needy share of cost is a
single hinge once income is counted (would test nothing new). PETI
has a genuine floor-inside-a-floor structure across three inputs.
MMMNA stipulated at the federal minimum $2,643.75 (Jul 2025–Jun
2026); PNA $50 labeled a stipulation, not a federal constant.

## D1 (2026-11-02): Test the arithmetic core, not full eligibility
Decision: the target functions are the EITC credit formula (TY2025,
single/HOH, 2 children, AGI = earned income) and the SNAP allotment
formula (FY2026, household of 2, earned income only), each stipulated
in full in PROTOCOL.md.
Evidence/reasoning: both are continuous piecewise-linear, so an exact
weight construction provably exists and "exact vs approximate" is a
clean readout. Full eligibility (categorical gates, BBCE) contains
step discontinuities no continuous network can represent exactly, and
the CivBench bac fixtures' gold is eligibility + citations, not
amounts — using them would conflate two different questions.
Consequence: results say nothing about automating eligibility, which
stays deterministic code regardless.

## D2 (2026-11-02): Stipulate constants in the prompt; verify them against primary-ish sources first
Decision: FY2026 SNAP constants from USDA FNS COLA figures (max
allotment hh2 $546, standard deduction $209, shelter cap $744, 20%
earned deduction, benefit = max − 30% of net); TY2025 EITC constants
from 26 USC 32(b)(1) rates + Rev. Proc. 2024-40 amounts via NY Tax
Dept's parameter page.
Consequence: the test measures compilation, not recall — same as the
paper, which gave Astra the rules and source code.

## D3 (2026-11-02): Freeze zero-shot weights by SHA-256 before writing the scorer
Decision: FREEZE.json hashes PROTOCOL.md, oracle.py, mlp.py and both
zero-shot .npz files before score.py existed.
Reasoning: the claim "no training, no fitting, no peeking" is only
worth anything if the order is auditable. Later attempts are labeled
and never overwrite attempt 1.

## D4 (2026-11-02): EITC attempt 2 changes one number, and the protocol's consistency check was wrong
Finding: PROTOCOL.md claimed 7,152 / 0.2106 = 33,960.0 exactly. It is
33,960.114; the published completed-phaseout amount is rounded to the
nearest $10, making the stipulated oracle discontinuous by $0.024 at
$57,310. Attempt 1 (hinge at $57,310) inherited a $0.024 tail error on
every point past the endpoint. Attempt 2 moves only the final hinge to
the true zero ($57,310.114): max error stays $0.024 but is confined to
the single seam point, where it is provably irreducible for a
continuous network. Full correction in ERRATUM.md.
Lesson recorded for CivBench work: published table constants may not
multiply out against their own statutory rates; treating them as exact
manufactures seam errors.

## D5 (2026-11-02): Retrain the baseline longer rather than report an underfit comparator
Finding: baseline v1 (EITC 3,000 steps / SNAP 5,000 steps of Adam,
batch 8,192) left the EITC net visibly underfit (train MSE 2.6e-2 $k²;
−$571 predicted at $70,000 where the truth is $0). Export fidelity was
checked first (numpy vs torch forward agree to 4e-4), so the gap was
optimization, not plumbing.
Decision: baseline v2 (EITC 20,000 steps, SNAP 15,000 steps, otherwise
identical) is the comparator in RESULTS.md; v1 weights and scores are
preserved in trained_v1/ and results/trained_v1_scores.json.
Reasoning: a weight-writing win over a strawman baseline proves
nothing; the interesting claim survives a competent baseline or it
does not survive.

## D7 — 2026-11-02 (evening): License CC BY 4.0; editing experiment run

- License chosen by Cory: free to use, cross-reference, and cite,
  with credit to Cory Trimm — implemented as CC BY 4.0 (LICENSE file
  with his copyright line and a suggested citation; README section).
- Editing experiment commissioned by Cory the same evening and run
  under a pre-registered protocol (PROTOCOL.md written and stale
  weights SHA-256-frozen before the edit script existed). Findings:
  a stale-FBR SSI network is exact for the wrong year (max $51.00
  error vs the 2026 rule, mean $17.65); the FBR lives in exactly 2
  of 1,297 parameters as a literal; editing those 2 restores
  exactness (2.3e-13) with the functional delta matching theory to
  2.9e-13; the edited file is array-identical to the independently
  constructed 2026 network. Control: the trained SSI network
  contains no parameter equal to either FBR (nearest 0.029/0.080
  away) — trained constants are not locatable, hence not surgically
  repairable.
- Second weight writer via OpenRouter scoped but NOT run: live
  account balance $24.04 (read from the OpenRouter API, no model
  calls made). Estimate presented to Cory anchored in dashboard
  per-run actuals (Sonnet $0.063/run, Opus ~$0.21/run, Oct 2026);
  pilot-first design (2 non-Anthropic frontier models x EITC) then
  full sweep, proposed hard cap $3. Awaiting his explicit yes, per
  the standing OpenRouter rule.

## D8 — 2026-11-02 (night): Second-writer replication via OpenRouter

- Cory authorized the pilot with a **$10 hard cap** (raised from the
  $3 proposed). Pre-registered protocol; one stateless call per
  model x experiment, one syntax-repair retry, no score feedback.
  Actual total spend **$0.653** (account-verified; $0.6336 across 10
  logged calls + ~$0.019 from a killed first launch whose in-flight
  call billed server-side). Well under cap; per-call costs came in
  ~4-10x below the dashboard-anchored estimate range.
- Results: GPT-5.6 Sol and Gemini 3.1 Pro both PASS SNAP, SSI, and
  Medicaid exactly. EITC split: Gemini reproduced the original
  attempt 1 functionally identically (max $0.024, Partial); GPT-5.6
  Sol declined twice, correctly deriving the $0.024 discontinuity
  at $57,310 from the constants and refusing to submit inexact
  weights — recorded as FAIL (no submission) per the gate, with the
  refusal preserved verbatim in raw/. No third call was made.
- Conclusion recorded: cross-vendor replication holds; the witness
  use case now has a documented behavioral fork (build-to-table vs
  refuse-on-inconsistency) worth surfacing wherever the results are
  presented.
- README hygiene (Cory's catch): the editing experiment's earlier
  README edits had silently failed to match; now verified in the
  findings table (row 5) + findings paragraph, and removed from
  What's next.

## D9 — 2026-11-02 (night): Weight-reading experiment

- Cory approved the full design (three readers cross-reading,
  defect net included). Run under pre-registered protocol + rubric
  with claims returned as executable expressions and graded
  behaviorally against the mystery networks themselves.
- Arm A: all 11 compiled-net reads behaviorally exact (float
  noise), all constants recovered, by all readers. Defect net
  flagged by BOTH blind readers, GPT quantifying the $51/mo harm.
  Spend $0.9394.
- Control (trained SSI): family identified, constants NOT
  recoverable (GPT read FBR $1,000, Gemini $914; both confident) —
  trained weights do not contain their constants in readable form.
- Arm B (25 behavior samples, control reads only): no rescue.
  GPT max err flat ($11.35 -> $11.91), Gemini far worse ($89.54 ->
  $414.46, sample overfit). The paper's behavior-rescue pattern did
  not reproduce; consistent with there being no exact rule inside
  a trained net for samples to reveal. Spend $0.4075; experiment
  total $1.3469 vs $6 cap.
- Secondary finding for the write-up: readers' recovered constants
  were mechanically perfect while their prose context was
  sometimes wrong-year (Gemini doubted the correct $994 against
  2024's $943). Grade numbers, not narration.
