# Protocol (DRAFT) — Post-training footprints: do SFT and GRPO reach the same benefits answers through the same weights?

**Status: DRAFT, written 2026-11-07 at Cory's direction to think ahead.
Not yet registered. No runs are authorized by this document; it becomes
the pre-registration when Cory approves it, and any amendment after
that point follows the tamper-audit convention (amend before running,
disclose in RESULTS).**

## Why this experiment exists

The weight-tamper audit mapped what *audits* can catch in a compiled
net. This experiment turns the same instruments on *legitimate*
post-training. Cory has SFT/QLoRA and GRPO post-trained checkpoints for
benefits-program tasks, **both starting from the same base model**
(confirmed by Cory 2026-11-07). Identical starting point, two training
methods, one task family: any difference in where the updates live, or
in which updates are causally responsible for the improved answers, is
attributable to the method rather than the base.

The question, in the spine's terms: **when post-training improves the
score, did it fix the mechanism or game the check?** A behavioral eval
cannot distinguish those. A weight-space audit can at least ask.

Two results from this repo set the priors — and warn against assuming
the answer:

- *Trained-net surgery:* a sparse gradient edit to the 32
  probe-implicated weights barely helped (violation −6%, accuracy 17×
  worse); a closed-form structural edit removed the violation exactly.
  Updates aimed at "the weights that look responsible" are not
  guaranteed to be the weights that are responsible.
- *Trained-net recovery (N11 control):* blind readers could not find
  the benefit-rate constant in a trained net's weights (reads of $1,000
  and $914 against a ~$999.97 output ceiling). In trained weights, a
  rule constant may have no decodable site at all — so this protocol
  asks about causal load-bearing (ablation), not decodability.

## Artifacts required (inputs, not yet in the workspace)

- Base model identifier and weights (the shared starting point).
- SFT artifact: QLoRA adapter files (expected small; copy into this
  repo's experiment folder or a scratch dir outside git if large).
- GRPO artifact: adapter files if GRPO was LoRA-based, else the full
  checkpoint (analysis then runs where the checkpoint lives — the
  Mac Studio — or on copied state dicts).
- Task/eval sets, frozen before any analysis: the Mac Studio
  QC-variance eval (1,400 frozen items) and the CalFresh MCQ benchmark
  from the SNAP QC post-training line, plus a held-out slice of
  programs/states absent from both training runs.
- Training-task provenance for both runs (same tasks? same data?) —
  recorded as a covariate, not assumed.

## Phase 0 — Footprint anatomy (weight math only; $0; no GPU)

For each method's delta ΔW (adapter delta, or checkpoint − base):

- Per-module and per-layer ΔW Frobenius norms; share of total Δ norm
  in the top 1 / 5 / 10 modules (concentration).
- Effective rank of each ΔW (singular-value spectrum; rank at 95%
  energy), for LoRA deltas where ΔW = BA this is bounded by the
  adapter rank — the measurement is how much of that budget is used.
- Cross-method overlap per module: cosine similarity between the SFT
  and GRPO ΔW, and principal angles between their top singular
  subspaces.

**Predictions, to be registered on approval (directional, falsifiable):**

- P1: GRPO's delta is more concentrated than SFT's (higher top-5
  module share) and lower in effective rank — RL post-training is
  expected to move less of the network to reach its gains.
- P2: Cross-method ΔW cosine similarity is low (< 0.3) in most modules
  despite similar task scores — the methods reach the behavior by
  different routes.
- P3 (null-friendly): if P1 and P2 both fail — footprints statistically
  indistinguishable on the registered metrics — that is the result,
  and Phase 1 narrows to the behavioral-equivalence question only.

## Phase 1 — Causal load-bearing (needs the checkpoints' machine)

- Module-group ablation: zero the delta one module-group at a time;
  measure the change on (a) items the post-training fixed, (b) items
  it did not change, (c) the held-out slice. Output: an ablation
  profile per method — which parts of the delta earn the score.
- Agreement analysis: on items where SFT and GRPO agree behaviorally,
  do their ablation profiles correlate? Agreement in outputs with
  disagreement in mechanism is the "same score, different machine"
  finding; disagreement in outputs localizes where to dig.
- Collateral check: ablations that *improve* held-out performance
  identify delta components that game the training distribution at
  the expense of generalization — the trained-net counterpart of the
  tamper audit's tents.

## Kill / pivot criteria

- If neither artifact can be obtained in analyzable form (adapter or
  state dict), the experiment stops at a one-page note saying so; no
  substitute artifacts.
- If Phase 0 kills P1/P2 (per P3), Phase 1 runs only if the behavioral
  equivalence question still looks live after Phase 0's write-up;
  decision is Cory's, recorded in the repo DECISION_LOG.

## Outputs

- `results/footprints.json` (Phase 0 measurements), ablation tables
  (Phase 1), RESULTS.md in this folder, README experiment-index line
  updated in the same pass, DECISION_LOG entry for the go/no-go after
  Phase 0.

## Open inputs from Cory

1. Was GRPO LoRA-based (adapter files) or a full fine-tune?
2. Were the SFT and GRPO training tasks/data the same set?
3. How do the artifacts get here — adapters copied into the workspace,
  or analysis run on the Mac Studio?
