# Results — CivBench narrative-order test, label side

**Verdict: at the label level there is nothing to permute — and
that is now verified mechanically, not assumed.** The deterministic
side of CivBench is order-invariant by representation. The order
risk lives entirely in the prose layer, which is the model side,
and that side remains gated CivBench Build 1 work.

## The audit

All 156 scored bac-* fixtures in the CivBench AISI-shift staging
repo were audited (read-only):

- Every scenario field in every fixture is a named scalar
  (int / str / bool / float). Household composition reaches the
  label computation only as aggregates — `household_size`, combined
  income fields — never as a member list.
- Exactly 4 fixtures have any non-scalar value:
  - `bac-eitc-027`: a `dependents` list — with **one** element
    (a 20-year-old child). Permutation is vacuous.
  - `bac-wic-011`: a `receives_benefits` list — with **one**
    element (`snap`). Permutation is vacuous.
  - `bac-ssi-031`, `bac-wic-015`: a `meta` dict — provenance
    metadata, not a label input.
- Scenario content is stable under key-order permutation for all
  156 fixtures (the label code reads named fields; the check
  confirms no fixture smuggles order in through the back door).

So a member-order permutation test against the deterministic
labels would pass trivially on 156 of 156 fixtures — not because
the labels were tested and held, but because the representation
gives order nowhere to act. Recording that distinction matters:
"labels are permutation-invariant" would overclaim a test that was
never really run.

## Where the order risk actually lives

The fixture scenarios are rendered into prose prompts ("Household
size is 2. Other facts: receives ss; age: 66; state: CA; ..."), and
prose *does* have an order: facts arrive in a sequence, members in
real intake narratives arrive in a sequence, and a caseworker model
reads them in that sequence. The narrative-order question is
therefore a question about models reading permuted prose against
fixed labels — the exact analogue, one layer up, of the trained
couple net failing split invariance one layer down.

That test is specified but not run here: permute fact order in the
rendered prompts of the witnessed gold core, run the caseworker
models, and report answer-flip rate per permutation class. It
belongs to CivBench Build 1, which is gated on its own spend
estimate and Cory's approval and is scheduled against the gold
core in December. Nothing in this queue's $5 authorization was
spent on it, by design.

## Cost

**$0.** Read-only local audit.
