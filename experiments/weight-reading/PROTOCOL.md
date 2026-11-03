# Protocol — weight reading (pre-registered)

Written 2026-11-02 before any reader saw any artifact. Cory approved
the design (all three readers, cross-reading, defect net included)
and the ~$1-3 API estimate; the runner enforces a hard abort at $6
measured spend regardless.

## Question

The other experiments show models can *write* a benefits formula
into weights, and that a compiled constant can be found by scanning
for it. This experiment tests *reading*: hand a model a network it
did not write — only the weight arrays, the architecture, and what
the inputs/output mean — and ask what rule it computes. This is the
audit use case: an artifact arrives with no code, and someone has to
say what is inside it.

The source paper's own attempt at a pure weight-read failed (it
needed behavior traces as well). Arm B below measures that gap
instead of assuming it away.

## Artifacts (anonymized, hash-frozen before any read)

Eleven networks, neutral codes, copied from earlier experiments:

- SNAP, SSI, Medicaid — each as compiled by three different writers
  (Muse Spark in-session; GPT-5.6 Sol; Gemini 3.1 Pro).
- **Defect net:** the stale-FBR SSI network from
  `../ssi-stale-fbr-edit/` (2024 rate $943 compiled in).
- **Control:** the *trained* SSI network from `../ssi-payment/`
  (gradient-trained, constants smeared). Expected to resist reading;
  included so the experiment can find the boundary honestly.

## Readers and assignment (cross-reading; nobody reads their own)

- GPT-5.6 Sol reads: Gemini's SNAP, Muse's SSI, Gemini's Medicaid,
  defect, control.
- Gemini 3.1 Pro reads: Muse's SNAP, GPT's SSI, Muse's Medicaid,
  defect, control.
- Muse Spark reads: GPT's SNAP, Gemini's SSI, GPT's Medicaid —
  in-session, by array analysis. **Disclosed non-blind:** Muse
  authored or scored every artifact in this repo and knows the
  ground truths. Its arm demonstrates the reading *procedure*;
  the API arms are the blind test.

Each reader is told only: the forward-pass convention, the array
shapes, what each input and the output mean (with domains and
scaling), and the dense weight values (rounded to 6 decimal places;
compiled constants are exact at <= 5 places, so nothing material is
lost — the trained control is the only artifact where rounding is
an approximation, and identification there does not turn on the
6th decimal). The defect read additionally provides a current-law
sheet (SSI 2026: FBR $994 individual; $20 general exclusion with
spill to earned; $65 earned exclusion; one-half disregard) and asks
for an audit: recover the artifact's constants and flag any that
disagree with current law.

## Answer format (fixed)

One JSON object: `program` (name the rule), `formula_text` (the
formula in words/notation), `constants` (every defining constant
recovered, named), `formula_python` (ONE Python expression over the
named inputs using only numbers, max, min, abs, + - * / and
parentheses — this makes the claim mechanically checkable),
`confidence` (0-1), `anomalies` (list).

## Rubric (fixed in advance, per read)

- **R1 family (0/1):** the named program/rule matches ground truth.
- **R2 behavioral exactness:** the `formula_python` claim is
  implemented and diffed against the mystery network on that rule's
  standard evaluation design (the original experiments' point sets).
  Max abs error <= $0.01 -> 1; <= $1 -> 0.5; else 0. For the defect
  net, R2 is graded against the *stale* oracle (does the claim
  describe the artifact as it actually is).
- **R3 constants:** fraction of the rule's defining constants that
  appear exactly among the numeric literals of the claim
  (SNAP: 546, 209, 0.8, 0.5, 744, 0.3 · SSI: 994, 20, 65, 0.5 ·
  defect SSI: 943 in place of 994 · Medicaid: 2643.75, 50).
- **R4 defect only (0/1):** the anomalies list flags that the
  artifact's benefit rate ($943) is not the current-law rate ($994).

## Arm B — behavior rescue

For any API read scoring R2 < 1, one follow-up call: the reader
gets 25 input/output samples from the mystery network (corners,
kink-adjacent probes, seeded randoms) plus its own Arm A answer,
and may revise. Arm B claims are scored with the same rubric and
reported separately. The A-to-B movement is the measured value of
behavior data, mirroring the paper's weights-alone failure.

## Cost discipline

API reads via OpenRouter (same harness pattern as
`../second-writer-openrouter/`): actual per-call cost from usage
accounting, logged after every call, abort before any call once
cumulative spend reaches $6. Raw replies preserved in `raw/`.
