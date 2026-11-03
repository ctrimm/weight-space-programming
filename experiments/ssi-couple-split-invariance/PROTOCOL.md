# Protocol — SSI couple split invariance (pre-registered)

Written 2026-11-02 before any analysis was run. Cory commissioned
this experiment after a preliminary swap check showed the trained
couple baseline changes its answer when the two spouses are swapped,
while the compiled nets do not.

## Background and question

Under the stipulated SSI couple rule (see
`../ssi-couple-payment/PROTOCOL.md`), the payment depends only on
the combined totals E = E1 + E2 and U = U1 + U2. The exclusions
apply once, to the combined amounts. It follows that *every*
reallocation of the same totals between the two members — one
earner or two, all unearned income with one spouse or split evenly,
and everything between — is the same household under the rule and
must produce the same payment.

Swapping the members is one point in that equivalence class. This
experiment measures the whole class:

> For each couple network in this repo, how much does its output
> vary across compositions that share the same combined totals?

## Arms

Four existing, frozen weight files; nothing is retrained or edited:

- **Original** — Muse Spark's in-session construction
  (`../ssi-couple-payment/zero_shot/couple_weights.npz`)
- **GPT-5.6 Sol** and **Gemini 3.1 Pro** — the second-writer
  replications (`../second-writer-witness-extension/weights/`)
- **Trained baseline** — the Adam-trained couple net
  (`../ssi-couple-payment/trained/couple_weights.npz`)
- **Oracle** — the stipulated rule itself, as a sanity control: its
  within-class range must be exactly zero up to float arithmetic.

## Design

- **Totals grid:** combined earned E in {0, 50, 100, 250, 500,
  1000, 2000, 3000, 4000} x combined unearned U in {0, 10, 20, 40,
  100, 250, 500, 1000, 2000, 3000} — 90 totals classes spanning the
  stipulated domain (E1, E2 <= 2000; U1, U2 <= 1500), including the
  exclusion thresholds (U = 20) and the payment phase-out region.
- **Splits:** for each totals class, earned share alpha and
  unearned share beta each range over {0, 0.01, 0.05, 0.1, 0.25,
  0.5, 0.75, 0.9, 0.95, 0.99, 1} — 121 compositions per class,
  10,890 evaluations per arm. E1 = alpha * E, U1 = beta * U, and
  member 2 takes the remainder.
- **Random arm:** 5,000 seeded random totals classes (E uniform in
  [0, 4000], U uniform in [0, 3000], seed 20261102), 8 seeded
  random splits each — 40,000 evaluations per arm — to confirm the
  grid is not cherry-picked.

## Metrics (per arm)

- **Invariance range** per class: max output − min output across
  the class's compositions. Report max, mean, and the share of
  classes with range > $0.01 and > $1, for grid and random arms.
- **Worst class:** the totals class with the largest range, with
  its min/max outputs and the splits that produce them.
- **Accuracy, kept separate:** each arm's error against the oracle
  across all grid compositions (max / mean / % within $0.01). A net
  can be invariant but wrong, or accurate on average but
  non-invariant; the two failures are reported independently.

## Pre-registered expectations

- The three compiled arms: invariance range at float noise
  (< 1e-9 dollars) in every class.
- The trained arm: materially positive ranges, concentrated near
  the exclusion thresholds and the payment phase-out boundary.
- The oracle control: range exactly 0 up to float arithmetic; any
  larger value invalidates the harness, not the arms.

## Cost

$0. Local numpy only; no API calls, no training.
