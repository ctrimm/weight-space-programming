# Protocol — SSI stale-FBR weight edit (pre-registered)

Written before the stale network was constructed and before any edit
was attempted. The edit script does not exist yet at the time of
writing. Gates below are fixed.

## The scenario

The CivBench audit of benefits-as-code (Nov 2026) found SSI gold
labels computed with the **2024** federal benefit rate ($943) while
labeled as 2026. Stale constants are how real benefits code breaks:
the rule structure stays right, one number goes out of date.

The paper this repo builds on reports the mirror-image feat in a
language model: locate the weights responsible for a defect and edit
just those. This experiment tests the same move on a benefits
formula, where the verdict is exact.

## Setup

1. Construct, by the same zero-shot method as `../ssi-payment/`, an
   SSI network that computes the stipulated SSI formula (see
   `../ssi-payment/PROTOCOL.md`) with FBR = **$943** (2024) instead
   of $994 (2026). Everything else identical: architecture
   2 -> 24 -> 24 -> 24 -> 1, same domain (E in [0, 3000],
   U in [0, 1500]), same evaluation design.
2. Freeze the stale weights (SHA-256) before the edit is designed.
3. Score the stale network two ways: against the stale oracle
   (FBR 943 — it should be exact; this validates the construction)
   and against the current oracle (FBR 994 — this measures the
   damage a stale constant does; predicted max error exactly $51).

## The edit

The surgeon (Muse Spark, in-session, same provenance as the other
experiments) is told only: *this network computes the SSI formula
with a stale FBR of $943; the 2026 FBR is $994; repair the network by
editing its weights. No retraining, no reconstruction: find where
the constant lives in the weights and change it there.*

Identification must be done **from the weight values themselves**
(scan the arrays; the stale constant should be findable as a literal
value), not by re-deriving the construction. The edit script records
every parameter it touches.

## Gates (fixed in advance)

- **G1 exact repair:** edited network vs current oracle (FBR 994):
  max abs error <= $1e-9 over the full evaluation set.
- **G2 surgical:** at most 8 of 1,297 parameters changed.
  (Prediction: exactly 2 — the layer-3 bias -0.943 and the output
  bias +0.943 — i.e. 0.154% of parameters.)
- **G3 confined effect:** the functional change (edited minus stale)
  must equal the theoretical delta of the FBR change everywhere:
  $51 where countable income C <= 943, (994 - C) where
  943 < C < 994, $0 where C >= 994; max deviation <= $1e-9.
- **G4 control:** the same value-scan applied to the *trained* SSI
  network (`../ssi-payment/trained/`) is run and reported. If the
  stale constant cannot even be located in the trained network, that
  contrast — compiled weights are auditable and repairable, trained
  weights are neither — is itself a result, not a failure of the
  experiment.

## Honest limits

- The surgeon also authored the original construction method (in a
  sibling experiment), so "find the constant" is not a blind test of
  an unknown network. It tests locatability and edit precision in a
  compiled network, plus the trained-network control.
- A real stale-constant bug in production code would be fixed in
  code, not weights. The claim tested here is narrow: *if* a formula
  is carried in weights at all, its parameters stay legible and
  repairable in a way trained parameters do not.
