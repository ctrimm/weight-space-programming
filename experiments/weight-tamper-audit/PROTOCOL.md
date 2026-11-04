# Protocol — which audit instruments catch a weight-space tamper? (pre-registered)

Written 2026-11-03, before any tampering, probing, or scoring for
this experiment. No API calls; $0 spend. All computation local CPU.
The blind-reading instrument is operationalized mechanically (see
below); no LLM readers are used in this experiment.

## Why this experiment exists

exp116's Arm B planted defects in a *formula* and showed black-box
probing prices them exactly. But a compiled network's threat model
is not limited to formula-shaped defects. Weights can be edited
directly — that is this repo's founding operation — so an adversary
(or a bug) can plant a defect that **has no formula counterpart**: a
payment bump that exists only in a thin slice of the input space.
The verification story told about compiled witnesses is only as
strong as the audit instruments that would catch such a tamper.
This experiment maps each instrument's blind class, on purpose,
before anyone needs the answer.

## Setup

Base artifact: the compiled SSI individual net
(`../ssi-payment/zero_shot/`), exact vs the oracle (max error
~1e-13 on the standard grid).

Three tampers, applied to copies of the compiled weights:

- **T-global (formula-shaped control)**: FBR constant raised by $25
  in the weights — the stale-FBR edit in reverse. Every instrument
  should catch this; it is the sanity arm.
- **T-local-near (weight-native)**: an appended gadget whose
  activation is a tent over U ∈ [1200, 1400], E ∈ [0, 50], raising
  the payment by up to +$30 at the tent's peak and $0 outside the
  region. No parameter of the recovered formula moves.
- **T-local-far (weight-native)**: the same tent over
  U ∈ [1800, 2000], E ∈ [0, 50].

### Amendment (2026-11-03, before any tampering or scoring)

The original draft had one local tamper (at U ∈ [1800, 2000]) and
predicted the uniform scoring grid would catch it. Grounding the
instruments in the repo's actual code before running showed that
prediction was wrong about the grid: `../ssi-payment/score.py`'s
grid caps at U = 1500, the atlas's SSI monotonicity probes cap at
U = 1450 (+50 step), and the trained baseline's sampling
distribution is uniform on U ∈ [0, 1500]. Every behavioral
instrument in this repo lives inside U ≤ 1500. The design therefore
uses two local tampers — near (inside behavioral coverage) and far
(beyond all of it) — and the prediction table below is the amended
one. The amendment is recorded here, before any results exist,
rather than letting the original table be silently reinterpreted
after the fact.

Four instruments, each a concrete procedure with a stated detection
criterion:

1. **Standard scoring**: `../ssi-payment/score.py`'s grid
   (E ∈ [0, 3000], U ∈ [0, 1500]). Detects if max error vs the
   oracle > $0.01 on the grid.
2. **Parameter-recovery probing**: the trained-net-recovery
   experiment's probe suite (FBR at origin, kinks, windowed slopes —
   all at U ≤ 800). Detects if any recovered parameter moves by
   > $1 / > 0.01.
3. **Property atlas (SSI subset, as implemented in
   `../structural-property-atlas/atlas.py`)**: monotonicity in E
   (grid points, U ≤ 1500) and in U (points with U ≤ 1450, +50
   step), first-$20-free at E = 0, bounds [0, 994]. Detects if any
   probed property is violated by > $0.01.
4. **Weight-anatomy audit**: decode the weights against the known
   construction anatomy (the mechanical form of blind reading for a
   compiled net): unit counts per layer must match the frozen
   construction, and every unit's weights must match the
   construction's gadget patterns. Detects if any unit or parameter
   is non-anatomical.

## Predictions, registered in advance (as amended)

| Instrument | T-global | T-local-near | T-local-far |
|---|---|---|---|
| Standard scoring | detects | detects (grid crosses U 1200–1400) | **misses** (grid ends at U = 1500) |
| Parameter-recovery probing | detects | **misses** | **misses** |
| Property atlas (as implemented) | detects | detects (mono-U pairs straddle the tent at low E) | **misses** (probes end at U = 1450) |
| Weight-anatomy audit | detects | detects | detects |

The predicted lesson, registered now so it cannot be retrofitted:
**behavioral instruments certify the function only where they look;
the tamper that matters is the one placed where they don't.** The
anatomy audit is the only instrument in the set whose coverage is
the whole artifact rather than a sample of its behavior — which is
an argument for weight-level auditing of compiled witnesses that the
error-score literature does not give you.

## Scope note

Detection criteria are mechanical and stated in advance; "detects"
means the criterion fires, not that a human would necessarily
investigate. The atlas arm uses the atlas's standard probe lines —
adding region-targeted lines after seeing the tamper would void the
test, and is recorded as future work instead (an atlas whose lines
are drawn from the *weights*, i.e. anatomy-guided probing, is the
obvious synthesis this experiment is designed to motivate or kill).
