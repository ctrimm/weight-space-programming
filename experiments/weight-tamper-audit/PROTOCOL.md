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

Two tampers, applied to copies of the compiled weights:

- **T-global (formula-shaped control)**: FBR constant raised by $25
  in the weights — the stale-FBR edit in reverse. Every instrument
  should catch this; it is the sanity arm.
- **T-local (weight-native)**: one hidden unit appended whose
  activation is a tent over the region U ∈ [1800, 2000], E ∈ [0, 50]
  (high unearned income, no earnings — a rare profile), with output
  weight set so the payment rises by up to +$30 at the tent's peak
  and $0 outside the region. The function changes nowhere else; no
  parameter of the recovered formula moves.

Four instruments, each a concrete procedure with a stated detection
criterion:

1. **Standard scoring**: the repo's accuracy grids (uniform grid +
   the sampler-style case distribution used by `../ssi-payment/score.py`).
   Detects if max error vs oracle > $0.01 on either grid.
2. **Parameter-recovery probing**: the trained-net-recovery
   experiment's probe suite (FBR at origin, kinks, windowed slopes).
   Detects if any recovered parameter moves by > $1 / > 0.01.
3. **Property atlas (SSI subset)**: monotonicity in E and in U along
   probe lines, payment ≤ FBR, payment = FBR at (0,0), non-negativity.
   Detects if any probed property is violated by > $0.01.
4. **Weight-anatomy audit**: decode the weights against the known
   construction anatomy (the mechanical form of blind reading for a
   compiled net): every hidden unit's implied rule fragment must fit
   the formula's grammar, and the parameter count / unit count must
   match the frozen construction. Detects if any unit or parameter
   is non-anatomical.

## Predictions, registered in advance

| Instrument | T-global | T-local |
|---|---|---|
| Standard scoring (uniform) | detects | detects (grid crosses the region) |
| Standard scoring (sampler) | detects | **misses** (region is off-distribution) |
| Parameter-recovery probing | detects | **misses** (no global parameter moves) |
| Property atlas | detects | detects **only if** a monotonicity line crosses the region; the standard atlas lines do not, so as-run it **misses** |
| Weight-anatomy audit | detects | detects |

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
