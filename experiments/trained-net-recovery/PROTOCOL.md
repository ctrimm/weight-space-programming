# Protocol — recovering a trained network's implicit formula (pre-registered)

Written 2026-11-03, before any probing, fitting, or scoring for this
experiment. No API calls; $0 spend. All computation local CPU.

## Why this experiment exists

The formula-recovery line (exp116–exp122, mirrored in this repo)
recovered exact parameters from two black boxes: a deterministic
benefit calculator (query probing) and the FY2024 QC dataset
(reconstruction and fitting). One black box was never put under the
same instrument: **a trained network itself.** The trained baselines
in this repo are scored by error (mean/max vs the oracle) and by
structural properties (the atlas), but nobody has asked the recovery
question of one: *what formula did it actually learn?*

The SSI individual rule is the right subject because its constants
are the crispest in the project — FBR $994, general exclusion $20,
earned exclusion $65 plus the unused-general-exclusion spill, earned
slope 1/2 — and because this repo holds both a compiled net that
encodes them exactly (`../ssi-payment/zero_shot/`) and a trained
baseline that approximates them (`../ssi-payment/trained/`).

## Method

The instrument is exp116's, adapted to a two-input function: treat
the network as an oracle (inputs in, payment out, no weight access)
and recover parameters from adaptive probes.

- **FBR_eff**: output at E = 0, U = 0.
- **Unearned kink** (the $20 exclusion): on the slice E = 0, find the
  U where the payment's slope vs U leaves 0 and becomes −1 (scan +
  bisection on finite differences).
- **Earned kink** (the $65 + spill): on slices U ∈ {0, 10, 20, 50},
  find the E where the slope vs E leaves 0; the statutory prediction
  is 85 − U for U ≤ 20 (spill) and 65 for U ≥ 20.
- **Slopes by region**: earned slope measured in windows
  E ∈ [200, 400], [800, 1000], [1400, 1600] at U ∈ {0, 100, 500};
  unearned slope in windows U ∈ [100, 300], [600, 800] at E = 0.
  A perfectly learned rule returns −0.5 and −1.0 in every window.
- **Drift summary**: the spread (max − min) of each recovered
  parameter across regions, for the trained net vs the compiled net
  run through the identical probing code (methods control).

Scoring grids and the oracle are `../ssi-payment/oracle.py`'s. The
trained net is evaluated through the repo's `mlp.py` forward pass
with the SSI input/output scaling from `../ssi-payment/`.

## Predictions, registered in advance

1. Compiled control: every parameter recovered to < $1 (kinks) and
   < 0.01 (slopes); drift ≈ 0. If this fails, the instrument is
   broken and the experiment stops here.
2. Trained net, point recoveries at the origin will look deceptively
   good: FBR_eff within $15 of $994, unearned kink within $10 of $20.
3. Trained net, regional drift will be large and specific: earned
   slope across windows will spread by > 0.05 (statutory: constant
   0.5), and the earned kink will move by > $5 across the U slices
   (statutory: exactly 85 − U / 65). I.e., the trained net does not
   have *a* formula with slightly wrong constants; it has a different
   local formula in different regions.
4. The unearned slope far out (U ∈ [600, 800]) will be closer to −1
   than the earned slope is to −0.5 in the far windows: one-input
   structure is learned better than the interaction.

## What would change my mind

If the trained net's recovered parameters are regionally stable
(drift within the compiled control's tolerance), then the error
scores undersold it: it learned the rule's actual parameter
structure, and the "trained nets learn a different function" claim
in this repo's narrative needs qualifying.
