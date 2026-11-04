# Results — recovering a trained network's implicit formula

**Verdict: the instrument works, and two of my four predictions
fail in a way that produces the experiment's actual finding.** The
trained SSI baseline did not learn the statutory formula with
regional drift, as predicted. It learned a *different, internally
consistent formula* — slopes essentially exact, FBR $8 high, and the
earned-income exclusion kink in the wrong place by up to $32 with
its statutory dependence on unearned income nearly erased.
Black-box probing localizes the training failure to one structural
feature — the exclusion-spill interaction — which neither error
scores nor the property atlas had isolated.

## The instrument (validated on the compiled control)

Identical probing code (FBR at origin, kinks by max curvature,
windowed least-squares slopes) run on both nets. The compiled net
returns the statute exactly: FBR 994.00, unearned kink 20, earned
kinks 85 / 75 / 65 / 65 at U = 0 / 10 / 20 / 50, slope spreads
0.0000. Prediction 1 holds; the instrument is not the story.

## The trained net's effective formula

| Parameter | Statute | Trained net (recovered) |
|---|---|---|
| FBR | $994 | **$1,001.91** (+$7.91) |
| Unearned kink ($20 exclusion) | $20 | $18 |
| Earned kink at U = 0 | $85 | **$53** (−$32) |
| Earned kink at U = 10 | $75 | $54 (−$21) |
| Earned kink at U = 20 | $65 | $55 (−$10) |
| Earned kink at U = 50 | $65 | $57 (−$8) |
| Earned kink movement across U slices | 20 (85 → 65) | **4** (53 → 57) |
| Earned slope (8 binding windows) | −0.500 | −0.498…−0.500, spread **0.0018** |
| Unearned slope (2 windows) | −1.000 | −1.007, −1.001 |

Read as a formula, the trained net implements approximately:
payment = max(0, 1002 − max(0, U − 18) − max(0, E − 55)/2) — the
right *shape*, with the earned exclusion treated as a flat ~$55
instead of the statute's $65-plus-spill (which is worth $85 when
U = 0). The kink error is largest exactly where the spill matters
most and shrinks as U grows — the signature of a net that never
learned the spill term and split the difference.

## Predictions scored

1. *Compiled control recovers exactly.* **Holds** (all parameters
   to < $1 / < 0.01, drift 0).
2. *Trained point recoveries look deceptively good (FBR within $15,
   unearned kink within $10).* **Holds** for those two (+$7.91, −$2).
3. *Regional drift will be large: earned-slope spread > 0.05, earned
   kink moving > $5 across U slices.* **Fails.** Slope spread is
   0.0018 — the slopes are learned essentially exactly and uniformly.
   The kink moves only $4 across slices. What the prediction got
   wrong is the failure's shape: not regional variation around the
   right parameters, but one wrong parameter (the kink) applied
   uniformly, plus a missing interaction (kink movement 4 vs the
   statutory 20).
4. *Unearned slope learned better than earned.* **Fails** (earned
   binding spread 0.0018 < unearned 0.0069; both near-exact).

## What this establishes

Error scoring said the trained SSI baseline has max error in the
tens of dollars; the atlas said it violates structural properties.
Neither said *what it learned instead*. Recovery does: a coherent
alternative rule whose only material deviations are a +$8 FBR, a
−$2 unearned kink, and a flattened, ~$30-early earned kink. For a
verification workflow that is the actionable output — it names the
exact feature (the $20-spill into the earned exclusion) that more
capacity, more data, or an architectural prior would need to fix,
and it gives a regression test for any retrain: recover the kink
function again and check 85 − U.

Method note: slope windows in the payment = 0 region are excluded
from drift summaries (there the statutory slope is itself 0);
results/recovery.json records all windows and the binding mask.

Artifacts: `probe.py`, `results/recovery.json`. $0 spend.
