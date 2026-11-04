# Results — certified error bands at an eligibility cliff

**Verdict: the rounding boundary's logic prices the cliff exactly
as registered. A constructed ramp net pays the maximum possible
error ($273 = half the $546 jump) at every band width, confines
its disagreement with the rule to exactly the certified band, and
the price of narrowing the band is a largest-weight that grows
like 1/w — a $2-wide band costs a max weight of 1,500, vs 8 for a
$400 band. The trained control cannot beat the error floor either
($284), but it has no band at all: it is off by more than $1 across
87% of the income domain. It pays small errors everywhere and
certifies nothing.**

## Constructions (grid step $0.25 over G ∈ [0, 6000])

| Band w (certified) | Max error | Measured disagreement width (>$0.01) | Width at >$1 | Max |weight| |
|---|---|---|---|---|
| $400 | $273.00 | 399.75 | 398.75 | 8.0 |
| $200 | $273.00 | 199.75 | 199.25 | 15.5 |
| $100 | $273.00 | 99.75 | 99.75 | 30.5 |
| $50 | $273.00 | 49.75 | 49.75 | 60.5 |
| $10 | $273.00 | 9.75 | 9.75 | 300.5 |
| $2 | $273.00 | 1.75 | 1.75 | 1,500.5 |

## Trained control

Max error $284.42; disagreement width 5,993.75 at the >$0.01
threshold and **5,225.5 at >$1** (of a 6,000-wide domain); max
|weight| 2.7. Split at the cliff (>$0.01): 2,995 below / 2,999
above — essentially symmetric.

## Predictions scored

1. *Max error exactly J/2 at every width.* **Holds** — $273.00 six
   times. The floor from the rounding proof binds the cliff, and
   the construction attains it, width-free.
2. *Measured disagreement region equals the certified band.* **Holds**
   — measured width is the certified w minus one grid step ($0.25)
   at every row; the certificate is exact, not conservative.
3. *Max |weight| grows like 1/w; the w = $2 net's largest weight
   ~250× the w = $400 net's.* **The law holds; the constant was
   loose drafting.** Max |weight| is dominated by the bias terms
   (≈ (L + w/2)/w with L = 3000): exactly hyperbolic in w, ratio
   1,500.5 / 8.0 = 188×, not ~250×. Registered error, reported.
4. *Trained control: max error ≥ 0.9 × $273, band wider than the
   w = $50 construction's, asymmetric.* **Error arm holds** ($284).
   **Width arm holds overwhelmingly** (5,994 vs 50 at >$0.01;
   5,226 vs 50 at >$1). **Asymmetry arm fails** — the split is
   2,995 / 2,999. The trained net's failure shape is not a lopsided
   band around the cliff; it is diffuse error across the whole
   plateau, which is worse for audit purposes than the protocol
   imagined: there is no region outside which it agrees with the
   rule.

## What this establishes

For a discontinuous rule, "the network approximates it" splits
into two very different claims. A constructed net concentrates all
of its unavoidable error into a band you choose in advance, whose
width is a design decision with a readable price in the weights
(1/w, constant ≈ the cliff's location). A trained net distributes
its error everywhere, so no sampling of its behavior can certify
where it agrees with the rule — the same coverage problem the
tamper audit (../weight-tamper-audit/) maps from the other side.
Inside either kind of band the net is wrong by up to $273, and
determinations belong in deterministic code; but as a witness or a
cross-check, only one of these artifacts can say where it might be
wrong.

Artifacts: `cliff.py`, `results/cliff.json`. $0 spend.
