# Results — which audit instruments catch a weight-space tamper?

**Verdict: the amended prediction table holds in every cell.**
Behavioral instruments certify the function only where they look:
each one misses exactly the tampers placed outside its coverage,
and the weight-anatomy audit — the one instrument whose coverage
is the artifact itself rather than a sample of its behavior — is
the only one that catches all three. The clean net fires nothing:
no instrument false-positives on the untampered witness.

## Detection matrix (measured)

| Instrument | T-global (FBR +$25) | T-local-near (tent, U 1200–1400) | T-local-far (tent, U 1800–2000) |
|---|---|---|---|
| Standard scoring (grid to U = 1500) | detects | detects | **misses** |
| Parameter-recovery probing (U ≤ 800) | detects | **misses** | **misses** |
| Property atlas, as implemented (probes to U = 1450) | detects | detects (mono-U straddle) | **misses** |
| Weight-anatomy audit | detects | detects | detects |

Tamper footprints, for scale (dense grid E ≤ 3000, U ≤ 2500):
T-global moves payments by up to $25 across 15.4% of the domain;
each tent moves payments by up to $30 across 7.6% of the domain —
in the far band, a region where the true payment is $0 at every
income, so the tampered net pays up to $30 to households owed
nothing, indefinitely, invisibly to every behavioral check this
repo runs.

How the anatomy audit catches each: T-global decodes to FBR $1,019
at the output site (≠ statute $994 — the same decode the blind
readers performed in the weight-reading experiment); the tents
activate spare units outside the construction's census
(L1 {0,1,2} → {0..6}, etc.) with non-gadget coefficients (10, −18…).

## Deviations from the protocol as first drafted, disclosed

1. The protocol's original single-tamper design and its "uniform
   grid detects" prediction were amended **before any run**, after
   grounding the instruments in the repo's code showed every
   behavioral instrument caps at U ≤ 1500. The amendment and the
   corrected table are in PROTOCOL.md; the run used the amended
   design (two tents: near, far).
2. The tents gate on U only, not on E ∈ [0, 50] as drafted: the
   net's three hidden layers cannot express the min-of-3 gadget,
   and inside either U band the true payment is $0 at every E
   (U − 20 > 994), so the strip is the depth-feasible form of the
   same tamper. Its footprint is larger, not smaller, than drafted.

## What this establishes

The compiled-witness story in this repo is: exactness you can
verify. This experiment adds the adversarial corollary: exactness
you can verify *behaviorally* extends exactly as far as your
probes reach, and an editor who knows where the probes stop can
plant a defect behind them. The defense is not more sampling — it
is auditing the artifact: a compiled net's weights decode to a
formula, and any unit or constant that does not decode is a
finding. That is the mechanical form of the blind weight-reading
result, and it is the only instrument in this set with whole-
artifact coverage. (The obvious synthesis — atlas probe lines
drawn from the weights, so behavioral coverage follows the
anatomy — is future work, recorded in the protocol.)

Artifacts: `audit.py`, `results/audit.json`. $0 spend.
