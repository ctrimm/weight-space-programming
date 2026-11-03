# Experiment: Medicaid patient liability (post-eligibility) — Protocol (pre-registered)

Written 2026-11-02, before any weights were constructed or scored.
Same question as the repo's other experiments: can an LLM compile a
stated benefits formula into network weights, exactly, with no
training — and does that beat training the same architecture?

## Why this Medicaid rule

Most of MAGI Medicaid is thresholds — step functions a continuous
network cannot represent exactly, so a compilation test there would
measure the wrong thing. The arithmetic core with real structure is
post-eligibility treatment of income (42 CFR 435.725 / SSA §1924):
what a nursing-facility resident actually owes the facility each
month after the spousal and personal allowances. (The other Medicaid
arithmetic the CivBench audit surfaced — California medically-needy
share of cost, 42 CFR 435.831 — is a single hinge once income is
counted, so it would test nothing SNAP didn't.)

## The rule (stipulated in full; no recall required)

Monthly patient liability for a nursing-facility resident with a
community spouse. Inputs: resident's gross monthly income I,
community spouse's own monthly income SI, resident's monthly health
insurance premiums P.

- Community spouse monthly income allowance
  A = max(0, 2,643.75 − SI) (the resident's income is redirected to
  bring the spouse up to the MMMNA; here the MMMNA is stipulated at
  the federal minimum, so no excess-shelter uplift and the federal
  maximum of $4,066.50 never binds)
- Liability L = max(0, I − 50 − A − P) ($50 personal needs allowance,
  then the spousal allowance, then premiums)

Constants: MMMNA federal minimum $2,643.75 (effective Jul 1, 2025 –
Jun 30, 2026, SSA/CMCS spousal impoverishment standards); federal
maximum MMMNA $4,066.50 for 2026 (non-binding under the stipulation).
Personal needs allowance: federal floor is $30; $50 is stipulated as
the state figure (many states use $50) and labeled a stipulation, not
a federal constant.

Stipulated simplifications: MMMNA held at the federal minimum (no
excess shelter allowance, no state variation); no dependent/family
allowance; no unpaid medical expense deductions; the allowance is not
separately capped by the resident's available income (the liability
floor at $0 makes the liability itself correct regardless); asset
tests (CSRA), the income cap / Miller trust rules, and categorical
eligibility are out of scope — payment arithmetic only. Domain:
I ∈ [0, $6,000], SI ∈ [0, $5,000], P ∈ [0, $600].

Structure being tested: a floor inside a floor — the spousal
allowance is itself a hinge of SI, and it enters the liability's own
hinge. Two nested stages, three inputs.

## Architecture

Inputs [I/1000, SI/1000, P/1000] → Linear(3→24) → ReLU →
Linear(24→24) → ReLU → Linear(24→1); output dollars = 1000 · (final
linear).

## Arms

1. **Zero-shot weight-written** — reasoning from the rule text only;
   one shot; `.npz` hashed (FREEZE.json) before the scorer is run.
2. **Trained baseline** — same architecture, Adam on uniform oracle
   samples (15,000 steps × 8,192, seed 20261102).
3. **Oracle** — the stipulated formula; the ceiling.

## Evaluation

Grid I $0–$6,000 × SI $0–$5,000 in $50 steps × P ∈ {0, 150, 300, 450,
600} (121 × 101 × 5 = 61,105 points) + 20,000 seeded uniform random
points + 5,000 seeded kink-jittered points (near SI = 2,643.75 and
near the liability-zero plane). Metrics: max / mean abs error, %
within $0.01, % within $1, region breakdown.

## Gate

PASS = zero-shot max abs error ≤ $0.01 on every set. Partial = max
≤ $1. Worse = fail, reported by region.
