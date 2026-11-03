# Experiment: Medicaid medically-needy share of cost (CA) — Protocol (pre-registered)

Written 2026-11-02 before construction. Part of the witness-pattern
extension Cory commissioned: compile checkable witnesses for the
rule families where the CivBench label audit actually found damage.

## Why this rule

The CivBench Medicaid audit (2026-11-01) found gold labels
bac-medicaid-038/039 wrong because the upstream oracle treats
share of cost > 0 as ineligibility. In California's medically
needy program (42 CFR 435.831), an eligible person enrolls *with*
a monthly share of cost; the amount is the eligibility-relevant
output. Those fixtures' correct SOCs ($1,380 / $880 on $2,500 of
income with $500 / $1,000 of medical expenses) are named probes in
the evaluation below: a witness that cannot reproduce the two
points the audit established by hand is not a witness.

## The rule (stipulated in full)

Medi-Cal medically needy, individual, monthly. Inputs: earned
income E, unearned income U, incurred medical expenses M (not
reimbursed by a third party).

- Countable income by SSI methodology (42 CFR 435.831(b) uses SSI
  income-counting): CU = max(0, U − 20); spill R = max(0, 20 − U);
  CE = max(0, E − 65 − R) / 2; C = CU + CE.
- Share of cost SOC = max(0, C − 600 − M), where $600 is
  California's maintenance need income level (MNIL) for an
  individual — a constant CA has frozen for decades (verified in
  the CivBench audit; the family-of-3 figure there was $934).

Stipulated simplifications: individual only; SSI counting without
its rounding steps; medical expenses enter as a single monthly
figure deducted from the share of cost (the spend-down mechanic);
no health-insurance premium deduction (CA allows one; excluded to
keep the tested structure the audit's); categorical eligibility
out of scope — as with the audit, the question is the amount, and
SOC > 0 means enrolled with that share, never "ineligible."
Domain: E, U in [0, $3,000], M in [0, $2,000].

What this tests beyond the earlier experiments: **composition**.
The SSI counting machine (three staged hinges, including the
spill) becomes a subroutine inside a second program's formula,
with a different threshold and an additional input entering after
the count. If compiled witnesses are going to cover a label
corpus, they will mostly be compositions like this.

## Architecture

Inputs [E/1000, U/1000, M/1000] → Linear(3→24) → ReLU →
Linear(24→24) → ReLU → Linear(24→24) → ReLU → Linear(24→1);
output dollars = 1000 · (final linear).

## Evaluation

Grid E, U in $50 steps × M in {0, 250, 500, 1000, 2000} (18,605
points) + 20,000 seeded uniform random points + 5,000 seeded
kink-jittered points + the two audit probes (0, 2500, 500) → 1380
and (0, 2500, 1000) → 880, which must match to the cent.
Gate: PASS = zero-shot max abs error ≤ $0.01 everywhere, probes
included. Arms: zero-shot written; trained baseline (Adam,
15,000 × 8,192, seed 20261102); oracle.

## Cost

$0 API spend (in-session construction, local training), per the
repo's cost accounting in the root README.
