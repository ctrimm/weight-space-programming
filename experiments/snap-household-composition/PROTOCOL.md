# Experiment: SNAP allotment, member-level household composition — Protocol (pre-registered)

Written 2026-11-03 before construction. Part of the queue Cory
commissioned after the split-invariance audit: the existing SNAP
net receives household totals, so it cannot even express *who*
earned the income. This experiment rebuilds the rule at member
level and tests composition directly.

## The rule (stipulated in full)

SNAP monthly allotment, FY2026, household of 2, no elderly or
disabled member. Inputs: each member's earned and unearned income
(E1, U1, E2, U2) and household shelter cost S.

- Combined earned E = E1 + E2; combined unearned U = U1 + U2.
- Adjusted income A = 0.80 * E + U - 209 (20% earned-income
  exclusion; $209 standard deduction for a household of 2).
- Excess shelter X = min(max(0, S - 0.5 * A), 744).
- Net income N = max(0, A - X).
- Allotment B = max(0, 546 - 0.30 * N) ($546 maximum allotment,
  household of 2).

Stipulated simplifications: no eligibility gates (gross/net income
tests are determinations, out of scope as everywhere in this repo);
A is used unfloored in the shelter calculation; no rounding.
Domain: E1, E2 in [0, $3,000]; U1, U2 in [0, $2,000]; S in
[0, $2,500].

## What this tests

1. **Composition invariance (should-not-change):** the allotment
   depends only on the combined totals (E, U) and S. Every
   reallocation of the same totals between the two members must
   produce the identical allotment. Measured as within-class range
   over composition classes, as in the SSI couple audit.
2. **Composition sensitivity (should-change control):** moving
   $100 of a member's income from earned to unearned — same member,
   same personal total — *must* change the allotment (unearned
   income gets no 20% exclusion and no halving anywhere). A test
   suite that only checks invariance would miss an implementation
   that wrongly treats all income alike; this control checks the
   delta against the oracle.
3. Whether the compiled-vs-trained gap from the couple rule
   reproduces in a second program family.

## Architecture

Inputs [E1/1000, U1/1000, E2/1000, U2/1000, S/1000] ->
Linear(5->24) -> ReLU -> Linear(24->24) -> ReLU -> Linear(24->1);
output dollars = 1000 * (final linear). The trained baseline uses
the same architecture.

## Evaluation

- **Accuracy:** 25,000 seeded uniform random points + 5,000 seeded
  kink-jittered points (near the A = 0, shelter, shelter-cap,
  N = 0, and B = 0 planes). Gate: PASS = zero-shot max abs error
  <= $0.01.
- **Composition classes:** combined E in {0, 250, 1000, 2500, 4000,
  6000} x combined U in {0, 100, 500, 2000, 4000} x S in {0, 800,
  2000} = 90 classes; earned and unearned shares each over the
  11-point share grid from the couple audit (121 compositions per
  class). Per arm: max / mean within-class range.
- **Should-change control:** 2,000 seeded households; shift $100
  from E1 to U1 where E1 >= 100; compare each arm's allotment delta
  against the oracle's delta (max / mean abs delta error).
- Arms: zero-shot written; trained baseline (Adam 15,000 x 8,192,
  seed 20261102). The second-writer replication, if run, is scored
  on this same design from its own folder.

## Cost

$0 API spend (in-session construction, local training).
