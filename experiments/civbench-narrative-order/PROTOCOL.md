# Protocol — CivBench narrative-order test, label side (pre-registered)

Written 2026-11-03. The queue item as scoped when Cory approved it:
the deterministic-label side runs locally at $0; running caseworker
models over permuted scenarios belongs to CivBench Build 1, which
has its own spend estimate and approval (December, on the witnessed
gold core) and is **not** part of this run.

## Question

If household members are presented in a different order, can a
CivBench label change? For the deterministic side this decomposes
into two checks:

1. **Representation audit:** do the bac-* fixture scenarios contain
   any orderable member structure (lists of members, per-member
   records) whose permutation could reach the label computation?
2. **Permutation check:** for any fixture that does contain such a
   structure, permute it and confirm the consumed content — and
   hence the label — is unchanged.

## Method

Read-only audit of all bac-* fixtures in the CivBench AISI-shift
staging repo (`~/workspace/civbench-aisi-shift/evals/`): scenario
value types per fixture, identification of list/dict values,
content stability under scenario key-order permutation, and
inspection of any list-valued fields (element counts and whether
the label rule quantifies over them order-free).

## Scope note

The model side of narrative order — permuting the *fact order in
rendered prompts* and measuring answer stability of caseworker
models — is the version of this test with real failure surface,
because the structured scenarios are rendered into prose prompts
and prose has an order. That run is Build 1 work: it needs the
spend estimate and Cory's approval that gate all CivBench
real-model runs, and it should run on the witnessed gold core, not
the silver labels.
