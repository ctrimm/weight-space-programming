# Results — SNAP allotment

Run 2026-11-02. Weight-writer: Muse Spark in-session.

| Arm | Max abs err | Mean abs err | Within $0.01 | Within $1 |
|---|---|---|---|---|
| Zero-shot weight-written | **$0.0000000000003** | ~$0 | **100%** | 100% |
| Trained (Adam, 15k steps) | $5.35 | $0.21 | 6.1% | 95.9% |

**Verdict: PASS** — exact compilation, thirteen orders of magnitude
inside the pre-registered gate, on all 40,049 evaluation points
including the kink-jittered set. The construction uses 3 of 24
first-layer units and 2 of 24 second-layer units; the rest are zero.
Verified against an independent scalar re-implementation of the rule
(max disagreement 3e-13).

The trained net's errors concentrate at the rule's kinks, as
pre-registered: kink-jittered points average $0.44 error vs $0.18 on
the plain grid, and the worst region is the intersection of two kinks
(shelter cap binding + benefit exhausted: mean $1.81, only 31% within
$1). Training approximates the formula; the written weights *are* the
formula. Raw scores: `results/`.

## Cost

$0 in API or cloud spend. The zero-shot weights were written by
Muse Spark in-session (no per-call charge), and the trained
baselines were trained on local CPU in this workspace. The only
priced resources consumed anywhere in this repo are the OpenRouter
calls in `../second-writer-openrouter/` ($0.653) and
`../weight-reading/` ($1.3469) — $2.00 total across the project,
each documented in that experiment's RESULTS.md.
