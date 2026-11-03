# Results — SSI federal payment

Run 2026-11-02. Weight-writer: Muse Spark in-session.

| Arm | Max abs err | Mean abs err | Within $0.01 | Within $1 |
|---|---|---|---|---|
| Zero-shot weight-written | **$0.0000000000002** | ~$0 | **100%** | 100% |
| Trained (Adam, 15k steps) | $9.54 | $0.15 | 26.6% | 95.4% |

**Verdict: PASS** — exact compilation on all 32,381 evaluation points
(dense grid + uniform random + kink-jittered), with the CFR-correct
$20 exclusion spill included. The construction stages three layers:
exclusions (countable unearned + the spill), the earned excess over
the combined $65 + spill exclusion, then the payment floor over the
assembled countable income. Five active units across 72 hidden slots.

The trained baseline's most instructive failure is the spill strip
itself. The spill only operates when unearned income is within $20 of
zero — a band 20 dollars wide in a 1,500-dollar domain — so uniformly
sampled training almost never sees it, and the trained net's mean
error in the spill regions is $4.88–$7.22 (small grid counts: 79 and
4 points) against $0.15 overall. Kink-jittered points overall average
$0.53 error vs $0.15 on the plain sets. A rule detail that is narrow
in input space is nearly invisible to sampling — and it is precisely
the detail the benefits-as-code oracle behind CivBench omits entirely
(its $20 exclusion never spills; SSI audit, 2026-11-01). Raw scores:
`results/`.

## Cost

$0 in API or cloud spend. The zero-shot weights were written by
Muse Spark in-session (no per-call charge), and the trained
baselines were trained on local CPU in this workspace. The only
priced resources consumed anywhere in this repo are the OpenRouter
calls in `../second-writer-openrouter/` ($0.653) and
`../weight-reading/` ($1.3469) — $2.00 total across the project,
each documented in that experiment's RESULTS.md.
