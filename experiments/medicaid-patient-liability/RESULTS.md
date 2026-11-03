# Results — Medicaid patient liability

Run 2026-11-02. Weight-writer: Muse Spark in-session.

| Arm | Max abs err | Mean abs err | Within $0.01 | Within $1 |
|---|---|---|---|---|
| Zero-shot weight-written | **$0.000000000002** | ~$0 | **100%** | 100% |
| Trained (Adam, 15k steps) | $31.67 | $0.80 | 3.3% | 79.1% |

**Verdict: PASS** — exact compilation on all 86,105 evaluation points
(three-input grid + uniform random + kink-jittered). The construction
is the smallest in the repo: one hinge for the community-spouse
allowance (MMMNA $2,643.75 at the federal minimum), pass-throughs for
income and premiums, and one hinge for the liability floor — a floor
inside a floor, staged across two layers.

The trained baseline is the weakest in the repo, and the reason is
structural: over half the domain has liability exactly $0, and the
trained net smears that floor (liability-zero regions average
$1.91–$2.90 error; kink-jittered points average $2.39, only 46.5%
within $1, worst single error $31.67). A model that is "79% within a
dollar" on this function would overstate what most residents owe in
most months — the floor is the function. Raw scores: `results/`.

## Cost

$0 in API or cloud spend. The zero-shot weights were written by
Muse Spark in-session (no per-call charge), and the trained
baselines were trained on local CPU in this workspace. The only
priced resources consumed anywhere in this repo are the OpenRouter
calls in `../second-writer-openrouter/` ($0.653) and
`../weight-reading/` ($1.3469) — $2.00 total across the project,
each documented in that experiment's RESULTS.md.
