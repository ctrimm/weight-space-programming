# Results — EITC credit

Run 2026-11-02. Weight-writer: Muse Spark in-session.

| Arm | Max abs err | Mean abs err | Within $0.01 | Within $1 |
|---|---|---|---|---|
| Zero-shot, attempt 1 | $0.024 | $0.0044 | 81.8% | 100% |
| Zero-shot, attempt 2 | $0.024 | $0.0000085 | **99.96%** | 100% |
| Trained (Adam, 20k steps) | $5.19 | $0.082 | 15.7% | 99.7% |

**Verdict: Partial by the letter of the pre-registered gate** (PASS
requires max ≤ $0.01 everywhere) — with the cause fully identified and
not in the construction. Attempt 2 is exact at every evaluation point
except x = $57,310 itself, where the error is $0.024. That point is
the seam documented in ERRATUM.md: the Rev. Proc. table's
completed-phaseout amount ($57,310) is the statutory phase-out line's
zero ($57,310.114) rounded to the nearest $10, so the stipulated rule
is discontinuous by $0.024 there. Any continuous network that matches
the formula immediately left of the seam must carry that $0.024 at the
seam point itself; it is the smallest achievable error, not a
construction defect.

Attempt 1's larger footprint (a $0.024 plateau over the whole tail)
came from trusting the rounded endpoint as exact — the same
arithmetic slip the protocol's own "consistency check" made, caught
by the run and corrected in ERRATUM.md rather than edited away.

Lesson that generalizes: published benefit tables round their amounts
independently of the statutory rates they derive from. A scorer (or a
benchmark) that treats rounded table constants as exact will
manufacture apparent model errors at exactly the seams. Raw scores:
`results/`.

## Cost

$0 in API or cloud spend. The zero-shot weights were written by
Muse Spark in-session (no per-call charge), and the trained
baselines were trained on local CPU in this workspace. The only
priced resources consumed anywhere in this repo are the OpenRouter
calls in `../second-writer-openrouter/` ($0.653) and
`../weight-reading/` ($1.3469) — $2.00 total across the project,
each documented in that experiment's RESULTS.md.
