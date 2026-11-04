# Results — second-writer replication on local open-weight models

**Verdict: the replication holds on one consumer GPU.** Two
open-weight Qwen models were served from a single RTX 3090 Ti in a
homelab. Each got one stateless prompt per rule, with no evaluation
points, no oracle code, and no hint of the original constructions.
Both compiled **six of the seven rules exactly**, to within a
trillionth of a dollar. On EITC both reproduced the original
writer's first attempt metric for metric (Partial, $0.024 seam),
just as Gemini did. No cell went without a submission.

## Scores (original evaluation designs, unchanged gates)

Max abs error in dollars. PASS = at most $0.01 everywhere.

| Rule (points) | Original (Muse Spark) | GPT-5.6 Sol | Gemini 3.1 Pro | **q38** (Qwen3.8-27B, 3-bit) | **strata** (Qwen3.8-Flash-Next, IQ3_S) |
|---|---|---|---|---|---|
| SNAP (40,049) | PASS 3e-13 | PASS 3.4e-13 | PASS 3.4e-13 | **PASS 3.4e-13** | **PASS 3.4e-13** |
| SSI individual (32,381) | PASS 2e-13 | PASS 2.3e-13 | PASS 2.3e-13 | **PASS 1.7e-13** | **PASS 2.3e-13** |
| Medicaid liability (86,105) | PASS 2e-12 | PASS 1.8e-12 | PASS 2.7e-12 | **PASS 2.7e-12** | **PASS 2.7e-12** |
| Share of cost (43,607) | PASS 1.4e-12 | PASS 1.4e-12 | PASS 1.4e-12 | **PASS 1.4e-12** | **PASS 1.4e-12** |
| SSI couple (44,762) | PASS 5.8e-13 | PASS 5.8e-13 | PASS 5.8e-13 | **PASS 5.8e-13** | **PASS 5.8e-13** |
| SNAP member-level (30,000) | PASS | PASS 4.3e-13 | PASS 4.8e-13 | **PASS 4.3e-13** | **PASS 4.8e-13** |
| EITC (2,818), attempt-1 rule | Partial $0.024, 81.8% within 1¢ | no submission (declined) | Partial $0.024, 81.8% | **Partial $0.024, 81.8%** | **Partial $0.024, 81.8%** |

Full metrics are in `results/replication_scores.json`, and the
runner's own lines are in `results/runner.log`.

The secondary checks also all held, for both writers:

- **Share-of-cost audit probes:** exactly $1,380.00 and $880.00,
  the two values the benefits-as-code oracle got wrong.
- **Couple zero-member slices:** max 3.4e-13. These are the slices
  where the trained baseline made its worst errors ($48).
- **Member-level SNAP:** composition-invariant across all 90 totals
  classes (max range 4.4e-13), and it matches the oracle on the
  earned-to-unearned should-change control (4.7e-13).

**Scorer calibration.** Before scoring the rig weights, the scorer
re-scored the committed Gemini 3.1 Pro weights for all seven rules
on this machine. It reproduced every published max error to within
1e-9 (`calibration_gemini_rescore` in the scores file).

## EITC: the seam, found by every writer

The prompt keeps the rule exactly as first stipulated, completed
phaseout at $57,310, with no warning. That rule has a built-in
2.4-cent contradiction. The phaseout line reaches $0.024 at
$57,310, not $0, and the rule says $0 from there on. A ReLU network
is continuous, so no network can jump at that point (see
`../eitc-credit/ERRATUM.md`).

Both local models found the contradiction: their reasoning traces
compute the $0.024 residual explicitly. Both then submitted the
same continuous construction the original writer submitted on
attempt 1, and the same one Gemini submitted. The phaseout runs to
its own zero, and the network leaves $0.024 from $57,310 to
$70,000. That gives max error $0.024, with 81.8% of points within
a cent (the whole tail is the miss) and 4e-12 everywhere below the
seam.

So four writers submitted the same continuous construction and one
(GPT-5.6 Sol) refused to submit an inexact network. Every writer
that reasoned about the seam reached the same 2.4 cents. The
original writer's 99.96% came on attempt 2, after the erratum, and
was not part of any second-writer prompt.

## Process

| | q38 | strata |
|---|---|---|
| Calls (including syntax-repair retries) | 11 | 7 |
| Cells that needed the syntax-repair retry | 4 (EITC, SNAP, share of cost, couple) | 0 |
| Completion tokens (reasoning + reply) | 119,508 | 111,597 |
| Largest single reply | 17,379 tokens | 24,922 tokens |
| Wall time | 30 min | 28 min |
| Transport retries | 0 | 0 |

- **q38's retries were syntax-only.** Twice it used an index past
  the array bounds (`"1,1"` in a [16,1] matrix), and twice it
  emitted malformed JSON. Each retry got only the parse error, as
  the protocol allows. The submitted constructions are its second
  replies. Read q38's 7/7 submissions with that in mind: the
  frontier writers needed one retry in total, the 3-bit 27B model
  needed four.
- **The larger output budget mattered for strata.** Two of its
  replies, EITC at 24,817 tokens and member-level SNAP at 24,922,
  went past the 20,000-token cap the OpenRouter runs used. Under
  that cap they would probably have been cut off before any JSON.
  The 32,768 cap was fixed in `PROTOCOL.md` before any call, but
  that is a real difference in conditions.
- **No leakage signal.** None of the 18 reasoning traces mention
  the repo, the paper, or the original writer.

## Cost

**$0 in API spend.** All calls ran on the homelab rig through its
LiteLLM gateway, about 58 minutes of GPU time in total.

Artifacts:
- `raw/`: replies, plus `*.reasoning.txt` reasoning traces
- `results/prompt_freeze.json`: prompt hashes. The `soc`, `couple`,
  and `snapcomp` hashes match the earlier runs' freezes.
- `weights/FREEZE.json`: weight hashes, written after the last call
  and before scoring.
