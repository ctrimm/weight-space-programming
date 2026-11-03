# Protocol — held-out test: SSA PIA from regulatory text only (pre-registered)

Written 2026-11-03, before any construction, oracle code, or API
call for this experiment. Cory authorized the final-experiments run
2026-11-03 with the quoted estimate of roughly $0.20–$0.50 for the
API calls in view; the runner enforces a **hard cap of $0.50**
measured spend for this experiment and aborts before any call that
would start after the cap is reached.

## Why this experiment exists

Every rule compiled in this repo before today was selected by the
experimenter and restated as a cleaned formula before any writer
saw it. That leaves the strongest objection unanswered: the writers
were handed the answer in equation form. This experiment removes
the cleanup step.

## Rule

The US Social Security primary insurance amount (PIA) for a worker
first eligible in **2027**, exactly as published by the Social
Security Administration at
https://www.ssa.gov/oact/cola/piaformula.html (see SOURCE_TEXT.md,
retrieved 2026-11-03). No experiment in this repo has touched
Social Security before; the 2027 bend points were published after
every prior experiment was run.

## Design

- **What the writers receive:** the verbatim SSA text in
  SOURCE_TEXT.md (definition, bend-point paragraph, formula
  paragraph), plus the fixed network spec below, plus one task
  instruction: implement the sum (a)+(b)+(c); do **not** implement
  the rounding sentence (rounding is a separate experiment,
  `../ssa-rounding-boundary/`). No cleaned formula, no hinge
  decomposition, no evaluation points, no oracle code.
- **Writers:** `openai/gpt-5.6-sol` and `google/gemini-3.1-pro-preview`
  via OpenRouter (same harness conventions as
  `../second-writer-openrouter/`), plus a control construction by
  Muse Spark in-session, built and hash-frozen **before** the API
  weights are scored.
- **Architecture (fixed, same family as the EITC task):** input
  [x/10000], x = average indexed monthly earnings in dollars;
  W1 [16,1], b1 [16], ReLU, W2 [1,16], b2 [1] (linear); output
  dollars = 1000 x final linear output. Sparse-JSON serialization,
  unlisted entries are 0.
- **Attempts:** one call per model. If a reply cannot be parsed as
  the specified JSON, exactly one retry with the parse error only
  (no scores or oracle information fed back). Retries disclosed.
- **Oracle:** an independent direct implementation of the SSA sum:
  0.90 x min(x, 1340) + 0.32 x the portion of x over 1340 through
  8077 + 0.15 x the portion of x over 8077, unrounded. Written from
  the same source text; never shown to the writers.
- **Scoring set (fixed in advance):** AIME grid $0–$20,000 in $0.25
  steps (80,001 points); kink neighborhoods $1,340 +/- $5 and
  $8,077 +/- $5 in $0.001 steps (20,002 points); 20,000 uniform
  random points, seed 20261103.
- **Gate (unchanged from the rest of the repo):** PASS = max abs
  error <= $0.01 on the full scoring set; Partial <= $1. A refusal
  or an unparseable final reply is recorded as FAIL, as before.

## Declared limitations

- The SSA text states the percentages and bend points in words;
  deriving weights still requires the writer to convert prose to
  arithmetic. That conversion is the skill under test.
- The writers may have seen this formula in training data. The
  experiment tests weight-writing from text under this repo's
  conditions, not discovery of an unknown rule.
- Single-input, three-segment rule: deliberately at the simple end.
  It is a held-out replication, not a difficulty escalation.
