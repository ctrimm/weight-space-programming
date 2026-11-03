# Protocol — second-writer replication, member-level SNAP (pre-registered)

Written 2026-11-03 before any API call for this experiment. Cory
authorized the full experiment queue on 2026-11-03 with a **$5
total OpenRouter cap**; this is the only item in the queue that
spends. Expected cost, grounded in the two previous second-writer
runs' measured per-call actuals (GPT-5.6 Sol $0.014–$0.053 per
construction; Gemini 3.1 Pro $0.056–$0.201): roughly **$0.25** for
the two calls, plus at most one syntax-repair retry per cell under
the same protocol as before.

## Design

Identical to the previous replications, one new task:

- **Writers:** `openai/gpt-5.6-sol` and
  `google/gemini-3.1-pro-preview` via OpenRouter.
- **Task:** the member-level SNAP rule from
  `../snap-household-composition/PROTOCOL.md`, restated in full in
  the prompt, with its fixed 5-input architecture. No evaluation
  points, oracle code, original weights, or derivations are shared.
  In particular, the writers are *not* told that the rule is
  composition-invariant; if their constructions are invariant, that
  is their own staging choice.
- **Prompts hash-frozen** before the first call
  (`results/prompt_freeze.json`); **weights hash-frozen** after the
  calls and before scoring (`weights/FREEZE.json`).
- **Attempts:** one call per model; one syntax-repair retry on a
  parse failure only (parse error fed back; never scores or hints).
- **Scoring:** the SNAP-composition experiment's own design —
  accuracy on its random + kink sets, composition-invariance range
  over its 90 totals classes, and the earned-to-unearned
  should-change control — applied unchanged. Gates as elsewhere:
  PASS = max abs error <= $0.01.
- **Cost discipline:** the runner checks cumulative measured spend
  before every call and aborts at the $5 cap Cory set for the
  queue. Response-level costs are reconciled against the account
  aggregate afterward, as in the previous runs.
