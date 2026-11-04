# Protocol — second-writer replication on local open-weight models (pre-registered)

Written 2026-11-03 before any task-prompt call for this experiment.
Brandon requested this run ("use the Rig AI compute to replicate
this study"). It spends no API money: every call goes to
open-weight models served on one RTX 3090 Ti in Brandon's homelab
("the rig"), through its LiteLLM gateway. The only cost is
electricity and GPU time.

## Question

Exact weight compilation replicated across three frontier writers
(Muse Spark in-session; GPT-5.6 Sol and Gemini 3.1 Pro over
OpenRouter). Does it need a frontier model, or can **open-weight
models small enough to run on one consumer GPU** do it under the
same stateless one-shot conditions?

## Design

- **Writers** (LiteLLM aliases on the rig, server-default sampling):
  - `q38`: Qwen3.8-27B, 3-bit quant with embedded draft head,
    llama.cpp via llama-swap, fully on GPU.
  - `strata`: Qwen3.8-Flash-Next, 125B MoE (~6B active), IQ3_S,
    Strata engine, split across GPU and system RAM.
  Both are reasoning models; reasoning comes back in a separate
  `reasoning_content` field and is preserved alongside the reply.
- **Tasks:** all seven rules from the three earlier second-writer
  runs. `eitc`, `snap`, `ssi`, and `medicaid` come from
  `../second-writer-openrouter/`; `soc` and `couple` from
  `../second-writer-witness-extension/`; `snapcomp` from
  `../second-writer-snap-composition/`. The prompts are not
  retyped. The runner imports each earlier runner's own `TASKS` /
  `FORMAT_SPEC` / prompt construction, and before the first call it
  checks the hashes for `soc`, `couple`, and `snapcomp` against
  those runs' `prompt_freeze.json` files. It aborts on any mismatch.
  The first pilot never froze its prompts, so those four get hashed
  here for the first time. The EITC prompt keeps the original
  $0.024 seam with no warning, exactly as before.
- **Prompt freeze:** SHA-256 of all seven prompts is written to
  `results/prompt_freeze.json` before the first task call.
- **Attempts:** one call per writer and task, with exactly one
  syntax-repair retry if the reply can't be parsed. The retry gets
  only the parse error, never scores, points, oracle output, or
  hints. Refusals and unparseable second replies count as no
  submission (FAIL).
- **Weights freeze:** after all calls and before scoring, SHA-256 of
  every `.npz` file is recorded in `weights/FREEZE.json`.
- **Scoring:** the original experiments' own evaluation designs,
  imported from their folders unchanged (same point sets, seeds,
  probes, slices, invariance classes). Gates are unchanged:
  PASS = max abs error <= $0.01; Partial <= $1; otherwise FAIL.
  Before scoring any rig weights, the scorer re-scores the
  committed Gemini 3.1 Pro weights and must reproduce their
  published max errors. That is a check that the harness runs the
  same way on this machine.

## Declared deviations from the OpenRouter runs

- **Output budget:** `max_tokens` = 32,768 (it was 20,000). On a
  local llama.cpp/Strata server, reasoning tokens count against
  this budget, and a 3-bit model reasoning at length could
  otherwise be truncated before it emits any JSON. That would
  measure the budget, not the skill. The extra headroom is fixed in
  advance and is the same for both writers.
- **Transport retries:** a call that fails at the transport layer
  with no model output (HTTP 502/503/504, connection refused or
  reset, timeout) is retried up to 3 times with a pause, and those
  retries are not counted as attempts. Reason: on the smoke test
  before this protocol was written, the `q38` lane crashed on three
  consecutive requests and then recovered on its own. A 502 from a
  restarting process says nothing about the model. All transport
  retries are logged.
- **Order:** every `q38` task runs first, then every `strata` task.
  The two lanes are mutually exclusive on the GPU, so this keeps
  model swaps to one.
- **No dollar cost:** `cost_log.json` records wall time and token
  counts instead.

## Declared asymmetries (beyond those already declared for the earlier runs)

- These are quantized models (3-bit class). Any failure mixes "the
  model can't" with "the quantized copy can't". This run does not
  separate the two.
- The single stateless prompt and the sparse-JSON format still
  apply, along with their stated biases: the writer works with no
  scratch tools, and sparse constructions are favored.
- Not replicated: the stale-FBR editing experiment (5). It is a
  weight edit, not a compile, and the earlier second-writer runs
  didn't cover it either.
