# Protocol — second weight writer via OpenRouter (pre-registered)

Written 2026-11-02 before any API call. Cory authorized this run
explicitly, with a **hard cap of $10 total measured spend**; the
runner aborts before any call that would start after the cap is
reached, and every call's actual cost is taken from the API
response's usage accounting, never estimated from tokens.

## Question

The four constructions in this repo were written by one model
(Muse Spark, in-session). Is exact weight compilation a property
of that model, or can other frontier models do it under the same
conditions?

## Design

- **Writers:** `openai/gpt-5.6-sol` and `google/gemini-3.1-pro-preview`,
  called through OpenRouter. Both are outside the Anthropic family,
  so agreement is cross-vendor, not cross-version.
- **Tasks:** the same four stipulated rules and architectures as the
  original experiments (EITC, SNAP, SSI, Medicaid), restated in full
  in each prompt. No evaluation points, no oracle code, no hint of
  the original constructions. The EITC prompt uses the rule exactly
  as originally stipulated (completed phaseout $57,310), with no
  warning about the rounding seam — the same conditions as the
  original attempt 1.
- **Output format:** a sparse JSON serialization of the weight
  arrays (shape + nonzero entries; unspecified entries are 0). This
  is a serialization choice, not a construction hint: the original
  constructions were also sparse, and any dense solution can be
  serialized the same way by omitting zeros.
- **Attempts:** one call per model x experiment. If a reply cannot
  be parsed as the specified JSON, exactly one retry is allowed with
  the parse error reported back (a syntax repair only — no scores,
  errors, or oracle information are ever fed back). Retries are
  logged and disclosed.
- **Scoring:** the original experiments' own evaluation designs
  (same point sets, same seeds), run after all calls complete.
  Gates unchanged: PASS = max abs error <= $0.01; Partial <= $1.
- **Cost discipline:** max_tokens bounded per call; cumulative
  measured cost checked before every call; abort at $10. All raw
  responses and the cost log are preserved in this folder.

## Known asymmetries vs the original run (declared in advance)

- The original writer worked in an interactive session and produced
  a construction script with derivations; these writers get one
  stateless prompt and return weights directly. Conditions are
  harsher for the API writers, and any failure should be read with
  that in mind.
- The sparse-JSON format quietly favors constructions with few
  nonzero parameters. A model attempting a dense solution faces a
  much longer output. This is a real, declared bias of the harness.
