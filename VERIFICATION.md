# Full-suite verification — 2026-11-03

Cory asked for the entire experiment set to be rechecked before
the results are used to support a publication and author outreach.
Method: every experiment's own scoring entry point was re-run in
the foreground against the weight files as shipped in this repo,
and the output compared with the decisive numbers in that
experiment's RESULTS.md. Nothing was retrained and no weight file
was modified; scoring is deterministic. Where a number below says
"reproduced," it matches the published figure to the digits shown.

## Reproduced as published

| Experiment | Decisive claim | Recheck |
|---|---|---|
| SNAP allotment | compiled max err 3.41e-13 | 3.411e-13 |
| EITC | attempt max err $0.024 (Partial, table seam) | $0.0240000000025 |
| SSI payment | compiled max err 2.27e-13 | 2.274e-13 |
| Medicaid patient liability | compiled max err 1.82e-12 | 1.819e-12 |
| SSI stale-FBR edit | stale vs 2026 oracle: max $51.00, mean $17.65, 63.6% within a cent; edited net exact | $51.00000000000024, $17.652, 63.65%; edited 2.27e-13 |
| Second writer (4 rules) | GPT + Gemini PASS SNAP/SSI/Medicaid; Gemini EITC Partial ($0.024); GPT EITC declined = FAIL | identical gates and errors, incl. Gemini EITC $0.0240000000025 |
| Weight reading | 11 artifacts frozen before reads | 11/11 SHA-256 hashes match FREEZE.json (the reads themselves were LLM calls; their grading rests on the preserved raw replies, not re-run) |
| Medicaid share of cost | compiled max err 1.36e-12; audit probes $1,380 / $880 | 1.364e-12 (probe values are in the folder's results files; the atlas rerun below independently puts the trained arm's probe error at $21.90, as published) |
| SSI couple payment | compiled max err 5.83e-13 | 5.826e-13 |
| Second writer, witness extension | 4/4 cells PASS exact; Gemini SOC array-identical | all PASS, max errs 4.83e-13 to 1.364e-12 as published |
| SSI couple split-invariance | trained grid max range $25.15, mean $3.64, 98.9% of classes move; compiled <= 8.9e-13 | $25.15, $3.641, 98.9%; compiled worst 8.882e-13 |
| Repair bake-off (post-D14 corrected figures) | augmentation best learned arm (acc $8.98, invariance $5.30); penalty weaker ($48.07 / $19.28); sum-first invariance exactly 0, acc $25.14 | $8.977 / $5.295; $48.07 / $19.28; $0 / $25.14 |
| Member-level SNAP + replication | original max 4.26e-13, invariance 2.84e-13; GPT 4.26e-13; Gemini 4.83e-13, invariance 3.84e-13 | all reproduced to the digits shown |
| Canonical circuits | Medicaid + SOC: one circuit, all writers; SNAP: original = GPT, Gemini different; SSI + couple: GPT = Gemini, original differs; EITC pairs not gauge-equivalent | identical equivalence pattern and active-unit counts |
| Structural property atlas | compiled arms: 0.0% of probes over a cent everywhere; trained violations as tabled (SNAP bounds $4.53/21.5%, EITC plateau $1.57/98.6%, Medicaid bounds $28.02, SOC probes $21.90, couple swap $23.47/73.4%) | all reproduced (one bound corrected — see below) |
| CivBench label side | 156 scored fixtures; composition held only as scalar fields; 2 one-element lists + 2 meta dicts are the only non-scalars | 156 fixtures; exactly bac-eitc-027 dependents (list), bac-wic-011 receives_benefits (list), bac-ssi-031 + bac-wic-015 meta (dict); key-permutation stability: True |
| SSA PIA held-out (new, 2026-11-03) | all three writers PASS, max 3.64e-12, identical 8-parameter circuits | run and scored in the foreground in this session; see experiments/ssa-pia-heldout/RESULTS.md |

## One correction found

- **Atlas bound.** RESULTS.md stated the compiled arms' worst
  adverse amount was "at most 4.4e-13 dollars." The rerun puts the
  SSI-couple original's worst monotonicity probe at **8.882e-13**.
  The sentence has been corrected in place with a dated note.
  Nothing depends on the difference: both figures are a
  trillionth-of-a-dollar scale, and the 0.0%-over-a-cent result
  reproduced exactly. (The 8.882e-13 figure is consistent with
  the split-invariance audit's own published bound of 8.9e-13.)

## Scope notes

- The four original trained baselines' headline maxima (SNAP
  $5.35, EITC $5.19, SSI $9.54, Medicaid $31.67) rest on their
  folders' build-time result files; the trained arms that carry
  the public claims (couple vanilla/augmentation/penalty/sum-first,
  split-invariance, atlas violations, member-level SNAP trained)
  were all re-scored today and reproduced.
- OpenRouter spend figures ($0.653 pilot, $1.3469 reading,
  $0.2095 witness replication, $0.1949 SNAP-composition
  replication, $0.0831 held-out) rest on the preserved per-call
  cost logs in each folder, reconciled against the account
  aggregate at run time; they were not re-spent for this check.
- The SSA rounding boundary experiment was executed today; its
  arms 1, 2, and 4 were measured in this session (max errors
  $0.0500, $0.0500, $0.0999 against a pre-registered $0.05 floor
  prediction). The trained arm's numbers land in its RESULTS.md.
