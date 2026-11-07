# Inspect AI port of the weight-tamper audit

The original experiment (`../audit.py`, see `../PROTOCOL.md` and
`../RESULTS.md`) asks which of four audit instruments catch which of
three weight-space tampers planted in a compiled SSI net. This folder
re-runs the same audit as an [Inspect AI](https://inspect.aisi.org.uk/)
(UK AISI) task, so the full detection matrix becomes a browsable
`.eval` log: one sample per matrix cell, with the instrument's
measurements attached to each sample.

Nothing is reimplemented. The Inspect solver imports `../audit.py`
and calls its tamper constructors (`t_global`, `t_local`) and
instrument functions (`inst_scoring`, `inst_probing`, `inst_atlas`,
`inst_anatomy`) directly. The scorer compares each cell's `detects`
flag against the expected matrix in `../results/audit.json`.

The 16 samples are the 4 conditions (clean, T-global, T-local-near,
T-local-far) × the 4 instruments (scoring, probing, atlas, anatomy).

## Setup

Uses the shared Inspect venv (inspect_ai is not in system python):

```bash
~/workspace/tooling/inspect-venv/bin/python -m pip install inspect_ai pytest numpy
# (already installed if the venv exists)
```

## Run the eval

From this directory:

```bash
cd experiments/weight-tamper-audit/inspect_port
~/workspace/tooling/inspect-venv/bin/inspect eval tamper_audit_task.py --model mockllm/model
```

No LLM is involved in the audit itself — the solver runs the
instruments — so the model is `mockllm/model` and the run costs $0.
Logs land in `./logs/` (one timestamped `.eval` file per run).

## Browse the results in Inspect View

```bash
~/workspace/tooling/inspect-venv/bin/inspect view --log-dir ./logs
```

Then open the printed URL (default http://127.0.0.1:7575). Each sample
is one matrix cell (`<condition>::<instrument>`); its transcript shows
the JSON record the solver produced — the `detects` flag plus the
instrument's key measurements (scoring max error, probing recovered
FBR, atlas property values, anatomy decoded FBR / active units /
issues) — and its score says whether that cell reproduced the
original audit's matrix.

## Tests

```bash
cd experiments/weight-tamper-audit/inspect_port
~/workspace/tooling/inspect-venv/bin/python -m pytest test_unit.py test_integration.py
```

- `test_unit.py` — the dataset/scorer mapping: the 4×4 sample grid,
  targets and metadata against `../results/audit.json`, the
  measurement extractors, and the scorer's comparison function.
- `test_integration.py` — runs the Inspect eval end-to-end via the
  `inspect_ai` eval API with `mockllm/model` and asserts all 16
  samples reproduce the detection matrix in `../results/audit.json`
  exactly.
