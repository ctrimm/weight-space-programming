"""Second-writer pilot runner. Calls GPT-5.6 Sol and Gemini 3.1 Pro
via OpenRouter, one call per experiment (one syntax-repair retry
allowed), sparse-JSON weights in reply. Actual per-call cost is read
from the API usage accounting; cumulative spend is checked before
every call and the run aborts at the $10 hard cap Cory set.

Credentials: injected via the custom.openrouter surrogate (secure
vault), never read or printed here.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response  # noqa: E402

HERE = Path(__file__).parent
CAP_USD = 10.00
MAX_TOKENS = 20_000
MODELS = ["openai/gpt-5.6-sol", "google/gemini-3.1-pro-preview"]

FORMAT_SPEC = """
NETWORK SPEC (fixed architecture; scaling is part of the architecture, not the weights):
- Forward pass: for each layer i, h = W_i @ h_prev + b_i, followed by ReLU after every layer except the last. W_i has shape [out_features, in_features]; entry "r,c" multiplies input component c into output unit r.
- {arch}
- Return the weights as ONE JSON object, no markdown, no commentary:
  {{"arrays": {{"W1": {{"shape": [rows, cols], "entries": {{"r,c": number, ...}}}}, "b1": {{"shape": [n], "entries": {{"i": number, ...}}}}, ...}}}}
  List ONLY nonzero entries; every unlisted entry is 0. Include every array named in the architecture, even if all its entries are 0 (then "entries": {{}}).
- Work in the scaled units the architecture defines. Derive the weights by reasoning about the formula. Do not approximate: the network will be checked against the formula on tens of thousands of inputs and must match to the cent everywhere.
"""

TASKS = {
    "eitc": {
        "arch": "Inputs: [x/10000] (x = annual earned income in dollars). Layers: W1 [16,1], b1 [16], then ReLU, then W2 [1,16], b2 [1] (final layer linear). Output dollars = 1000 * (final linear output). Arrays: W1, b1, W2, b2.",
        "rule": """FORMULA (US Earned Income Tax Credit, tax year 2025, single/head-of-household filer, 2 qualifying children; earned income x in dollars, stipulate AGI = earned income):
- if x <= 17880: credit = 0.40 * x
- if 17880 < x <= 23350: credit = 7152
- if 23350 < x < 57310: credit = 7152 - 0.2106 * (x - 23350)
- if x >= 57310: credit = 0
Domain: x in [0, 70000].""",
    },
    "snap": {
        "arch": "Inputs: [E/1000, S/1000] (E = monthly gross earned income, S = monthly shelter cost, dollars). Layers: W1 [24,2], b1 [24], ReLU, W2 [24,24], b2 [24], ReLU, W3 [1,24], b3 [1] (final layer linear). Output dollars = 1000 * (final linear output). Arrays: W1, b1, W2, b2, W3, b3.",
        "rule": """FORMULA (US SNAP monthly allotment, FY2026, household of 2, earned income only, no elderly/disabled member):
- Adjusted income A = 0.80 * E - 209
- Excess shelter X = max(0, S - 0.5 * A); capped: Xc = min(X, 744)
- Net income N = max(0, A - Xc)
- Allotment B = max(0, 546 - 0.30 * N)
Domain: E in [300, 4000], S in [0, 2500]. No eligibility gates, no rounding.""",
    },
    "ssi": {
        "arch": "Inputs: [E/1000, U/1000] (E = monthly earned income, U = monthly unearned income, dollars). Layers: W1 [24,2], b1 [24], ReLU, W2 [24,24], b2 [24], ReLU, W3 [24,24], b3 [24], ReLU, W4 [1,24], b4 [1] (final layer linear). Output dollars = 1000 * (final linear output). Arrays: W1, b1, W2, b2, W3, b3, W4, b4.",
        "rule": """FORMULA (US Supplemental Security Income federal monthly payment, individual, 2026):
- Countable unearned income CU = max(0, U - 20)  (the $20 general exclusion applies to unearned income first)
- Unused exclusion spills to earned income: R = max(0, 20 - U)
- Countable earned income CE = max(0, E - 65 - R) / 2  ($65 exclusion plus the spill, then one-half disregarded)
- Countable income C = CU + CE
- Payment P = max(0, 994 - C)
Domain: E in [0, 3000], U in [0, 1500]. Federal payment only, no rounding.""",
    },
    "medicaid": {
        "arch": "Inputs: [I/1000, SI/1000, P/1000] (I = resident's gross monthly income, SI = community spouse's own monthly income, P = resident's monthly health insurance premiums, dollars). Layers: W1 [24,3], b1 [24], ReLU, W2 [24,24], b2 [24], ReLU, W3 [1,24], b3 [1] (final layer linear). Output dollars = 1000 * (final linear output). Arrays: W1, b1, W2, b2, W3, b3.",
        "rule": """FORMULA (US Medicaid post-eligibility patient liability for a nursing-facility resident with a community spouse, monthly):
- Community spouse income allowance A = max(0, 2643.75 - SI)
- Liability L = max(0, I - 50 - A - P)   ($50 personal needs allowance, then the spousal allowance, then premiums)
Domain: I in [0, 6000], SI in [0, 5000], P in [0, 600]. No other deductions or caps.""",
    },
}
ORDER = ["eitc", "snap", "medicaid", "ssi"]


def call_api(model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "HTTP-Referer": "https://github.com/ctrimm/weight-space-programming",
                 "X-Title": "weight-space-programming second-writer pilot"},
    )
    add_surrogate_to_request(req, "custom.openrouter", allowed_hosts=["openrouter.ai"])
    with urllib.request.urlopen(req, timeout=600) as resp:
        return read_json_response(resp)


def parse_weights(text: str, task: str) -> dict[str, np.ndarray]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found in reply")
    spec = json.loads(m.group(0))
    arrays = spec["arrays"]
    want = {}
    arch = TASKS[task]["arch"]
    for name in re.findall(r"\b[Wb]\d\b", arch):
        want.setdefault(name, None)
    out: dict[str, np.ndarray] = {}
    for name in want:
        if name not in arrays:
            raise ValueError(f"missing array {name}")
        shape = arrays[name]["shape"]
        arr = np.zeros(shape, dtype=np.float64)
        for key, val in arrays[name].get("entries", {}).items():
            idx = tuple(int(p) for p in key.split(","))
            if len(idx) != len(shape):
                idx = idx + (0,) * (len(shape) - len(idx)) if len(shape) == 1 else idx
            arr[idx] = float(val)
        out[name] = arr
    return out


def main() -> None:
    log_path = HERE / "results" / "cost_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"calls": [], "total_usd": 0.0}
    spent = float(log["total_usd"])

    for model in MODELS:
        mslug = model.split("/")[-1]
        for task in ORDER:
            dest = HERE / "weights" / f"{mslug}__{task}.npz"
            if dest.exists():
                print(f"skip (done): {model} {task}", flush=True)
                continue
            prompt = TASKS[task]["rule"] + "\n" + FORMAT_SPEC.format(arch=TASKS[task]["arch"])
            for attempt in (1, 2):
                if spent >= CAP_USD:
                    print(f"HARD CAP ${CAP_USD} reached (${spent:.4f}); aborting before {model} {task}", flush=True)
                    log_path.write_text(json.dumps(log, indent=2))
                    return
                t0 = time.time()
                try:
                    resp = call_api(model, prompt)
                except urllib.error.HTTPError as e:
                    body = e.read().decode()[:500]
                    print(f"HTTP {e.code} for {model} {task} attempt {attempt}: {body}", flush=True)
                    log["calls"].append({"model": model, "task": task, "attempt": attempt,
                                         "error": f"HTTP {e.code}: {body}", "cost_usd": 0.0})
                    log_path.write_text(json.dumps(log, indent=2))
                    break
                usage = resp.get("usage", {}) or {}
                cost = float(usage.get("cost", 0.0) or 0.0)
                spent += cost
                log["total_usd"] = spent
                text = resp["choices"][0]["message"]["content"] or ""
                raw_path = HERE / "raw" / f"{mslug}__{task}__attempt{attempt}.txt"
                raw_path.write_text(text)
                entry = {"model": model, "task": task, "attempt": attempt, "cost_usd": cost,
                         "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                         "seconds": round(time.time() - t0, 1), "raw_file": raw_path.name}
                try:
                    w = parse_weights(text, task)
                    np.savez(dest, **w)
                    entry["parse"] = "ok"
                    entry["nonzero"] = int(sum(int(np.count_nonzero(v)) for v in w.values()))
                    print(f"{model} {task} attempt {attempt}: parsed, {entry['nonzero']} nonzero, ${cost:.4f} (total ${spent:.4f})", flush=True)
                    log["calls"].append(entry)
                    log_path.write_text(json.dumps(log, indent=2))
                    break
                except Exception as exc:  # parse failure -> one syntax-repair retry
                    entry["parse"] = f"failed: {exc}"
                    print(f"{model} {task} attempt {attempt}: PARSE FAILED ({exc}), ${cost:.4f} (total ${spent:.4f})", flush=True)
                    log["calls"].append(entry)
                    log_path.write_text(json.dumps(log, indent=2))
                    if attempt == 1:
                        prompt = (TASKS[task]["rule"] + "\n" + FORMAT_SPEC.format(arch=TASKS[task]["arch"])
                                  + f"\nIMPORTANT: a previous reply could not be parsed ({exc}). Reply with ONLY the JSON object, no other text.")
                    continue
    print(f"DONE. total measured spend: ${spent:.4f} of ${CAP_USD} cap", flush=True)
    log_path.write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
