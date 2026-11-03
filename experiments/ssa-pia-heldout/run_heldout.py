"""Held-out runner: SSA PIA from regulatory text only.

Order of operations (per PROTOCOL.md):
1. Build the in-session control construction and record its
   SHA-256 in results/FREEZE.json BEFORE any API call.
2. Call GPT-5.6 Sol and Gemini 3.1 Pro via OpenRouter with the
   verbatim SSA text (SOURCE_TEXT.md) + fixed network spec.
   One syntax-repair retry per model, parse error only.
3. Actual per-call cost from API usage accounting; cumulative
   spend checked before every call; hard cap $0.50 (Cory's
   authorized estimate range for the final experiments).

Credentials via the custom.openrouter surrogate; never printed.
"""
from __future__ import annotations

import hashlib
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
CAP_USD = 0.50
MAX_TOKENS = 20_000
MODELS = ["openai/gpt-5.6-sol", "google/gemini-3.1-pro-preview"]

SSA_TEXT = """SOURCE TEXT (Social Security Administration, "Primary Insurance Amount", retrieved 2026-11-03 from https://www.ssa.gov/oact/cola/piaformula.html):

"The PIA is the sum of three separate percentages of portions of average indexed monthly earnings. The portions depend on the year in which a worker attains age 62, becomes disabled before age 62, or dies before attaining age 62.

For 2027 these portions are the first $1,340, the amount between $1,340 and $8,077, and the amount over $8,077. These dollar amounts are the "bend points" of the 2027 PIA formula.

For an individual who first becomes eligible for old-age insurance benefits or disability insurance benefits in 2027, or who dies in 2027 before becoming eligible for benefits, his/her PIA will be the sum of: (a) 90 percent of the first $1,340 of his/her average indexed monthly earnings, plus (b) 32 percent of his/her average indexed monthly earnings over $1,340 and through $8,077, plus (c) 15 percent of his/her average indexed monthly earnings over $8,077. We round this amount to the next lower multiple of $.10 if it is not already a multiple of $.10."
"""

TASK_INSTRUCTION = """
TASK: Write the weights of a neural network that computes the sum (a)+(b)+(c) above — the primary insurance amount for a worker first eligible in 2027, as a function of average indexed monthly earnings x (dollars). Implement the sum ONLY: do NOT implement the final rounding sentence (that rounding is the subject of a separate experiment). Domain: x in [0, 20000].
"""

FORMAT_SPEC = """
NETWORK SPEC (fixed architecture; scaling is part of the architecture, not the weights):
- Forward pass: for each layer i, h = W_i @ h_prev + b_i, followed by ReLU after every layer except the last. W_i has shape [out_features, in_features]; entry "r,c" multiplies input component c into output unit r.
- Inputs: [x/10000] (x = average indexed monthly earnings in dollars). Layers: W1 [16,1], b1 [16], then ReLU, then W2 [1,16], b2 [1] (final layer linear). Output dollars = 1000 * (final linear output). Arrays: W1, b1, W2, b2.
- Return the weights as ONE JSON object, no markdown, no commentary:
  {"arrays": {"W1": {"shape": [rows, cols], "entries": {"r,c": number, ...}}, "b1": {"shape": [n], "entries": {"i": number, ...}}, ...}}
  List ONLY nonzero entries; every unlisted entry is 0. Include every array named in the architecture, even if all its entries are 0 (then "entries": {}).
- Work in the scaled units the architecture defines. Derive the weights by reasoning about the formula. Do not approximate: the network will be checked against the formula on tens of thousands of inputs and must match to the cent everywhere.
"""


def build_control() -> Path:
    dest = HERE / "weights" / "claude-control.npz"
    W1 = np.zeros((16, 1)); b1 = np.zeros(16)
    W2 = np.zeros((1, 16)); b2 = np.zeros(1)
    # h0 = relu(t), h1 = relu(t - 0.134), h2 = relu(t - 0.8077), t = x/10000
    # PIA = 0.9x - 0.58*max(0, x-1340) - 0.17*max(0, x-8077)
    # output = 1000 * (9*h0 - 5.8*h1 - 1.7*h2)
    W1[0, 0] = 1.0; b1[0] = 0.0
    W1[1, 0] = 1.0; b1[1] = -0.134
    W1[2, 0] = 1.0; b1[2] = -0.8077
    W2[0, 0] = 9.0; W2[0, 1] = -5.8; W2[0, 2] = -1.7
    np.savez(dest, W1=W1, b1=b1, W2=W2, b2=b2)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    freeze = {"control_file": dest.name, "sha256": digest,
              "built_before_api_calls": True,
              "note": "Control construction hash-frozen before writer weights were scored."}
    (HERE / "results" / "FREEZE.json").write_text(json.dumps(freeze, indent=2))
    print(f"control built + frozen: sha256 {digest[:16]}...", flush=True)
    return dest


def call_api(model: str, prompt: str) -> dict:
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": MAX_TOKENS,
               "usage": {"include": True}}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "HTTP-Referer": "https://github.com/ctrimm/weight-space-programming",
                 "X-Title": "weight-space-programming ssa-pia-heldout"})
    add_surrogate_to_request(req, "custom.openrouter", allowed_hosts=["openrouter.ai"])
    with urllib.request.urlopen(req, timeout=600) as resp:
        return read_json_response(resp)


def parse_weights(text: str) -> dict[str, np.ndarray]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found in reply")
    spec = json.loads(m.group(0))
    arrays = spec["arrays"]
    out = {}
    for name in ("W1", "b1", "W2", "b2"):
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
    build_control()
    prompt = SSA_TEXT + TASK_INSTRUCTION + FORMAT_SPEC
    (HERE / "results" / "prompt.txt").write_text(prompt)

    log_path = HERE / "results" / "cost_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"calls": [], "total_usd": 0.0}
    spent = float(log["total_usd"])

    for model in MODELS:
        mslug = model.split("/")[-1]
        dest = HERE / "weights" / f"{mslug}__pia.npz"
        if dest.exists():
            print(f"skip (done): {model}", flush=True)
            continue
        p = prompt
        for attempt in (1, 2):
            if spent >= CAP_USD:
                print(f"HARD CAP ${CAP_USD} reached (${spent:.4f}); aborting before {model} attempt {attempt}", flush=True)
                log_path.write_text(json.dumps(log, indent=2))
                return
            t0 = time.time()
            try:
                resp = call_api(model, p)
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:500]
                print(f"HTTP {e.code} for {model} attempt {attempt}: {body}", flush=True)
                log["calls"].append({"model": model, "attempt": attempt,
                                     "error": f"HTTP {e.code}: {body}", "cost_usd": 0.0})
                log_path.write_text(json.dumps(log, indent=2))
                break
            usage = resp.get("usage", {}) or {}
            cost = float(usage.get("cost", 0.0) or 0.0)
            spent += cost
            log["total_usd"] = spent
            text = resp["choices"][0]["message"]["content"] or ""
            raw_path = HERE / "raw" / f"{mslug}__pia__attempt{attempt}.txt"
            raw_path.write_text(text)
            entry = {"model": model, "attempt": attempt, "cost_usd": cost,
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"),
                     "seconds": round(time.time() - t0, 1), "raw_file": raw_path.name}
            try:
                w = parse_weights(text)
                np.savez(dest, **w)
                entry["parse"] = "ok"
                entry["nonzero"] = int(sum(int(np.count_nonzero(v)) for v in w.values()))
                entry["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
                print(f"{model} attempt {attempt}: parsed, {entry['nonzero']} nonzero, ${cost:.4f} (total ${spent:.4f})", flush=True)
                log["calls"].append(entry)
                log_path.write_text(json.dumps(log, indent=2))
                break
            except Exception as exc:
                entry["parse"] = f"failed: {exc}"
                print(f"{model} attempt {attempt}: PARSE FAILED ({exc}), ${cost:.4f} (total ${spent:.4f})", flush=True)
                log["calls"].append(entry)
                log_path.write_text(json.dumps(log, indent=2))
                if attempt == 1:
                    p = prompt + f"\nIMPORTANT: a previous reply could not be parsed ({exc}). Reply with ONLY the JSON object, no other text."
                continue
    print(f"DONE. total measured spend: ${spent:.4f} of ${CAP_USD} cap", flush=True)
    log_path.write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
