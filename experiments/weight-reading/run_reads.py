"""Arm A: blind weight reads by GPT-5.6 Sol and Gemini 3.1 Pro via
OpenRouter. Prompts are generated ONLY from the mystery artifact
files + framing text (architecture, IO semantics, weights). Ground
truth mapping is never read by this script beyond family framing —
which is keyed by artifact code ranges defined here from the pack
layout the experimenter published in PROTOCOL.md (framing is part of
the audit scenario: the auditor knows what the inputs mean).

Actual cost from API usage accounting; abort at $6 cumulative.
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
CAP_USD = 6.00
MAX_TOKENS = 24_000
READERS = ["openai/gpt-5.6-sol", "google/gemini-3.1-pro-preview"]
ASSIGNMENTS = {
    "openai/gpt-5.6-sol": ["N03", "N04", "N09", "N10", "N11"],
    "google/gemini-3.1-pro-preview": ["N01", "N05", "N07", "N10", "N11"],
}
FAMILY = {  # framing per artifact code (published in PROTOCOL.md)
    "N01": "snap", "N02": "snap", "N03": "snap",
    "N04": "ssi", "N05": "ssi", "N06": "ssi",
    "N07": "medicaid", "N08": "medicaid", "N09": "medicaid",
    "N10": "ssi_defect", "N11": "ssi",
}
FRAMING = {
    "snap": ("Inputs (dollars, before scaling): E = monthly gross earned income, "
             "S = monthly shelter cost. The network receives [E/1000, S/1000] and its "
             "output, times 1000, is a monthly benefit amount in dollars. "
             "Domain: E in [300, 4000], S in [0, 2500]. Input variables for your formula: E, S."),
    "ssi": ("Inputs (dollars, before scaling): E = monthly earned income, "
            "U = monthly unearned income. The network receives [E/1000, U/1000] and its "
            "output, times 1000, is a monthly payment in dollars. "
            "Domain: E in [0, 3000], U in [0, 1500]. Input variables for your formula: E, U."),
    "medicaid": ("Inputs (dollars, before scaling): I = a nursing-facility resident's gross "
                 "monthly income, SI = the resident's community spouse's own monthly income, "
                 "P = the resident's monthly health insurance premiums. The network receives "
                 "[I/1000, SI/1000, P/1000] and its output, times 1000, is the monthly amount "
                 "in dollars the resident owes. Domain: I in [0, 6000], SI in [0, 5000], "
                 "P in [0, 600]. Input variables for your formula: I, SI, P."),
}
LAW_SHEET = """

AUDIT CONTEXT: you are auditing this artifact against current law.
Current law for this program (2026): the federal benefit rate for an
individual is $994/month; a $20 monthly general income exclusion
applies to unearned income first and any unused part applies to
earned income; a $65 monthly earned income exclusion applies after
that; one-half of remaining earned income is disregarded; the
payment is the benefit rate minus total countable income, floored
at $0. In "anomalies", flag every constant you recover from the
weights that disagrees with this current-law sheet.
"""
ANSWER_SPEC = """
Analyze the weights and determine what function this network computes.
Reply with ONE JSON object, no markdown, no commentary:
{"program": "<name the program/rule you believe this computes>",
 "formula_text": "<the full formula in plain notation, in dollar terms>",
 "constants": {"<constant name>": <value>, ...},
 "formula_python": "<ONE Python expression over the input variables named above, using only numbers, max, min, abs, +, -, *, /, parentheses; it takes dollar inputs and returns dollars>",
 "confidence": <0.0 to 1.0>,
 "anomalies": ["<anything in the weights that looks wrong, stale, or inconsistent>", ...]}
The forward pass is: for each layer i, h = W_i @ h_prev + b_i, with ReLU after every layer except the last. W_i entry [r][c] multiplies input component c into unit r. Weight values below are rounded to 6 decimal places.
"""


def weights_json(code: str) -> str:
    with np.load(HERE / "mystery" / f"{code}.npz") as z:
        w = {k: np.round(z[k].astype(np.float64), 6).tolist() for k in sorted(z.files)}
    return json.dumps(w)


def arch_text(code: str) -> str:
    with np.load(HERE / "mystery" / f"{code}.npz") as z:
        parts = []
        i = 1
        while f"W{i}" in z.files:
            parts.append(f"W{i} shape {list(z[f'W{i}'].shape)}, b{i} shape {list(z[f'b{i}'].shape)}")
            i += 1
    return "Architecture: " + "; ".join(parts) + "."


def build_prompt(code: str) -> str:
    fam = FAMILY[code]
    base_fam = "ssi" if fam == "ssi_defect" else fam
    p = (f"You are given the complete weights of a small neural network (artifact {code}) "
         f"with no code and no documentation.\n\n{FRAMING[base_fam]}\n\n{arch_text(code)}\n")
    if fam == "ssi_defect":
        p += LAW_SHEET
    p += ANSWER_SPEC
    p += "\nWEIGHTS (JSON):\n" + weights_json(code)
    return p


def call_api(model: str, prompt: str) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": MAX_TOKENS, "usage": {"include": True}}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "HTTP-Referer": "https://github.com/ctrimm/weight-space-programming",
                 "X-Title": "weight-space-programming weight-reading arm A"})
    add_surrogate_to_request(req, "custom.openrouter", allowed_hosts=["openrouter.ai"])
    with urllib.request.urlopen(req, timeout=900) as resp:
        return read_json_response(resp)


def main() -> None:
    log_path = HERE / "results" / "cost_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"calls": [], "total_usd": 0.0}
    spent = float(log["total_usd"])
    for model in READERS:
        mslug = model.split("/")[-1]
        for code in ASSIGNMENTS[model]:
            claim_path = HERE / "claims" / f"{mslug}__{code}__armA.json"
            raw_path = HERE / "raw" / f"{mslug}__{code}__armA.txt"
            if raw_path.exists():
                print(f"skip (done): {model} {code}", flush=True)
                continue
            if spent >= CAP_USD:
                print(f"HARD CAP ${CAP_USD} reached (${spent:.4f}); aborting before {model} {code}", flush=True)
                log_path.write_text(json.dumps(log, indent=2))
                return
            t0 = time.time()
            try:
                resp = call_api(model, build_prompt(code))
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:400]
                print(f"HTTP {e.code} {model} {code}: {body}", flush=True)
                log["calls"].append({"model": model, "artifact": code, "arm": "A",
                                     "error": f"HTTP {e.code}: {body}", "cost_usd": 0.0})
                log_path.write_text(json.dumps(log, indent=2))
                continue
            usage = resp.get("usage", {}) or {}
            cost = float(usage.get("cost", 0.0) or 0.0)
            spent += cost
            log["total_usd"] = spent
            text = resp["choices"][0]["message"]["content"] or ""
            raw_path.write_text(text)
            entry = {"model": model, "artifact": code, "arm": "A", "cost_usd": cost,
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"),
                     "seconds": round(time.time() - t0, 1)}
            m = re.search(r"\{.*\}", text, re.DOTALL)
            try:
                claim = json.loads(m.group(0)) if m else (_ for _ in ()).throw(ValueError("no JSON"))
                claim_path.write_text(json.dumps(claim, indent=2))
                entry["parse"] = "ok"
                print(f"{model} {code}: claim parsed (program: {claim.get('program')!r}, "
                      f"conf {claim.get('confidence')}), ${cost:.4f} (total ${spent:.4f})", flush=True)
            except Exception as exc:
                entry["parse"] = f"failed: {exc}"
                print(f"{model} {code}: PARSE FAILED ({exc}), ${cost:.4f} (total ${spent:.4f})", flush=True)
            log["calls"].append(entry)
            log_path.write_text(json.dumps(log, indent=2))
    print(f"ARM A DONE. total measured spend: ${spent:.4f} of ${CAP_USD} cap", flush=True)


if __name__ == "__main__":
    main()
